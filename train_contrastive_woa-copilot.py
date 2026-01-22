import os
import re
import math
import random
from typing import List, Dict, Tuple

import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader

# ======================
#  配置部分
# ======================

class Config:
    # 路径相关
    WOA_LOG_CSV = r"E:\rcq\pythonProject\Data\ShipsEar_16k_30s_hop15_WOA\woa_augmentation_log.csv"
    # 如果你的 log 不在这个路径，改成实际路径即可

    # 采样率 & 时长
    SAMPLE_RATE = 16000
    MAX_DURATION_SEC = 10.0   # 每个样本最多用 10s，可按需改长/改短
    MONO = True

    # 训练相关
    BATCH_SIZE = 16           # 实际 batch size = 16 identity，每个 identity 有两个 view
    MAX_NUM_WORKERS = 4
    TORCH_NUM_THREADS = min(4, os.cpu_count() or 1)
    TORCH_INTEROP_THREADS = 1
    DATA_LOADER_TIMEOUT = 120
    GPU_NUM_WORKERS = 0
    N_EPOCHS = 100
    LR = 1e-3
    LR_MIN = 1e-5
    WEIGHT_DECAY = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LOG_INTERVAL = 20
    CKPT_PATH = "./encoder_contrastive_woa.pt"

    # 对比学习
    CONTRASTIVE_TEMPERATURE = 0.1
    USE_SUPERVISED_CONTRASTIVE = True  # True 时利用类别标签做 supervised contrastive
    GRAD_CLIP_NORM = 1.0

    # 编码器结构
    DYN_HIDDEN_CHANNELS = 128
    CNN_BASE_DIVISOR = 4
    CNN_MID_DIVISOR = 2
    CNN_MIN_BASE_CHANNELS = 16
    CNN_MIN_MID_CHANNELS = 32

    # 投影头维度
    EMBED_DIM = 128
    PROJ_DIM = 128

    DROPOUT = 0.1

    # 数值稳定性
    MASK_VALUE_LIMIT = -1e9
    MIN_TEMPERATURE = 1e-6
    NORMALIZE_EPS = 1e-6
    MIN_SE_HIDDEN = 8
    WAVEFORM_MAX_ABS = 1.0
    MAX_NONFINITE_WARNINGS = 5


cfg = Config()

# ======================
#  工具函数
# ======================

def fix_path(path: str) -> str:
    """把 Windows 下的反斜杠路径安全地转成当前 OS 可读路径。"""
    return os.path.normpath(path)

def set_seed(seed: int = 2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(2026)
torch.set_num_threads(cfg.TORCH_NUM_THREADS)
torch.set_num_interop_threads(cfg.TORCH_INTEROP_THREADS)


# ======================
#  数据集定义：从 woa_augmentation_log.csv 构造对比学习 batch
# ======================

class WOAContrastiveDataset(Dataset):
    """
    每个 __getitem__ 返回：
      - wav1: Tensor [1, T]
      - wav2: Tensor [1, T]
      - label: int (类别 label，用于 optional supervised contrastive)
    每个索引对应一个 identity（如 ClassA_dredger_id80），而非单条样本。
    """
    def __init__(
        self,
        csv_path: str,
        split_prefix: str = "train",   # 根据 rel_path 是否以 'train' 开头筛选训练
        min_samples_per_id: int = 2,
        sample_rate: int = 16000,
        max_duration_sec: float = 10.0,
    ):
        super().__init__()
        self._warned_paths: set = set()
        self._sr_ratio_cache: Dict[int, float] = {}
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")
        self.sample_rate = sample_rate
        self.max_len = int(max_duration_sec * sample_rate)

        csv_path = fix_path(csv_path)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"WOA log CSV not found: {csv_path}")

        # 读取 CSV，自动识别分隔符
        self.df = pd.read_csv(csv_path, engine="python")
        # 如果分隔是制表符，可以改为:
        # self.df = pd.read_csv(csv_path, sep='\t')

        required_cols = {"rel_path", "out_path"}
        missing_cols = required_cols.difference(self.df.columns)
        if missing_cols:
            missing = ", ".join(sorted(missing_cols))
            raise ValueError(f"Missing required columns in CSV: {missing}")

        # 过滤出对应 split（如 train）
        # 这里根据 rel_path 的前缀简单判断；若你在 log 中有单独的 split 列，可改用 split 列
        mask = self.df["rel_path"].astype(str).str.replace("\\", "/").str.startswith(split_prefix)
        self.df = self.df[mask].reset_index(drop=True)

        # 提取 identity 和 class label
        self.df["identity"], self.df["class_label"] = zip(*self.df["rel_path"].map(self._parse_identity_and_class))

        # 只保留拥有至少 `min_samples_per_id` 条数据的 identity
        counts = self.df["identity"].value_counts()
        valid_ids = set(counts[counts >= min_samples_per_id].index)
        self.df = self.df[self.df["identity"].isin(valid_ids)].reset_index(drop=True)

        # 为每个 identity 建立索引列表
        self.id_to_indices: Dict[str, List[int]] = {}
        self.id_to_class: Dict[str, int] = {}
        for idx, row in self.df.iterrows():
            _id = row["identity"]
            cls = row["class_label"]
            if _id not in self.id_to_indices:
                self.id_to_indices[_id] = []
                self.id_to_class[_id] = cls
            self.id_to_indices[_id].append(idx)

        # identity 列表（Dataset 的索引）
        self.identities: List[str] = sorted(self.id_to_indices.keys())

        # 建立 class_label -> 连续整数
        # 比如 ClassA, ClassB, ... -> 0,1,...
        unique_classes = sorted(set(self.id_to_class[_id] for _id in self.identities))
        self.class_map = {c: i for i, c in enumerate(unique_classes)}
        # 把 id_to_class 映射成真正的 int label
        for _id in self.identities:
            self.id_to_class[_id] = self.class_map[self.id_to_class[_id]]

        print(f"[WOAContrastiveDataset] loaded {len(self.df)} rows, {len(self.identities)} identities "
              f"with >= {min_samples_per_id} samples each.")

    def _parse_identity_and_class(self, rel_path: str) -> Tuple[str, str]:
        """
        从 rel_path 解析 identity 和 class_label。
        示例 rel_path:
          'train\\ClassA\\dredger\\train_ClassA_dredger_id80_seg000.wav'
        我们希望：
          identity = 'ClassA_dredger_id80'
          class_label = 'ClassA'（或者 'ClassA_dredger' 也可以，看你想怎么定义类别）
        """
        rel_path = rel_path.replace("\\", "/")
        base = os.path.basename(rel_path)  # train_ClassA_dredger_id80_seg000.wav
        # 按 '_' 切分
        parts = base.split("_")
        # 假设命名格式固定：train_ClassA_dredger_id80_seg000.wav
        # 对应 parts = ['train', 'ClassA', 'dredger', 'id80', 'seg000.wav']
        if len(parts) < 5:
            # fallback：直接用文件名
            identity = base
            class_label = "Unknown"
            return identity, class_label

        class_name = parts[1]          # ClassA
        subtype = parts[2]             # dredger
        id_part = parts[3]             # id80
        # 作为 identity-key
        identity = f"{class_name}_{subtype}_{id_part}"

        # 类别标签你可以按需选择：只用 ClassA，或者 ClassA + subtype
        class_label = f"{class_name}_{subtype}"  # 例如 ClassA_dredger
        return identity, class_label

    def __len__(self):
        return len(self.identities)

    def _load_and_crop(self, wav_path: str) -> torch.Tensor:
        wav_path = fix_path(wav_path)
        try:
            info = torchaudio.info(wav_path)
            sr = info.sample_rate
            num_frames = info.num_frames
            if num_frames is None:
                raise RuntimeError(
                    f"Audio file metadata incomplete: missing num_frames for {wav_path}. "
                    "File may be corrupted or in an unsupported format."
                )
            target_frames = self.max_len
            if sr != self.sample_rate:
                key = (sr, self.sample_rate)
                ratio = self._sr_ratio_cache.get(key)
                if ratio is None:
                    ratio = self.sample_rate / sr
                    self._sr_ratio_cache[key] = ratio
                target_frames = math.ceil(self.max_len / ratio)
            if num_frames > target_frames:
                start = random.randint(0, num_frames - target_frames)
                wav, sr = torchaudio.load(wav_path, frame_offset=start, num_frames=target_frames)
            else:
                wav, sr = torchaudio.load(wav_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load audio: {wav_path}. {exc}") from exc
        wav = wav.to(torch.float32)
        if self.MONO:
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
        # 重采样（如果需要）
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=self.sample_rate)

        # 裁剪或补零到固定长度
        T = wav.shape[1]
        if T > self.max_len:
            # 随机裁剪一段
            start = random.randint(0, T - self.max_len)
            wav = wav[:, start:start + self.max_len]
        elif T < self.max_len:
            pad_len = self.max_len - T
            wav = F.pad(wav, (0, pad_len))

        if torch.any(~torch.isfinite(wav)):
            if wav_path not in self._warned_paths and len(self._warned_paths) < cfg.MAX_NONFINITE_WARNINGS:
                print(f"[WARN] Non-finite waveform values detected: {wav_path}. Replacing with zeros.")
                self._warned_paths.add(wav_path)
            wav = torch.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)

        max_abs = wav.abs().max()
        if max_abs > cfg.WAVEFORM_MAX_ABS:
            wav = wav / (max_abs + cfg.NORMALIZE_EPS) * cfg.WAVEFORM_MAX_ABS

        return wav

    @property
    def MONO(self):
        return cfg.MONO

    def __getitem__(self, index):
        # index 是 identity 的索引
        identity = self.identities[index]
        indices = self.id_to_indices[identity]
        cls = self.id_to_class[identity]

        # 从该 identity 中随机选两条不同的样本，作为两个 view
        if len(indices) >= 2:
            idx1, idx2 = random.sample(indices, 2)
        else:
            # 理论上不会到这里，因为我们保证了 min_samples_per_id >= 2
            idx1 = idx2 = indices[0]

        row1 = self.df.iloc[idx1]
        row2 = self.df.iloc[idx2]

        path1 = row1["out_path"]  # 使用已经 WOA 增强后的音频
        path2 = row2["out_path"]

        wav1 = self._load_and_crop(path1)  # [1, T]
        wav2 = self._load_and_crop(path2)

        return wav1, wav2, cls


# ======================
#  SincConv 前端（简化版 SincNet）
# ======================

class SincConv1d(nn.Module):
    """
    基于 Sinc 的 1D 卷积核（物理引导滤波器），
    这里是一个常见实现的简化版：学习每个 band-pass 的低/高截止频率，
    在时间域构造 sinc 滤波器再做 conv1d。
    """
    def __init__(self, out_channels, kernel_size, sample_rate,
                 min_low_hz=50, min_band_hz=50):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # 初始化频率
        low_hz = np.linspace(min_low_hz,
                             sample_rate / 2 - (min_low_hz + min_band_hz),
                             out_channels)
        band_hz = np.full(out_channels, min_band_hz)

        self.low_hz = nn.Parameter(torch.tensor(low_hz, dtype=torch.float32))
        self.band_hz = nn.Parameter(torch.tensor(band_hz, dtype=torch.float32))

        # Hamming 窗
        n_lin = torch.linspace(0, self.kernel_size - 1, steps=self.kernel_size)
        self.register_buffer("window", 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / (self.kernel_size - 1)))
        # 对称中心
        n_0 = (self.kernel_size - 1) / 2.0
        self.register_buffer("n_0", torch.tensor(n_0))
        self.register_buffer("n", torch.arange(self.kernel_size) - n_0)


    def forward(self, x):
        """
        x: [B, 1, T]
        输出: [B, out_channels, T_conv]
        """
        device = x.device
        low = self.min_low_hz + torch.abs(self.low_hz)            # 保证 > min_low_hz
        band = self.min_band_hz + torch.abs(self.band_hz)         # 保证 > min_band_hz
        high = low + band

        # 归一化频率
        low = low / (self.sample_rate / 2)
        high = high / (self.sample_rate / 2)

        # 构造时间轴
        n = self.n.to(device).unsqueeze(0)  # [1, kernel_size]
        low = low.unsqueeze(1)  # [out_channels, 1]
        high = high.unsqueeze(1)  # [out_channels, 1]

        # sinc band-pass 滤波器（向量化）
        band_pass = (2 * high * self._sinc(2 * high * n) -
                     2 * low * self._sinc(2 * low * n))
        band_pass = band_pass * self.window.to(device).unsqueeze(0)
        filters = band_pass.unsqueeze(1)  # [out_channels, 1, kernel_size]

        return F.conv1d(x, filters, stride=1, padding=self.kernel_size // 2)

    @staticmethod
    def _sinc(x):
        # sin(pi x) / (pi x)
        eps = 1e-8
        return torch.where(torch.abs(x) < eps, torch.ones_like(x), torch.sin(math.pi * x) / (math.pi * x))


# ======================
#  动态卷积模块（CondConv 风格）
# ======================

class DynamicConv1d(nn.Module):
    """
    简单的 Dynamic Conv:
      - 有 K 个基础卷积核
      - 根据输入的全局统计（avg-pool）生成 K 个 mixing 权重
      - 输出为 K 个 conv 结果按权重线性组合
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_kernels=4, stride=1, padding=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_kernels = num_kernels
        self.stride = stride
        if padding is None:
            padding = kernel_size // 2
        self.padding = padding

        # K 个基础卷积核
        self.weight = nn.Parameter(
            torch.randn(num_kernels, out_channels, in_channels, kernel_size) * 0.01
        )
        self.bias = nn.Parameter(torch.zeros(num_kernels, out_channels))

        # 路由网络：输入是 [B, in_channels] 的全局平均池化
        self.routing = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # [B, C, T] -> [B, C, 1]
            nn.Flatten(start_dim=1),  # [B, C]
            nn.Linear(in_channels, num_kernels),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        """
        x: [B, C_in, T]
        """
        B, C, T = x.shape
        # routing weights: [B, K]
        alpha = self.routing(x)  # 使用输入特征做权重
        # 对每个 kernel 做一次 conv，然后组合
        outs = []
        for k in range(self.num_kernels):
            w_k = self.weight[k]
            b_k = self.bias[k]
            y_k = F.conv1d(x, w_k, b_k, stride=self.stride, padding=self.padding)
            outs.append(y_k.unsqueeze(1))  # [B, 1, C_out, T_out]
        # stack -> [B, K, C_out, T_out]
        outs = torch.cat(outs, dim=1)
        # alpha: [B, K] -> [B, K, 1, 1]
        alpha = alpha.view(B, self.num_kernels, 1, 1)
        y = torch.sum(alpha * outs, dim=1)  # [B, C_out, T_out]
        return y


# ======================
#  辅助模块：残差卷积 & 通道注意力
# ======================

class TemporalResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size // 2) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(self.bn2(self.conv2(x)))
        return F.relu(x + residual)


def _calc_se_hidden(channels: int, reduction: int) -> int:
    return min(channels, max(cfg.MIN_SE_HIDDEN, channels // reduction))


class SqueezeExcite1d(nn.Module):
    def __init__(self, channels: int, reduction: int):
        super().__init__()
        hidden = _calc_se_hidden(channels, reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x):
        scale = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        scale = F.relu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale)).unsqueeze(-1)
        return x * scale


# ======================
#  Encoder: Sinc 前端 + TF-like block + Dynamic Conv + GlobalPool
# ======================

class PhyLDCEncoder(nn.Module):
    """
    轻量级 1D CNN Encoder（对比学习用）：
      - 多层下采样卷积 + BN + ReLU
      - Global pooling 输出 embedding
    """
    def __init__(self,
                 dyn_hidden_channels: int = 128,
                 embed_dim: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        if not 0 <= dropout <= 1:
            raise ValueError("dropout must be in [0, 1]")
        if dyn_hidden_channels <= 0:
            raise ValueError("dyn_hidden_channels must be positive")
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")

        base_channels = max(cfg.CNN_MIN_BASE_CHANNELS, dyn_hidden_channels // cfg.CNN_BASE_DIVISOR)
        mid_channels = max(cfg.CNN_MIN_MID_CHANNELS, dyn_hidden_channels // cfg.CNN_MID_DIVISOR)

        self.features = nn.Sequential(
            nn.Conv1d(1, base_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
            nn.Conv1d(base_channels, mid_channels, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(),
            nn.Conv1d(mid_channels, dyn_hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(dyn_hidden_channels),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(dyn_hidden_channels, embed_dim)

    def forward(self, x):
        """
        x: [B, 1, T] waveform
        输出: [B, embed_dim]
        """
        x = self.features(x)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)


# ======================
#  对比学习投影头 + 整体模型
# ======================

class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, proj_dim)
        )

    def forward(self, x):
        return self.net(x)


class ContrastiveModel(nn.Module):
    """
    整体对比学习模型：
      - encoder: PhyLDCEncoder
      - proj_head: ProjectionHead
    Stage 3 的时候，只需要加载 encoder 部分即可。
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.encoder = PhyLDCEncoder(
            dyn_hidden_channels=cfg.DYN_HIDDEN_CHANNELS,
            embed_dim=cfg.EMBED_DIM,
            dropout=cfg.DROPOUT
        )
        self.proj = ProjectionHead(cfg.EMBED_DIM, cfg.PROJ_DIM)

    def forward(self, x):
        """
        x: [B, 1, T]
        返回:
          z: encoder 的 embedding (Stage3 可用)
          p: 对比学习的投影后的向量
        """
        z = self.encoder(x)
        p = self.proj(z)
        return z, p


# ======================
#  对比损失 (NT-Xent / InfoNCE)
# ======================

def _validate_temperature(temperature: float) -> None:
    if temperature < cfg.MIN_TEMPERATURE:
        raise ValueError(f"temperature must be >= {cfg.MIN_TEMPERATURE}")


def _mask_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_floor = torch.finfo(logits.dtype).min
    mask_value = cfg.MASK_VALUE_LIMIT if mask_floor < cfg.MASK_VALUE_LIMIT else mask_floor
    return logits.masked_fill(mask, mask_value)


def contrastive_loss_nt_xent(z1, z2, temperature: float = 0.1):
    """
    SimCLR 风格的 NT-Xent loss（无标签版，只有 "同索引" 为正样本）。
    输入:
      z1, z2: [B, D]
    输出:
      标量 loss
    """
    _validate_temperature(temperature)
    batch_size = z1.size(0)

    # L2 normalize
    z1 = F.normalize(z1, dim=1, eps=cfg.NORMALIZE_EPS)
    z2 = F.normalize(z2, dim=1, eps=cfg.NORMALIZE_EPS)

    representations = torch.cat([z1, z2], dim=0)  # [2B, D]
    similarity_matrix = F.cosine_similarity(
        representations.unsqueeze(1), representations.unsqueeze(0), dim=-1
    )

    # 构造正样本 mask
    labels = torch.arange(batch_size, device=z1.device)
    labels = torch.cat([labels, labels], dim=0)   # [2B]

    # mask：对角线不算，且只把 (i, i+B) / (i+B, i) 当成正
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z1.device)
    # 正样本索引：i 与 i+batch_size
    pos_mask = torch.zeros_like(mask)
    for i in range(batch_size):
        pos_mask[i, i + batch_size] = True
        pos_mask[i + batch_size, i] = True

    # logits
    logits = _mask_logits(similarity_matrix / temperature, mask)  # 忽略自己

    # 对每个样本，只有一个正样本
    # log( exp(sim(pos)/temp) / sum(exp(sim(all)/temp)) )
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    pos_count = pos_mask.sum(dim=1)
    if torch.any(pos_count == 0):
        raise ValueError(
            "no positive samples found for some instances in the batch; "
            "increase batch size or check label distribution"
        )
    loss_pos = (pos_mask * log_prob).sum(dim=1) / pos_count  # [2B]
    loss = -loss_pos.mean()

    return loss


def contrastive_loss_supervised(z1, z2, labels, temperature: float = 0.1):
    """
    Supervised contrastive loss: use same-class samples as positives.
    输入:
      z1, z2: [B, D]
      labels: [B]
    """
    _validate_temperature(temperature)
    batch_size = z1.size(0)
    if labels.size(0) != batch_size:
        raise ValueError("labels size must match batch size")

    z = torch.cat([z1, z2], dim=0)
    z = F.normalize(z, dim=1, eps=cfg.NORMALIZE_EPS)
    labels = labels.repeat(2)

    logits = torch.matmul(z, z.t()) / temperature
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    logits = _mask_logits(logits, mask)

    pos_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
    pos_mask = pos_mask & ~mask
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    pos_count = pos_mask.sum(dim=1)
    if torch.any(pos_count == 0):
        raise ValueError(
            "no positive samples found for some instances in the batch; "
            "increase batch size or check label distribution"
        )
    loss_pos = (pos_mask * log_prob).sum(dim=1) / pos_count
    return -loss_pos.mean()


# ======================
#  训练循环
# ======================

def train_contrastive(cfg: Config):
    # Dataset & DataLoader
    csv_path = fix_path(cfg.WOA_LOG_CSV)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"WOA log CSV not found: {csv_path}")

    min_samples_per_id = 2
    dataset = WOAContrastiveDataset(
        csv_path=csv_path,
        split_prefix="train",
        min_samples_per_id=min_samples_per_id,
        sample_rate=cfg.SAMPLE_RATE,
        max_duration_sec=cfg.MAX_DURATION_SEC,
    )
    if len(dataset) == 0:
        raise ValueError(
            f"No valid identities found in CSV: {csv_path} "
            f"(requires min {min_samples_per_id} samples per identity)"
        )

    device = torch.device(cfg.DEVICE)
    print(f"Using device: {device}")
    cpu_workers = min(cfg.MAX_NUM_WORKERS, os.cpu_count() or 1)
    num_workers = cfg.GPU_NUM_WORKERS if device.type == "cuda" else cpu_workers

    loader = DataLoader(
        dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        timeout=cfg.DATA_LOADER_TIMEOUT if num_workers > 0 else 0
    )

    model = ContrastiveModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.N_EPOCHS, eta_min=cfg.LR_MIN)

    for epoch in range(1, cfg.N_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        processed_steps = 0
        skipped_steps = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{cfg.N_EPOCHS}")
        for step, (wav1, wav2, cls) in enumerate(pbar, start=1):
            wav1 = wav1.to(device)  # [B, 1, T]
            wav2 = wav2.to(device)

            _, z1 = model(wav1)   # (encoder_z1, proj_z1)，这里我们只要 proj 输出参与 loss
            _, z2 = model(wav2)

            if torch.any(~torch.isfinite(z1)) or torch.any(~torch.isfinite(z2)):
                print(f"[WARN] Non-finite embeddings at epoch {epoch}, step {step}, skip batch.")
                skipped_steps += 1
                continue

            if cfg.USE_SUPERVISED_CONTRASTIVE:
                loss = contrastive_loss_supervised(
                    z1, z2, cls.to(device), temperature=cfg.CONTRASTIVE_TEMPERATURE
                )
            else:
                loss = contrastive_loss_nt_xent(z1, z2, temperature=cfg.CONTRASTIVE_TEMPERATURE)

            optimizer.zero_grad()
            loss.backward()
            if cfg.GRAD_CLIP_NORM > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP_NORM)
            optimizer.step()

            epoch_loss += loss.item()
            processed_steps += 1
            if step % cfg.LOG_INTERVAL == 0 and processed_steps > 0:
                pbar.set_postfix({"loss": epoch_loss / processed_steps})
        scheduler.step()

        avg_epoch_loss = epoch_loss / max(1, processed_steps)
        print(f"[Epoch {epoch}] avg contrastive loss = {avg_epoch_loss:.4f}")
        if skipped_steps > 0:
            print(f"[Epoch {epoch}] skipped {skipped_steps} batches due to non-finite embeddings.")

        # 每若干 epoch 保存一下 encoder 权重（用于 Stage3）
        if epoch % 10 == 0 or epoch == cfg.N_EPOCHS:
            ckpt = {
                "encoder_state_dict": model.encoder.state_dict(),
                "proj_state_dict": model.proj.state_dict(),
                "cfg": cfg.__dict__,
                "epoch": epoch,
                "loss": avg_epoch_loss,
            }
            torch.save(ckpt, cfg.CKPT_PATH)
            print(f"Saved checkpoint to {cfg.CKPT_PATH}")

    print("Contrastive pretraining done.")


if __name__ == "__main__":
    train_contrastive(cfg)
