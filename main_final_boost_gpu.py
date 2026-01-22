import os
import glob
import random
import time
import math
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler  # [新增] 混合精度训练
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE


def prepare_woa_waveform(y, sr, target_sr, target_len, split):
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        if split == 'train':
            start = random.randint(0, len(y) - target_len)
            y = y[start:start + target_len]
        else:
            y = y[:target_len]
    wav = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
    max_abs = wav.abs().max()
    if max_abs > 1.0:
        wav = wav / (max_abs + 1e-6)
    return wav


# ================= 0. 日志与工具 =================
class Logger(object):
    def __init__(self, filename='result_final_boost_gpu_oridata.txt'):
        self.terminal = print
        self.log = open(filename, 'a', encoding='utf-8')

    def print(self, message):
        ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        msg = f"[{ts}] {message}"
        self.terminal(msg)
        self.log.write(msg + '\n')
        self.log.flush()

    def close(self): self.log.close()


logger = Logger()


def log(msg): logger.print(msg)


CONFIG = {
    "data_root": "./ShipsEar_16k_30s_hop15",
    "sample_rate": 16000,
    "duration": 30.0,
    "target_len": 938,  # Align to ~30s
    "woa_encoder_ckpt": "./encoder_contrastive_woa.pt",
    "woa_freeze": True,
    "woa_duration_default": 10.0,
    "woa_test_split": "test",

    # 双分辨率配置
    "fft_high": 4096, "hop_high": 512, "mels_high": 128,  # 流1: 高频分辨率
    "fft_low": 512, "hop_low": 512, "mels_low": 64,  # 流2: 高时分辨率

    # 训练参数
    "batch_size": 32,  # [优化] GPU上可以适当增大 batch_size
    "epochs": 100,
    "lr": 0.001,
    "weight_decay": 1e-3,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    "num_workers": 8  # [优化] 增加数据加载线程
}

CLASSES = ["ClassA", "ClassB", "ClassC", "ClassD", "ClassE"]
CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASSES)}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # [优化] 开启 benchmark 加速
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    log(f"Random seed set to {seed}, Device: {CONFIG['device']}")


set_seed(CONFIG['seed'])


# ================= 1. 数据增强与数据集 =================

class SpecAugment:
    def __init__(self, freq_mask=20, time_mask=40, num_masks=2):
        self.freq_mask = freq_mask
        self.time_mask = time_mask
        self.num_masks = num_masks

    def __call__(self, spec):
        cloned = spec.clone()
        F, T = cloned.shape
        for _ in range(self.num_masks):
            f = random.randint(0, self.freq_mask)
            f0 = random.randint(0, F - f)
            cloned[f0:f0 + f, :] = 0.0
            t = random.randint(0, self.time_mask)
            t0 = random.randint(0, T - t)
            cloned[:, t0:t0 + t] = 0.0
        return cloned


def get_spectrogram(y, sr, n_fft, hop_length, n_mels, target_len):
    # Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    log_mel = librosa.power_to_db(mel, ref=np.max)

    # Padding/Cropping
    if log_mel.shape[1] < target_len:
        log_mel = np.pad(log_mel, ((0, 0), (0, target_len - log_mel.shape[1])))
    else:
        log_mel = log_mel[:, :target_len]

    # Standardize
    mean = log_mel.mean()
    std = log_mel.std() + 1e-6
    norm_mel = (log_mel - mean) / std
    return torch.tensor(norm_mel, dtype=torch.float32).unsqueeze(0)


class ShipsEarDualDataset(Dataset):
    def __init__(self, root_dir, split="train", config=CONFIG, woa_cfg=None):
        self.config = config
        self.split = split
        self.files = []
        self.labels = []
        self.spec_aug = SpecAugment() if split == 'train' else None
        woa_cfg = woa_cfg or {}
        self.woa_sample_rate = int(woa_cfg.get("SAMPLE_RATE", config["sample_rate"]))
        self.woa_duration = float(woa_cfg.get("MAX_DURATION_SEC", config["woa_duration_default"]))
        self.woa_target_len = int(self.woa_sample_rate * self.woa_duration)

        for fpath in glob.glob(os.path.join(root_dir, split, "**", "*.wav"), recursive=True):
            for cls in CLASSES:
                if cls in fpath.split(os.sep):
                    self.files.append(fpath)
                    self.labels.append(CLASS_TO_IDX[cls])
                    break
        log(f"[{split}] Loaded {len(self.files)} files.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        wav_path = self.files[idx]
        label = self.labels[idx]

        try:
            y, sr = librosa.load(wav_path, sr=self.config['sample_rate'])
        except Exception as e:
            print(f"Error loading {wav_path}: {e}")
            y = np.zeros(int(self.config['sample_rate'] * self.config['duration']))
            sr = self.config['sample_rate']

        # Audio Alignment
        tgt_samps = int(self.config['sample_rate'] * self.config['duration'])
        if len(y) < tgt_samps:
            y = np.pad(y, (0, tgt_samps - len(y)))
        else:
            # Training: Random Crop; Test: Center Crop
            if self.split == 'train':
                start = random.randint(0, len(y) - tgt_samps)
                y = y[start:start + tgt_samps]
            else:
                y = y[:tgt_samps]

        # Stream 1: High Freq Resolution
        spec1 = get_spectrogram(y, sr, CONFIG['fft_high'], CONFIG['hop_high'], CONFIG['mels_high'],
                                CONFIG['target_len'])
        # Stream 2: High Time Resolution
        spec2 = get_spectrogram(y, sr, CONFIG['fft_low'], CONFIG['hop_low'], CONFIG['mels_low'], CONFIG['target_len'])
        woa_wave = prepare_woa_waveform(y, sr, self.woa_sample_rate, self.woa_target_len, self.split)

        if self.spec_aug:
            spec1 = self.spec_aug(spec1.squeeze(0)).unsqueeze(0)
            spec2 = self.spec_aug(spec2.squeeze(0)).unsqueeze(0)

        return spec1, spec2, woa_wave, label


# ================= 2. 模型：Dual-Resolution CNN + ArcFace =================

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.se = SEBlock(out_ch)
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.se(x)
        return self.pool(x)


class DualResNet(nn.Module):
    def __init__(self, feat_dim=256):
        super().__init__()
        # Branch 1 (High Freq)
        self.b1 = nn.Sequential(
            ConvBlock(1, 16), ConvBlock(16, 32), ConvBlock(32, 64), ConvBlock(64, 128),
            nn.AdaptiveAvgPool2d(1)
        )
        # Branch 2 (High Time)
        self.b2 = nn.Sequential(
            ConvBlock(1, 16), ConvBlock(16, 32), ConvBlock(32, 64), ConvBlock(64, 128),
            nn.AdaptiveAvgPool2d(1)
        )
        # Fusion
        self.fc = nn.Sequential(
            nn.Linear(128 + 128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, feat_dim),
            nn.BatchNorm1d(feat_dim)  # BN is important for ArcFace
        )

    def forward(self, x1, x2):
        f1 = self.b1(x1).flatten(1)
        f2 = self.b2(x2).flatten(1)
        cat = torch.cat([f1, f2], dim=1)
        feat = self.fc(cat)
        return feat


class WOAEncoder1D(nn.Module):
    def __init__(self, embed_dim=128, dyn_hidden=128, dropout=0.1):
        super().__init__()
        base_channels = max(16, dyn_hidden // 4)
        mid_channels = max(32, dyn_hidden // 2)
        self.features = nn.Sequential(
            nn.Conv1d(1, base_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
            nn.Conv1d(base_channels, mid_channels, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(),
            nn.Conv1d(mid_channels, dyn_hidden, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(dyn_hidden),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(dyn_hidden, embed_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)


class DualResNetWithWOA(nn.Module):
    def __init__(self, feat_dim=128, woa_embed_dim=128):
        super().__init__()
        self.backbone = DualResNet(feat_dim=feat_dim)
        self.woa_proj = nn.Sequential(
            nn.Linear(woa_embed_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU()
        )
        self.fuse = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU()
        )

    def forward(self, x1, x2, woa_feat):
        base_feat = self.backbone(x1, x2)
        woa_feat = self.woa_proj(woa_feat)
        fused = torch.cat([base_feat, woa_feat], dim=1)
        return self.fuse(fused)


class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label=None):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        if label is None:
            return cosine * self.s

        phi = cosine - self.m
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


# ================= 3. 训练与测试 (含 TTA) =================

def main():
    start_time = time.time()
    log("=== Dual-Resolution + ArcFace Training Start (GPU Optimized) ===")

    # 1. Dataset
    if not os.path.exists(CONFIG['woa_encoder_ckpt']):
        log(
            "Error: WOA encoder checkpoint not found at "
            f"{CONFIG['woa_encoder_ckpt']}. Please ensure the contrastive training "
            "has been completed and the checkpoint exists."
        )
        return
    woa_ckpt = torch.load(CONFIG['woa_encoder_ckpt'], map_location='cpu')
    woa_cfg = woa_ckpt.get("cfg", {})
    train_ds = ShipsEarDualDataset(CONFIG['data_root'], split='train', woa_cfg=woa_cfg)
    test_ds = ShipsEarDualDataset(CONFIG['data_root'], split='test', woa_cfg=woa_cfg)

    # [优化] 使用 pin_memory 和 persistent_workers 加速数据流
    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=True,
        persistent_workers=True
    )
    # 验证集 Batch size 设为 1，方便做 TTA
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=True,
        persistent_workers=True
    )

    # Class Weights for CE part
    counts = [0] * 5
    for l in train_ds.labels: counts[l] += 1
    w = [sum(counts) / (5 * c) if c > 0 else 0 for c in counts]
    weights_t = torch.FloatTensor(w).to(CONFIG['device'])
    log(f"Class Weights: {w}")

    # 2. Model & Loss
    model = DualResNetWithWOA(feat_dim=128, woa_embed_dim=woa_cfg.get("EMBED_DIM", 128)).to(CONFIG['device'])
    woa_encoder = WOAEncoder1D(
        embed_dim=woa_cfg.get("EMBED_DIM", 128),
        dyn_hidden=woa_cfg.get("DYN_HIDDEN_CHANNELS", 128),
        dropout=woa_cfg.get("DROPOUT", 0.1)
    ).to(CONFIG['device'])
    woa_encoder.load_state_dict(woa_ckpt["encoder_state_dict"])
    if CONFIG["woa_freeze"]:
        woa_encoder.eval()
        for param in woa_encoder.parameters():
            param.requires_grad = False
    arcface = ArcFace(in_features=128, out_features=5, s=30.0, m=0.3).to(CONFIG['device'])

    log(f"Model Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    criterion = nn.CrossEntropyLoss()  # ArcFace output is already scaled logits
    scaler = GradScaler()  # [新增] 混合精度训练

    # Optimizing both Backbone and ArcFace Head
    opt_params = [{'params': model.parameters()}, {'params': arcface.parameters()}]
    if not CONFIG["woa_freeze"]:
        opt_params.append({'params': woa_encoder.parameters()})
    optimizer = optim.AdamW(opt_params, lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])

    best_acc = 0.0
    history = {'t_loss': [], 't_acc': [], 'v_acc': []}

    for epoch in range(CONFIG['epochs']):
        # --- Train ---
        model.train()
        t_loss, corr, tot = 0, 0, 0

        # [修复] 移除原来的 Mixup 空循环，直接进行标准训练
        # ArcFace 对 Margin 比较敏感，简单起见在 GPU 优化版中暂不使用 Mixup

        for x1, x2, woa_wave, y in train_loader:
            # [优化] non_blocking=True 加速数据传输
            x1 = x1.to(CONFIG['device'], non_blocking=True)
            x2 = x2.to(CONFIG['device'], non_blocking=True)
            woa_wave = woa_wave.to(CONFIG['device'], non_blocking=True)
            y = y.to(CONFIG['device'], non_blocking=True)

            optimizer.zero_grad()

            # [新增] 混合精度上下文
            with autocast():
                if CONFIG["woa_freeze"]:
                    with torch.no_grad():
                        woa_feat = woa_encoder(woa_wave)
                else:
                    woa_feat = woa_encoder(woa_wave)
                feat = model(x1, x2, woa_feat)
                logits = arcface(feat, y)
                loss = criterion(logits, y)

            # [新增] 梯度缩放
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            t_loss += loss.item()
            _, pred = logits.max(1)
            tot += y.size(0)
            corr += pred.eq(y).sum().item()

        scheduler.step()
        train_acc = 100. * corr / tot
        train_loss = t_loss / len(train_loader)

        # --- Validation (Test-Time Augmentation) ---
        model.eval()
        v_corr, v_tot = 0, 0

        # TTA: For each test sample, we just predict once here to save time during epoch.
        # Final evaluation will use TTA.
        with torch.no_grad():
            for x1, x2, woa_wave, y in test_loader:
                x1 = x1.to(CONFIG['device'], non_blocking=True)
                x2 = x2.to(CONFIG['device'], non_blocking=True)
                woa_wave = woa_wave.to(CONFIG['device'], non_blocking=True)
                y = y.to(CONFIG['device'], non_blocking=True)

                with autocast():
                    woa_feat = woa_encoder(woa_wave)
                    feat = model(x1, x2, woa_feat)
                    logits = arcface(feat)  # No label needed for inference

                _, pred = logits.max(1)
                v_tot += y.size(0)
                v_corr += pred.eq(y).sum().item()

        val_acc = 100. * v_corr / v_tot

        log(f"Ep {epoch + 1:03d} | Loss: {train_loss:.4f} T_Acc: {train_acc:.2f}% | V_Acc: {val_acc:.2f}% | Time: {time.time() - start_time:.1f}s")

        history['t_loss'].append(train_loss)
        history['t_acc'].append(train_acc)
        history['v_acc'].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({'model': model.state_dict(), 'head': arcface.state_dict()}, "best_final_model.pth")
            log(f">>> Best Saved! {best_acc:.2f}%")

    # ================= 4. Final Evaluation with TTA =================
    log("=== Final Evaluation with TTA (5 crops) ===")
    checkpoint = torch.load("best_final_model.pth")
    model.load_state_dict(checkpoint['model'])
    arcface.load_state_dict(checkpoint['head'])
    model.eval()

    # Custom TTA Loop
    # We need to access the dataset directly to generate multiple crops
    y_true, y_pred, y_feats = [], [], []

    # Iterate over test dataset indices
    for i in range(len(test_ds)):
        wav_path = test_ds.files[i]
        label = test_ds.labels[i]

        # Load audio
        try:
            raw_y, sr = librosa.load(wav_path, sr=CONFIG['sample_rate'])
        except:
            continue

        # TTA: Generate 5 random crops
        logits_sum = torch.zeros(1, 5).to(CONFIG['device'])

        crops = 5
        for _ in range(crops):
            # Random Crop
            tgt = int(CONFIG['sample_rate'] * CONFIG['duration'])
            if len(raw_y) < tgt:
                pad_y = np.pad(raw_y, (0, tgt - len(raw_y)))
            else:
                start = random.randint(0, len(raw_y) - tgt)
                pad_y = raw_y[start:start + tgt]

            # Get dual specs
            s1 = get_spectrogram(pad_y, sr, CONFIG['fft_high'], CONFIG['hop_high'], CONFIG['mels_high'],
                                 CONFIG['target_len'])
            s2 = get_spectrogram(pad_y, sr, CONFIG['fft_low'], CONFIG['hop_low'], CONFIG['mels_low'],
                                 CONFIG['target_len'])

            with torch.no_grad():
                with autocast():
                    woa_wave = prepare_woa_waveform(
                        pad_y,
                        sr,
                        train_ds.woa_sample_rate,
                        train_ds.woa_target_len,
                        CONFIG["woa_test_split"]
                    )
                    woa_feat = woa_encoder(woa_wave.unsqueeze(0).to(CONFIG['device']))
                    feat = model(
                        s1.unsqueeze(0).to(CONFIG['device']),
                        s2.unsqueeze(0).to(CONFIG['device']),
                        woa_feat
                    )
                    logits = arcface(feat)  # returns scaled cosine
                logits_sum += logits

        # Average Logits
        avg_logits = logits_sum / crops
        _, pred = avg_logits.max(1)

        y_true.append(label)
        y_pred.append(pred.item())
        y_feats.append(feat.cpu().float().numpy().squeeze())  # Keep last feat for t-SNE (convert to float32)

    # Metrics
    log("\n" + classification_report(y_true, y_pred, target_names=CLASSES, digits=4))

    # Plots
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f'Final Boost Matrix (TTA)')
    plt.savefig('cm_final.png')

    tsne = TSNE(2, random_state=42)
    X_emb = tsne.fit_transform(np.array(y_feats))
    plt.figure(figsize=(10, 8))
    for i, c in enumerate(CLASSES):
        idx = np.array(y_true) == i
        plt.scatter(X_emb[idx, 0], X_emb[idx, 1], label=c, alpha=0.7)
    plt.legend()
    plt.savefig('tsne_final.png')

    log(f"Total Time: {(time.time() - start_time) / 60:.2f} min")
    logger.close()


if __name__ == "__main__":
    if not os.path.exists(CONFIG['data_root']):
        log(f"Error: {CONFIG['data_root']} not found.")
    else:
        main()
