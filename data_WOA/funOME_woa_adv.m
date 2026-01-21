function [y, Hk, cfg_used] = funOME_woa_adv(x, fs, ch, cfg)
% 高级 WOA 多径叠加：shadowing + delay jitter + Thorp 吸收
% 添加峰值保护，避免写入WAV文件时被裁剪

if nargin < 4, cfg = struct(); end
cfg_defaults = struct( ...
    'use_shadowing', true, ...
    'shadow_sigma_db', 3.0, ...
    'use_delay_jitter', true, ...
    'delay_jitter_max', 0.5e-3, ...
    'use_absorption', true, ...
    'normalize_output', true, ...
    'output_mode', 'same', ... % 'same' or 'full'
    'peak_limit_mode', 'clip', ... % 新增：峰值处理模式 'clip', 'scale', 'none'
    'seed', [] ...
    );
cfg_used = cfg_defaults;
if ~isempty(cfg)
    fn = fieldnames(cfg);
    for i=1:numel(fn)
        cfg_used.(fn{i}) = cfg.(fn{i});
    end
end

if ~isempty(cfg_used.seed)
    rng(cfg_used.seed);
end

x = x(:);
N = length(x);
if ~isfield(ch,'Amp') || ~isfield(ch,'tau')
    error('ch must contain fields Amp and tau.');
end
% Robustly extract numeric values from potentially struct-wrapped fields
Amp0 = local_extract_numeric(ch.Amp, 'Amp');
Amp0 = double(Amp0(:).');
tau0 = local_extract_numeric(ch.tau, 'tau');
tau0 = double(tau0(:).');

P = numel(Amp0);
if P == 0
    y  = zeros(N,1);
    Hk = [];
    return;
end

% 1) shadowing
if cfg_used.use_shadowing
    sigma_db = cfg_used.shadow_sigma_db;
    dB_perturb = sigma_db * randn(size(Amp0));
    Amp = Amp0 .* 10.^(dB_perturb/20);
else
    Amp = Amp0;
end

% 2) delay jitter
if cfg_used.use_delay_jitter
    dt_max = cfg_used.delay_jitter_max;
    dt  = (2*rand(size(tau0)) - 1) * dt_max;
    tau = tau0 + dt;
else
    tau = tau0;
end

[~, imax] = max(abs(Amp));
tau = tau - tau(imax);

max_delay_samp = ceil(max(abs(tau)) * fs);
Nfft = 2^nextpow2(N + max_delay_samp);

Fx = fft(x, Nfft).';
fk = (0:(Nfft-1)) * (fs / Nfft);

phase = exp(-1j * 2 * pi * (tau(:) * fk));

% 5) Thorp 吸收
if cfg_used.use_absorption
    alpha_db_per_km = thorp_alpha(fk);
    if isfield(ch, 'meta') && isfield(ch.meta, 'range_m')
        % Robustly extract numeric value from potentially struct-wrapped range_m
        d_all = local_extract_numeric(ch.meta.range_m, 'range_m');
        if numel(d_all) == 1
            d_all = repmat(d_all, size(Amp));
        end
        d_all = double(d_all(:).');
    else
        warning('ch.meta.range_m not present; using nominal 1 km for absorption scaling.');
        d_all = ones(1,P) * 1000;
    end
    dist_km = d_all / 1e3;
    G_mat = 10 .^ ( - (dist_km(:) * alpha_db_per_km) / 20 );
else
    G_mat = ones(P, Nfft);
end

Amp_mat = Amp(:) * ones(1, Nfft);
H_mat   = Amp_mat .* phase .* G_mat;
Hk      = sum(H_mat, 1);

Yk    = Fx .* Hk;
y_full = ifft(Yk, Nfft).';

if strcmpi(cfg_used.output_mode, 'full')
    y = real(y_full(1:Nfft));
else
    y = real(y_full(1:N));
end

% 归一化处理
if cfg_used.normalize_output
    rms_x = rms(x);
    rms_y = rms(y);
    if rms_y > eps
        y = y * (rms_x / rms_y);
    end
end

% 峰值保护：避免写入WAV文件时被裁剪
max_abs_val = max(abs(y));
if max_abs_val > 1
    switch lower(cfg_used.peak_limit_mode)
        case 'clip'
            % 硬限幅：直接裁剪超出范围的部分
            y = max(min(y, 1), -1);
            % 可以输出调试信息
            % fprintf('信号峰值 %.3f > 1，已应用硬限幅\n', max_abs_val);
        case 'scale'
            % 软归一化：按比例缩放整个信号
            y = y / max_abs_val;
            % 可以输出调试信息
            % fprintf('信号峰值 %.3f > 1，已应用缩放 (%.3f倍)\n', max_abs_val, 1/max_abs_val);
        case 'none'
            % 不处理，让audiowrite裁剪（会显示警告）
            % 保持原样
        otherwise
            % 默认硬限幅
            y = max(min(y, 1), -1);
    end
end

end

function alpha_db = thorp_alpha(f_hz)
f_khz = f_hz / 1e3;
f2    = f_khz.^2;
alpha_db = 0.11 * f2 ./ (1 + f2) + 44 * f2 ./ (4100 + f2) + 2.75e-4 * f2 + 0.003;
alpha_db(~isfinite(alpha_db)) = max(alpha_db(isfinite(alpha_db)));
alpha_db(alpha_db < 0) = 0;
end

function val = local_extract_numeric(field, field_name)
% Extract numeric value from a field that may be a struct or numeric array
% If field is a struct, tries to extract numeric data from common field names
    if isnumeric(field)
        val = field;
    elseif isstruct(field)
        % Try common field names that might contain the actual numeric value
        possible_fields = {'data', 'value', 'val', field_name};
        val = [];
        for i = 1:numel(possible_fields)
            if isfield(field, possible_fields{i})
                candidate = field.(possible_fields{i});
                if isnumeric(candidate)
                    val = candidate;
                    break;
                end
            end
        end
        if isempty(val)
            % If no numeric field found, try to get first numeric field
            fnames = fieldnames(field);
            for i = 1:numel(fnames)
                candidate = field.(fnames{i});
                if isnumeric(candidate)
                    val = candidate;
                    break;
                end
            end
        end
        if isempty(val)
            error('Cannot extract numeric value from struct field: %s', field_name);
        end
    else
        error('Field %s is neither numeric nor struct (type: %s)', field_name, class(field));
    end
end