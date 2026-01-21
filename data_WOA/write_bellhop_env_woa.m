function envFile = write_bellhop_env_woa(envPath, ssp, H, bottom_type, src_z, rcv_z, r_vec)
% 写出符合 Bellhop 格式的 .env 文件（更健壮的输入预处理和检查）
% envPath: full path without extension
% ssp: struct with fields .z and .c (深度、声速)
% H: water depth (m)
% bottom_type: optional (不写入 .env)
% src_z: scalar 或向量 (m)
% rcv_z: 接收深度向量 (m)
% r_vec: 接收距离向量 (m)
%
% 返回 envFile（含扩展名）或在出错时抛错

envFile = [envPath '.env'];
fid = fopen(envFile, 'w');
if fid == -1
    error('Cannot open file for writing: %s', envFile);
end

% ------------------- 预处理 SSP --------------------
z = double(ssp.z(:));
c = double(ssp.c(:));

% 去掉 NaN/Inf
valid = isfinite(z) & isfinite(c);
z = z(valid);
c = c(valid);

if isempty(z)
    fclose(fid);
    error('Empty/invalid SSP provided for env: %s', envFile);
end

% 裁剪到 [0,H]
z = max(z, 0);
z = min(z, H);

% 合并、排序并去重（保留第一次出现的）
[zu, ia] = unique(z, 'stable');
cu = c(ia);
[zu, idxs] = sort(zu);
cu = cu(idxs);

z = zu;
c = cu;

% 确保包含 0 与 H（插值/外推）
if z(1) > 0
    c0 = interp1(z, c, 0, 'linear', 'extrap');
    z = [0; z];
    c = [c0; c];
end
if z(end) < H
    cH = interp1(z, c, H, 'linear', 'extrap');
    z = [z; H];
    c = [c; cH];
end

% 防止非常小的重复（再去重）
[~, ia] = unique(z, 'stable');
z = z(ia);
c = c(ia);

% ------------------- 写文件 --------------------------
try
    % 标题 / 频率 / NMedia
    fprintf(fid, '''WOA-based SSP: %s''\n', envPath);
    fprintf(fid, '1000.00\n'); % frequency
    fprintf(fid, '1\n');       % NMedia

    % SSP section header （C-linear, Vacuum, dB/wavelength）
    fprintf(fid, '''CVW''   ! C-linear, Vacuum, dB/wavelength\n');
    fprintf(fid, '%d  %.1f  %.1f\n', numel(z), 0.0, H);

    % 为了兼容性，采用每行写一个样点（并在行末标记 / ）
    for i = 1:numel(z)
        fprintf(fid, '%8.2f  %8.2f  /\n', z(i), c(i));
    end

    % SOURCES
    fprintf(fid, '''R''  0.0\n');  % no offset
    if isempty(src_z)
        src_z = 0.0;
    end
    % 支持向量或标量 src_z
    if numel(src_z) > 1
        fprintf(fid, '%d\n', numel(src_z));       % NSD
        fprintf(fid, ' ');
        fprintf(fid, sprintf('%8.2f ', double(src_z(:)')));
        fprintf(fid, '/\n');
    else
        fprintf(fid, '%d\n', 1);       % NSD
        fprintf(fid, '%8.2f /\n', double(src_z));
    end

    % RECEIVERS (depths)
    fprintf(fid, '%d\n', numel(rcv_z));  % NRD
    fprintf(fid, ' ');
    fprintf(fid, sprintf('%8.2f ', double(rcv_z(:)')));
    fprintf(fid, '/\n');

    % RANGES (转换为 km)
    fprintf(fid, '%d\n', numel(r_vec)); % NR
    fprintf(fid, ' ');
    fprintf(fid, sprintf('%8.2f ', double(r_vec(:)') / 1000)); % km
    fprintf(fid, '/\n');

    % BOTTOM / Ray trace params (简单默认值，与原实现保持一致)
    fprintf(fid, '''A''\n');        % Ray trace
    fprintf(fid, '201\n');
    fprintf(fid, '-30.0  30.0  /\n');
    zbox     = max(H, max(rcv_z(:)) * 1.1);        % allow beams to reach deepest receiver
    rbox_km  = max(r_vec(:)) / 1000 * 1.2;         % ensure propagation covers max range
    fprintf(fid, '0.0  %.1f  %.3f\n', zbox, rbox_km);

    fclose(fid);
catch ME
    if fid ~= -1, fclose(fid); end
    rethrow(ME);
end

% 简单校验：文件大小 > 0
finfo = dir(envFile);
if isempty(finfo) || finfo.bytes < 10
    error('Env file seems too small: %s (size=%d)', envFile, finfo.bytes);
end

end
