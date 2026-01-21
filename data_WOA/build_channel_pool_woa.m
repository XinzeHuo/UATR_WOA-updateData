function ChannelPool = build_channel_pool_woa(arr_dir, max_paths, normalize_amplitude)
% build_channel_pool_woa
%   从 ASCII .arr 文件构建通道池 (ChannelPool)
%   要求存在 read_arrivals_asc.m
%
% 输入:
%   arr_dir            : 存放 .arr 的文件夹
%   max_paths          : 每个通道最多保留的路径数 (按功率排序)
%   normalize_amplitude: 是否对每个通道幅度归一化到 max|A|=1
%
% 输出:
%   ChannelPool(k) 结构体数组，字段：
%       .Amp        : P x 1  复振幅
%       .tau        : P x 1  时延 (秒)
%       .tau_rel    : P x 1  相对时延（以最强路径对齐为 0）
%       .power_db   : 总功率 (dB)
%       .Npaths     : 有效路径数
%       .meta       : 元信息 (range/src_z/rcv_z/env_name/ssp_type/H/bottom_type)

    if nargin < 2 || isempty(max_paths)
        max_paths = 32;
    end
    if nargin < 3 || isempty(normalize_amplitude)
        normalize_amplitude = true;
    end

    arr_files = dir(fullfile(arr_dir, '*.arr'));
    if isempty(arr_files)
        error('No .arr files found in directory: %s', arr_dir);
    end

    ChannelPool = struct('Amp', {}, 'tau', {}, 'tau_rel', {}, ...
                         'power_db', {}, 'Npaths', {}, 'meta', {});

    idx = 1;
    NarrMax = 1000;  % 允许的最大到达数（可按需调整）

    for k = 1:numel(arr_files)
        arr_name = arr_files(k).name;
        arr_path = fullfile(arr_dir, arr_name);
        fprintf('[%d/%d] Reading arrivals: %s\n', k, numel(arr_files), arr_name);

        % ---------- 读 ASCII .arr ----------
        try
            [amp1, delay, SrcAngle, RcvrAngle, NumTopBnc, NumBotBnc, narrmat, Pos] = ...
                read_arrivals_asc(arr_path, NarrMax); %#ok<ASGLU>
        catch ME
            warning('Failed to read %s : %s', arr_path, ME.message);
            continue;
        end

        % amp1, delay 通常是 [Nrec x NarrMax] 或 [Nch x NarrMax]
        % narrmat(i) 表示第 i 个接收“通道”真正的到达数（后面是 0）
        [Nch, NarrMax_eff] = size(amp1); %#ok<NASGU>

        for ich = 1:Nch
            % 每个 ich 看成一个“通道”
            if numel(narrmat) >= ich
                nEff = narrmat(ich);
            else
                % 如果 narrmat 形状不明，保守地全部用上再去掉全 0
                nEff = size(amp1, 2);
            end
            if nEff <= 0
                continue;
            end

            Amp_all = squeeze(amp1(ich, 1:nEff)).';
            tau_all = squeeze(delay(ich, 1:nEff)).';

            % 去掉全 0 的路径
            valid = abs(Amp_all) > 0;
            Amp_all = Amp_all(valid);
            tau_all = tau_all(valid);

            if isempty(Amp_all)
                continue;
            end

            % ---------- 按功率排序并截断 ----------
            [~, order] = sort(abs(Amp_all).^2, 'descend');
            keepN = min(max_paths, numel(order));
            keep  = order(1:keepN);

            Amp = Amp_all(keep);
            tau = tau_all(keep);

            % ---------- 以最强路径对齐相对时延 ----------
            [~, imax_rel] = max(abs(Amp));
            tau_ref = tau(imax_rel);
            tau_rel = tau - tau_ref;

            % ---------- 可选归一化 ----------
            if normalize_amplitude && ~isempty(Amp)
                Amp = Amp ./ max(abs(Amp));
            end

            % ---------- 计算总功率 ----------
            total_power = sum(abs(Amp).^2);
            power_db    = 10 * log10(max(total_power, eps));

            % ---------- 元信息 (尽量填，缺省为 NaN) ----------
            meta = struct();
            meta.arr_file   = arr_name;
            meta.range_m    = NaN;
            meta.src_z_m    = NaN;
            meta.rcv_z_m    = NaN;

            % 如果 Pos 里有可用字段就尝试取一下，但不强依赖
            if exist('Pos','var') && isstruct(Pos)
                % Compatible with nested structure returned by read_arrivals_asc: Pos.r.r / Pos.r.z / Pos.s.z
                if isnan(meta.range_m)
                    v = try_get_numeric(Pos, {'r','r'}, 1);
                    if ~isnan(v), meta.range_m = v; end
                end
                if isnan(meta.range_m)
                    v = try_get_numeric(Pos, {'r'}, 1);
                    if ~isnan(v), meta.range_m = v; end
                end
                if isnan(meta.range_m)
                    v = try_get_numeric(Pos, {'rr'}, 1);
                    if ~isnan(v), meta.range_m = v; end
                end

                if isnan(meta.src_z_m)
                    v = try_get_numeric(Pos, {'s','z'}, 1);
                    if ~isnan(v), meta.src_z_m = v; end
                end
                if isnan(meta.src_z_m)
                    v = try_get_numeric(Pos, {'s'}, 1);
                    if ~isnan(v), meta.src_z_m = v; end
                end
                if isnan(meta.src_z_m)
                    v = try_get_numeric(Pos, {'sd'}, 1);
                    if ~isnan(v), meta.src_z_m = v; end
                end
                if isnan(meta.src_z_m)
                    v = try_get_numeric(Pos, {'zs'}, 1);
                    if ~isnan(v), meta.src_z_m = v; end
                end

                if isnan(meta.rcv_z_m)
                    v = try_get_numeric(Pos, {'r','z'}, ich);
                    if ~isnan(v), meta.rcv_z_m = v; end
                end
                if isnan(meta.rcv_z_m)
                    v = try_get_numeric(Pos, {'rd'}, ich);
                    if ~isnan(v), meta.rcv_z_m = v; end
                end
                if isnan(meta.rcv_z_m)
                    v = try_get_numeric(Pos, {'zr'}, ich);
                    if ~isnan(v), meta.rcv_z_m = v; end
                end
            end

            % 解析文件名中的 H / 底质 / ssp_type
            [meta.env_name, meta.ssp_type, meta.H, meta.bottom_type] = parse_env_meta(arr_name);

            if isnan(meta.rcv_z_m)
                meta.rcv_z_m = 10;
            end

            % ---------- 写入 ChannelPool ----------
            ChannelPool(idx).Amp      = Amp(:);
            ChannelPool(idx).tau      = tau(:);
            ChannelPool(idx).tau_rel  = tau_rel(:);
            ChannelPool(idx).power_db = power_db;
            ChannelPool(idx).Npaths   = numel(Amp);
            ChannelPool(idx).meta     = meta;

            idx = idx + 1;
        end
    end

    fprintf('WOA Channel pool built: total %d channel entries.\n', numel(ChannelPool));
end

% 简单获取嵌套字段的数值（如果存在且为 numeric），否则返回 NaN
function val = try_get_numeric(s, path_cells, idx)
    if nargin < 3 || isempty(idx)
        idx = 1;
    end
    val = NaN;
    cur = s;
    for k = 1:numel(path_cells)
        key = path_cells{k};
        if ~isstruct(cur) || ~isfield(cur, key)
            return;
        end
        cur = cur.(key);
    end
    if isnumeric(cur) && ~isempty(cur) && numel(cur) >= idx
        val = cur(idx);
    end
end

% -------------------------------------------------------------------------
function [env_name, ssp_type, H, bottom_type] = parse_env_meta(filename)
    [~, name, ~] = fileparts(filename);
    env_name    = name;
    ssp_type    = '';
    H           = NaN;
    bottom_type = '';

    parts = split(name, {'_','-'});
    parts = parts(:);

    ssp_candidates = {'munk','summer','isothermal','winter','deep', ...
                      'summer_shallow','winter_shallow','deep_channel'};
    for i = 1:numel(parts)
        p = lower(parts{i});
        for j = 1:numel(ssp_candidates)
            if contains(p, ssp_candidates{j})
                ssp_type = ssp_candidates{j};
                break;
            end
        end
        if ~isempty(ssp_type), break; end
    end

    for i = 1:numel(parts)
        p = parts{i};
        if startsWith(lower(p), 'h')
            numstr = regexp(p, '\d+','match','once');
            if ~isempty(numstr)
                H = str2double(numstr);
                break;
            end
        end
    end

    for i = 1:numel(parts)
        p = lower(parts{i});
        if contains(p,'sand')
            bottom_type = 'sand'; break;
        elseif contains(p,'mud')
            bottom_type = 'mud'; break;
        end
    end

    if isnan(H)
        num_tokens = regexp(name, '[0-9]+','match');
        if ~isempty(num_tokens)
            for nt = 1:numel(num_tokens)
                val = str2double(num_tokens{nt});
                if val > 10 && val < 10000
                    H = val;
                    break;
                end
            end
        end
    end
end
