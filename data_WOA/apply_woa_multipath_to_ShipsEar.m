% apply_woa_multipath_to_ShipsEar.m
clearvars;
close all;
clc;

%% 1. 配置
METADATA_PATH    = 'E:\rcq\pythonProject\Data\ShipsEar_16k_30s_hop15\metadata.csv';
CLEAN_DATA_ROOT  = 'E:\rcq\pythonProject\Data\ShipsEar_16k_30s_hop15';
OUTPUT_ROOT      = 'E:\rcq\pythonProject\Data\ShipsEar_16k_30s_hop15_WOA';
CHANNEL_POOL_MAT = 'E:\rcq\pythonProject\UWTRL-MEG-main\data_gen\ChannelPool_WOA.mat';

TARGET_FS            = 16000;
RESAMPLE_IF_MISMATCH = true;
RNG_SEED             = 20260114;

cfg.use_shadowing    = true;
cfg.shadow_sigma_db  = 3.0;
cfg.use_delay_jitter = true;
cfg.delay_jitter_max = 0.5e-3;
cfg.use_absorption   = true;
cfg.normalize_output = true;
cfg.output_mode      = 'same';
cfg.seed             = [];

CHANNEL_SELECTION_MODE = 'random';
SEEN_CHANNEL_IDX   = [];
UNSEEN_CHANNEL_IDX = [];

WRITE_WAV = true;
OVERWRITE = true;
LOG_FILE  = fullfile(OUTPUT_ROOT, 'woa_augmentation_log.csv');

USE_PARPOOL = false;

%% 2. 初始化
if ~isempty(RNG_SEED)
    rng(RNG_SEED);
end

if ~exist(CHANNEL_POOL_MAT, 'file')
    error('Channel pool file not found: %s', CHANNEL_POOL_MAT);
end
S = load(CHANNEL_POOL_MAT, 'ChannelPool');
if ~isfield(S, 'ChannelPool')
    error('Loaded MAT does not contain ChannelPool variable.');
end
ChannelPool  = S.ChannelPool;
num_channels = numel(ChannelPool);
fprintf('Loaded ChannelPool with %d entries.\n', num_channels);

if strcmpi(CHANNEL_SELECTION_MODE, 'by_split')
    if isempty(SEEN_CHANNEL_IDX) || isempty(UNSEEN_CHANNEL_IDX)
        error('CHANNEL_SELECTION_MODE="by_split" requires SEEN_CHANNEL_IDX and UNSEEN_CHANNEL_IDX to be set.');
    end
end

if ~exist(METADATA_PATH, 'file')
    error('Metadata CSV not found: %s', METADATA_PATH);
end
opts = detectImportOptions(METADATA_PATH);
if ismember('filepath', opts.VariableNames)
    opts = setvartype(opts, 'filepath', 'string');
end
meta = readtable(METADATA_PATH, opts);
num_files = height(meta);
fprintf('Metadata rows: %d\n', num_files);

if ~exist(OUTPUT_ROOT, 'dir')
    mkdir(OUTPUT_ROOT);
end

logFID = fopen(LOG_FILE, 'w');
if logFID == -1
    error('Cannot open log file for writing: %s', LOG_FILE);
end
fprintf(logFID, 'idx,rel_path,in_path,out_path,orig_fs,used_fs,channel_idx,env_file,range_m,src_z,rcv_z,Npaths,power_db\n');

if USE_PARPOOL
    error('USE_PARPOOL=true 尚未实现并发安全写入逻辑，请先设置为 false。');
end

h = waitbar(0,'WOA augmentation running...');
tic;
fprintf('Start augmentation loop...\n');

for i = 1:num_files
    rel_path = '';
    try
        rel_path    = char(meta.filepath(i));
        in_wav_path = fullfile(CLEAN_DATA_ROOT, rel_path);

        if ~exist(in_wav_path, 'file')
            warning('File missing: %s (row %d)', in_wav_path, i);
            fprintf(logFID, '%d,%s,%s,%s,%d,%d,%d,%s,%.2f,%.2f,%.2f,%d,%.3f\n', ...
                i, rel_path, in_wav_path, 'MISSING', 0, 0, -1, '', NaN, NaN, NaN, 0, NaN);
            continue;
        end

        [x, fs] = audioread(in_wav_path);
        if size(x,2) > 1
            x = mean(x, 2);
        end
        orig_fs = fs;
        if RESAMPLE_IF_MISMATCH && fs ~= TARGET_FS
            x = resample(x, TARGET_FS, fs);
            fs = TARGET_FS;
        end

        switch lower(CHANNEL_SELECTION_MODE)
            case 'random'
                ch_idx = randi(num_channels);
            case 'by_split'
                if ismember('split', meta.Properties.VariableNames)
                    split_val = string(meta.split(i));
                    if contains(lower(split_val), 'train')
                        ch_idx = SEEN_CHANNEL_IDX(randi(numel(SEEN_CHANNEL_IDX)));
                    else
                        ch_idx = UNSEEN_CHANNEL_IDX(randi(numel(UNSEEN_CHANNEL_IDX)));
                    end
                else
                    ch_idx = randi(num_channels);
                end
            otherwise
                ch_idx = randi(num_channels);
        end
        ch = ChannelPool(ch_idx);

        cfg_local       = cfg;
        cfg_local.seed  = uint32(mod(double(i)*9973 + double(ch_idx)*1315423911, 2^31-1));

        [y, Hk, cfg_used] = funOME_woa_adv(x, fs, ch, cfg_local); %#ok<NASGU>

        out_wav_path = fullfile(OUTPUT_ROOT, rel_path);
        out_dir      = fileparts(out_wav_path);
        if ~exist(out_dir, 'dir')
            mkdir(out_dir);
        end
        if WRITE_WAV
            if exist(out_wav_path, 'file') && ~OVERWRITE
                warning('Output exists and OVERWRITE=false. Skipping write: %s', out_wav_path);
            else
                audiowrite(out_wav_path, y, fs);
            end
        end

        env_file = '';
        range_m  = NaN; src_z = NaN; rcv_z = NaN; Npaths = NaN; power_db = NaN;
        if isfield(ch,'meta')
            meta_ch = ch.meta;
            if isfield(meta_ch,'arr_file'), env_file = meta_ch.arr_file; end
            if isfield(meta_ch,'range_m')
                range_m = local_extract_numeric(meta_ch.range_m, 'range_m');
                if numel(range_m) > 1, range_m = range_m(1); end
            end
            if isfield(meta_ch,'src_z_m')
                src_z = local_extract_numeric(meta_ch.src_z_m, 'src_z_m');
                if numel(src_z) > 1, src_z = src_z(1); end
            end
            if isfield(meta_ch,'rcv_z_m')
                rcv_z = local_extract_numeric(meta_ch.rcv_z_m, 'rcv_z_m');
                if numel(rcv_z) > 1, rcv_z = rcv_z(1); end
            end
        end
        if isfield(ch,'Amp')
            Amp_val = local_extract_numeric(ch.Amp, 'Amp');
            Npaths   = numel(Amp_val);
            power_db = 10*log10(sum(abs(Amp_val).^2) + eps);
        end

        fprintf(logFID, '%d,%s,%s,%s,%d,%d,%d,%s,%.2f,%.2f,%.2f,%d,%.3f\n', ...
            i, rel_path, in_wav_path, out_wav_path, orig_fs, fs, ch_idx, ...
            env_file, range_m, src_z, rcv_z, Npaths, power_db);

    catch ME
        fprintf('Error processing index %d (%s): %s\n', i, rel_path, ME.message);
        % 保持列数一致写入占位
        fprintf(logFID, '%d,%s,%s,%s,%d,%d,%d,%s,%.2f,%.2f,%.2f,%d,%.3f\n', ...
            i, rel_path, 'ERROR', 'ERROR', 0, 0, -1, 'ERROR', NaN, NaN, NaN, 0, NaN);
    end

    if mod(i,100)==0 || i==num_files
        waitbar(i/num_files, h, sprintf('%d / %d processed', i, num_files));
    end
end

fclose(logFID);
close(h);
toc;
fprintf('All done. Augmented data saved under: %s\nLog file: %s\n', OUTPUT_ROOT, LOG_FILE);

%% Helper function to extract numeric values from potentially struct-wrapped fields
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
            warning('Cannot extract numeric value from struct field: %s. Using NaN.', field_name);
            val = NaN;
        end
    else
        warning('Field %s is neither numeric nor struct (type: %s). Using NaN.', field_name, class(field));
        val = NaN;
    end
end
