% generate_woa_envs.m
% 批量生成 Bellhop environment (.env) 文件并调用 bellhop 运行
% 修改版：增强参数扫描，确保生成有效文件

clear; clc;

% --- 配置路径 ---
% 请确保该路径下有真实的 .mat 文件
matfile   = 'E:\rcq\pythonProject\Data\WOA23_mat\woa23_00.mat'; 
out_folder = 'woa_envs';
if ~exist(out_folder,'dir'), mkdir(out_folder); end

% --- 参数配置 ---
% 1. SSP 类型（对应不同的月份或位置）
SSP_TYPES = {'munk', 'summer_shallow', 'winter_shallow', 'deep_channel'};

% 2. 几何配置
% H_list: 水深 (m)
% R_list: 收发水平距离 (m). 增加了一些近距离点以确保一定有路径
H_list    = [50, 200, 1000];                    
R_list    = [500, 2000, 5000, 10000];              

% 3. 底质
BOT_list  = {'sand', 'mud'};

env_idx   = 1;
success_count = 0;

% 添加 bellhop 工具箱路径（如果需要）
% addpath('你的/at/工具箱/路径');

% 可选：显式指定 bellhop.exe 路径（为空时使用系统 PATH 或 MATLAB 版 bellhop）
BELLHOP_EXE = '';
if ~isempty(BELLHOP_EXE) && exist(BELLHOP_EXE, 'file') ~= 2
    error('BELLHOP_EXE not found: %s', BELLHOP_EXE);
end

for s = 1:numel(SSP_TYPES)
    for h = 1:numel(H_list)
        depth_val = H_list(h);
        
        % 提取经纬度
        [lat, lon, month] = pick_lat_lon_for_ssp(SSP_TYPES{s});
        
        % 生成深度网格：至少每 10m 一个点，保证插值精度
        depth_grid = unique([0:10:depth_val, depth_val])'; 

        % 获取 SSP
        try
            ssp = load_ssp_from_WOA(lat, lon, month, depth_grid, matfile);
        catch ME
            fprintf('Skip %s: %s\n', SSP_TYPES{s}, ME.message);
            continue;
        end

        % 设置收发深度
        % Src: 100-1100 m (6 sampling points)
        % Rcv: 固定 10 m
        src_z = 100:200:1100;
        max_allowable_depth = max(1, min(depth_val - 1, max(ssp.z)));
        src_z = src_z(src_z <= max_allowable_depth);
        if isempty(src_z)
            warning('All src_z exceed max depth %.1f m; using max depth instead.', max_allowable_depth);
            src_z = max_allowable_depth;
        end
        rcv_z = 10;
        
        % 对每个距离生成一个 env
        for r = 1:numel(R_list)
            range_val = R_list(r);
            
            % 如果距离远大于水深很多倍（例如 50m 水深传 10km），路径可能很少
            % 但我们依然生成，看看 Bellhop 能不能算出来
            
            for b = 1:numel(BOT_list)
                bot_type = BOT_list{b};

                envName = sprintf('env_%03d_%s_H%d_R%d_%s', ...
                    env_idx, SSP_TYPES{s}, depth_val, range_val, bot_type);
                envPath = fullfile(out_folder, envName);
                
                % 生成 .env
                try
                    write_bellhop_env_woa(envPath, ssp, depth_val, bot_type, src_z, rcv_z, range_val);
                catch ME
                    warning('Write failed: %s', ME.message);
                    continue;
                end
                
                % 运行 Bellhop
                % 注意：必须确保 bellhop.exe 在系统路径中，或有 MATLAB 版 bellhop
                run_status = 1;
                try
                    % 目录以避免文件名长度问题，且 bellhop 默认在当前目录找 .env
                    old_dir = pwd;
                    cd(out_folder);

                    % 调用 Bellhop (优先 MATLAB 版，再使用 exe)
                    ran_bellhop = false;
                    if exist('bellhop', 'file') == 2
                        try
                            bellhop(envName);
                            ran_bellhop = true;
                        catch ME
                            warning('MATLAB bellhop failed: %s', ME.message);
                        end
                    end
                    env_arg = envName;
                    if isempty(regexp(env_arg, '^[\w-]+$', 'once'))
                        error('Invalid env name for bellhop: %s', env_arg);
                    end
                    if ~ran_bellhop
                        bellhop_cmd = 'bellhop.exe';
                        if ~isempty(BELLHOP_EXE)
                            bellhop_cmd = BELLHOP_EXE;
                        end
                        if contains(bellhop_cmd, '"')
                            error('BELLHOP_EXE contains invalid quotes: %s', bellhop_cmd);
                        end
                        status = system(sprintf('"%s" "%s"', bellhop_cmd, env_arg));
                        if status ~= 0 && isempty(BELLHOP_EXE)
                            warning('bellhop.exe not found in PATH; set BELLHOP_EXE to its full path.');
                        end
                    end

                    % 简单的检查：看是否生成了 .arr 文件
                    if exist([env_arg '.arr'], 'file')
                        success_count = success_count + 1;
                    else
                        run_status = 0;
                        % fprintf('Bellhop run finished but no .arr for %s\n', envName);
                    end

                    cd(old_dir);
                catch ME
                    cd(old_dir);
                    warning('Bellhop run failed for %s: %s', envName, ME.message);
                    run_status = 0;
                end
                
                env_idx = env_idx + 1;
            end
        end
    end
end

fprintf('All tasks done. Total Envs: %d, Success .arr: %d\n', env_idx-1, success_count);
