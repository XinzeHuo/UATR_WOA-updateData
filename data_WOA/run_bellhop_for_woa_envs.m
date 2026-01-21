% run_bellhop_for_woa_envs.m （增加预检与失败时保存 env 快照）
clear; clc;

% === 修改这里：指定您的bellhop.exe路径 ===
BELLHOP_EXE = 'E:\rcq\ToolBox\at\bin\bellhop.exe';

env_dir = 'woa_envs';
cd(env_dir);

env_files = dir('*.env');
fprintf('Found %d env files.\n', numel(env_files));

for k = 1:numel(env_files)
    [~, envName, ~] = fileparts(env_files(k).name);
    fprintf('[%d/%d] Running bellhop for %s\n', k, numel(env_files), envName);

    % 预检：跳过空文件或非常小的 env（可能是写入失败）
    finfo = dir(env_files(k).name);
    if isempty(finfo) || finfo.bytes < 20
        warning('Skipping %s: file too small or missing (%d bytes).', env_files(k).name, finfo.bytes);
        % 把文件内容（若有）存到 debug 目录以便排查
        try
            bytes = fileread(env_files(k).name);
            dbgdir = 'env_debug';
            if ~exist(dbgdir,'dir'), mkdir(dbgdir); end
            writelog = fullfile(dbgdir, sprintf('%s_snapshot.txt', envName));
            fid = fopen(writelog,'w');
            if fid ~= -1
                fprintf(fid, '--- %s (%d bytes) ---\n\n', env_files(k).name, finfo.bytes);
                fprintf(fid, '%s\n', bytes);
                fclose(fid);
            end
        catch
        end
        continue;
    end

    try
        % 如果有 bellhop.m（MATLAB 版本）：
        bellhop(envName);

        % 如果是系统命令版本，用这行代替上面那行：
        % system(sprintf('bellhop %s', envName));

    catch ME
        warning('Bellhop failed for %s: %s', envName, ME.message);
        % 保存出错 env 的快照，便于在本地用文本查看
        try
            dbgdir = 'env_debug';
            if ~exist(dbgdir,'dir'), mkdir(dbgdir); end
            copyfile([envName '.env'], fullfile(dbgdir, [envName '.env']));
        catch
        end
    end
end

cd('..');
fprintf('Done.\n');