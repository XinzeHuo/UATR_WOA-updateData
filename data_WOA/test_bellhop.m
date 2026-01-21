% diagnose_current_installation.m
clear; clc;

fprintf('当前Bellhop安装状态诊断\n');
fprintf('=============================\n\n');

% 检查您的现有安装
bellhop_path = 'E:\rcq\ToolBox\at\';
if exist(bellhop_path, 'dir')
    fprintf('✅ 找到声学工具箱目录: %s\n', bellhop_path);
    
    % 列出目录内容
    dirs = dir(bellhop_path);
    fprintf('目录内容:\n');
    for i = 1:min(10, length(dirs))
        if ~strcmp(dirs(i).name, '.') && ~strcmp(dirs(i).name, '..')
            fprintf('  %s\n', dirs(i).name);
        end
    end
else
    fprintf('❌ 目录不存在: %s\n', bellhop_path);
end

% 检查bin目录
bin_path = fullfile(bellhop_path, 'bin');
if exist(bin_path, 'dir')
    fprintf('\n✅ 找到bin目录\n');
    
    % 检查bellhop.exe
    bellhop_exe = fullfile(bin_path, 'bellhop.exe');
    if exist(bellhop_exe, 'file')
        fprintf('  ✅ 找到bellhop.exe\n');
    else
        fprintf('  ❌ 未找到bellhop.exe\n');
        fprintf('  当前bin目录中的文件:\n');
        files = dir(bin_path);
        for i = 1:min(5, length(files))
            if ~files(i).isdir
                fprintf('    %s\n', files(i).name);
            end
        end
    end
else
    fprintf('\n❌ bin目录不存在\n');
end