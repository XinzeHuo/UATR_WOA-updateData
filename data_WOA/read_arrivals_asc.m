function [amp1, delay, SrcAngle, RcvrAngle, NumTopBnc, NumBotBnc, narrmat, Pos] = read_arrivals_asc(fname, NarrMax)
% 解析 Bellhop ASCII .arr 到达文件（容错，返回矩阵形式）
% 输出：
%   amp1         : [Nch x Nmax] 复振幅（或实数振幅）
%   delay        : [Nch x Nmax] 时延（秒）
%   SrcAngle     : [Nch x Nmax] 发射角（度）
%   RcvrAngle    : [Nch x Nmax] 到达角（度）
%   NumTopBnc    : [Nch x Nmax] 顶侧反射次数
%   NumBotBnc    : [Nch x Nmax] 底侧反射次数
%   narrmat      : [Nch x 1]     每个通道的到达数
%   Pos          : struct with fields:
%                    Pos.r.r : ranges (m)
%                    Pos.s.z : src depths (m)
%                    Pos.r.z : receiver depths (m)
%
% 使用示例：
%  [amp1,delay,SrcAngle,RcvrAngle,NumTopBnc,NumBotBnc,narrmat,Pos] = ...
%      read_arrivals_asc('env_001_munk_H50_R500_sand.arr');

if nargin < 2 || isempty(NarrMax)
    NarrMax = 2000;
end
ARRIVAL_COLS = 7; % columns: amp, launch angle, travel time, angle_in/out, top_bounce, bot_bounce

fid = fopen(fname,'r');
if fid == -1
    error('Cannot open file: %s', fname);
end

% 读取所有行到 cell array（便于回溯/预览）
lines = {};
tline = fgetl(fid);
while ischar(tline)
    lines{end+1} = tline; %#ok<AGROW>
    tline = fgetl(fid);
end
fclose(fid);

if numel(lines) < 4
    error('File %s seems too short to be an arrivals file.', fname);
end

% ---------- 解析头部（基于示例格式） ----------
% 第一行包含频率等，第二行 src depth，第三行 rcv depths，第四行 ranges
nums1 = sscanf(lines{1}, '%f');
% 有��实现频率等在一行多个数值；我们只取有用项
if isempty(nums1)
    freq = NaN;
else
    freq = nums1(1);
end

src_z = sscanf(lines{2}, '%f')';
rcv_z = sscanf(lines{3}, '%f')';
ranges = sscanf(lines{4}, '%f')'; % 通常为米

% 构建 Pos 结构
Pos = struct();
Pos.r.r = ranges;
Pos.s.z = src_z;
Pos.r.z = rcv_z;

% 从第5行（或第6行）开始搜索到达块。寻找“整数字段行”后面跟随到达行的模式。
start_idx = 5;
Nlines = numel(lines);

arrival_blocks = {}; % each block = MxK numeric rows
i = start_idx;
while i <= Nlines
    s = strtrim(lines{i});
    if isempty(s)
        i = i + 1;
        continue;
    end
    nums = sscanf(s, '%f')';
    if numel(nums) == 1 && nums==floor(nums) % candidate integer describing narr
        % look ahead to next non-empty line
        j = i + 1;
        while j <= Nlines && isempty(strtrim(lines{j})), j = j + 1; end
        if j <= Nlines
            nextNums = sscanf(lines{j}, '%f')';
            narr = nums(1);
            if narr < 0 || narr > NarrMax
                i = i + 1;
                continue;
            end
            has_arrival_data     = numel(nextNums) >= 3;
            is_zero_arrival_mark = (narr == 0 && numel(nextNums) >= 1);
            % arrival line should have >= 3 numeric tokens (amp,time,...)，通常 >=6
            if has_arrival_data || is_zero_arrival_mark
                % collect arrival rows
                rows = [];
                cur = j;
                cnt = 0;
                if narr > 0
                    while cur <= Nlines && cnt < narr
                        lineA = strtrim(lines{cur});
                        if isempty(lineA)
                            cur = cur + 1;
                            continue;
                        end
                        vals = sscanf(lineA, '%f')';
                        % if this line is numeric and plausible arrival (>=3 numbers), accept
                        if numel(vals) >= 3
                            rows = [rows; vals]; %#ok<AGROW>
                            cnt = cnt + 1;
                            cur = cur + 1;
                        else
                            % unexpected token -> break (stop reading)
                            break;
                        end
                    end
                else
                    rows = zeros(0, ARRIVAL_COLS);
                end
                arrival_blocks{end+1} = rows; %#ok<AGROW>
                % advance i to cur
                i = cur;
                continue;
            else
                % next line not an arrival -> skip this integer (ambiguous)
                i = i + 1;
                continue;
            end
        else
            break;
        end
    else
        % If this line itself looks like an arrival line (sometimes files omit the integer marker),
        % then try to collect a contiguous run of arrival lines until a single-integer or EOF.
        if numel(nums) >= 3
            rows = [];
            cur = i;
            while cur <= Nlines
                la = strtrim(lines{cur});
                if isempty(la), break; end
                v = sscanf(la, '%f')';
                if numel(v) >= 3
                    rows = [rows; v]; %#ok<AGROW>
                    cur = cur + 1;
                else
                    break;
                end
            end
            if ~isempty(rows)
                arrival_blocks{end+1} = rows; %#ok<AGROW>
            end
            i = cur;
            continue;
        else
            i = i + 1;
            continue;
        end
    end
end

% 如果没有解析到任何到达块，返回空
if isempty(arrival_blocks)
    amp1 = []; delay = []; SrcAngle = []; RcvrAngle = [];
    NumTopBnc = []; NumBotBnc = []; narrmat = []; return;
end

% 将块转换为矩阵：确定最大到达数
nBlocks = numel(arrival_blocks);
maxN = max(cellfun(@(x) size(x,1), arrival_blocks));

% 预分配
amp1      = zeros(nBlocks, maxN);
delay     = zeros(nBlocks, maxN);
SrcAngle  = zeros(nBlocks, maxN);
RcvrAngle = zeros(nBlocks, maxN);
NumTopBnc = zeros(nBlocks, maxN);
NumBotBnc = zeros(nBlocks, maxN);
narrmat   = zeros(nBlocks,1);

for bi = 1:nBlocks
    rows = arrival_blocks{bi};
    narr = size(rows,1);
    narrmat(bi) = narr;
    for r = 1:narr
        vals = rows(r,:);
        % 根据常见 Bellhop-arr 列顺序做映射（基于示例）
        % 示例行格式（根据你提供的样例）：
        % amp   ???(deg?)   time   angle_in   angle_out   ntop   nbot
        % 1.82389619E-03   540.00000      0.36044961      -24.600000       24.566238               3           2
        % 因此我们采用：
        % col1 = amplitude
        % col2 = src/launch angle 或虚设项（保存到 SrcAngle）
        % col3 = travel time (delay)
        % col4 = arrival angle in/dependent (store to RcvrAngle or other)
        % col5 = maybe complementary angle (ignored or stored)
        % col6 = NumTopBnc
        % col7 = NumBotBnc
        % 如果列数少于 7，则按存在列数填充
        cnum = numel(vals);
        amp1(bi,r) = vals(1);
        if cnum >= 3
            delay(bi,r) = vals(3);
        else
            delay(bi,r) = 0;
        end
        if cnum >= 2
            SrcAngle(bi,r) = vals(2);
        end
        if cnum >= 5
            RcvrAngle(bi,r) = vals(5);
        elseif cnum >= 4
            RcvrAngle(bi,r) = vals(4);
        end
        if cnum >= 6
            NumTopBnc(bi,r) = vals(6);
        end
        if cnum >= 7
            NumBotBnc(bi,r) = vals(7);
        end
    end
end

% 将零列/行按需要裁剪（保持固定尺寸返回）
% 返回矩阵 amp1, delay 等（Nch x Nmax），narrmat (Nch x 1)
end
