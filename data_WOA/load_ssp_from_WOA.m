function ssp = load_ssp_from_WOA(lat, lon, month, depth_grid, matfile)
% load_ssp_from_WOA  从本地 .mat（含 WOA 风格数据）提取声速剖面 (SSP)
%
% 优先使用 mat 文件中已给出的声速 3D 数组 ssp(lon,lat,depth)，
% 若不存在，则根据温度/盐度计算声速（Mackenzie 公式）。

if nargin < 5
    error('Usage: ssp = load_ssp_from_WOA(lat, lon, month, depth_grid, matfile)');
end
if isempty(month)
    month = 0;
end
depth_grid = depth_grid(:);

% ---- 加载 mat 文件 ----
if ~exist(matfile,'file')
    error('MAT file not found: %s', matfile);
end
data = load(matfile);

% 若 load 返回的顶层只有一个字段且该字段本身是 struct，则下钻一层，避免后续点索引报错
if ~isstruct(data)
    error('Loaded MAT is not a struct; dot indexing unsupported (got %s).', class(data));
end
fn = fieldnames(data);
if numel(fn)==1 && isstruct(data.(fn{1}))
    data = data.(fn{1});
    fn = fieldnames(data);
end

% ---- 自动识别或查找经纬深度向量 ----
lat_names   = {'lat','latitude','y','LAT','latitude_g','Lat'};
lon_names   = {'lon','longitude','x','LON','longitude_g','Lon'};
depth_names = {'depth','z','depths','DEPTH','Depth'};

lat_field   = '';
lon_field   = '';
depth_field = '';

for n = 1:numel(lat_names)
    idx = find(strcmpi(fn, lat_names{n}),1);
    if ~isempty(idx), lat_field = fn{idx}; break; end
end
for n = 1:numel(lon_names)
    idx = find(strcmpi(fn, lon_names{n}),1);
    if ~isempty(idx), lon_field = fn{idx}; break; end
end
for n = 1:numel(depth_names)
    idx = find(strcmpi(fn, depth_names{n}),1);
    if ~isempty(idx), depth_field = fn{idx}; break; end
end

% 如果还没找到，则自动检测
if isempty(lat_field)
    for i=1:numel(fn)
        v = data.(fn{i});
        if isnumeric(v) && isvector(v) && numel(v) > 1 && numel(v) < 10000
            if all(v >= -90 & v <= 90) && (issorted(v) || issorted(flip(v)))
                lat_field = fn{i};
                break;
            end
        end
    end
end
if isempty(lon_field)
    for i=1:numel(fn)
        v = data.(fn{i});
        if isnumeric(v) && isvector(v) && numel(v) > 1 && numel(v) < 10000
            if all(v >= -180 & v <= 360) && (issorted(v) || issorted(flip(v)))
                if ~strcmp(fn{i}, lat_field)
                    lon_field = fn{i};
                    break;
                end
            end
        end
    end
end
if isempty(depth_field)
    for i=1:numel(fn)
        v = data.(fn{i});
        if isnumeric(v) && isvector(v) && numel(v) > 1 && numel(v) < 20000
            if all(v >= 0) && (issorted(v) || issorted(flip(v)))
                if ~strcmp(fn{i}, lat_field) && ~strcmp(fn{i}, lon_field)
                    depth_field = fn{i};
                    break;
                end
            end
        end
    end
end

if isempty(lat_field) || isempty(lon_field) || isempty(depth_field)
    error('Failed to auto-detect lat/lon/depth vectors in mat file. Inspect variables: %s', strjoin(fn,', '));
end

lat_vals   = double(data.(lat_field)(:));
lon_vals   = double(data.(lon_field)(:));
depth_vals = double(data.(depth_field)(:));

% ---- 经纬度对应到 WOA 网格 ----
if max(lon_vals) > 180 && lon < 0
    lon_search = mod(lon,360);
else
    lon_search = lon;
end
[~, ilat] = min(abs(lat_vals - lat));
[~, ilon] = min(abs(lon_vals - lon_search));

% ======================================================================
%  优先路径：若 mat 中直接有声速 3D 数组 ssp(lon,lat,depth)，直接使用
% ======================================================================
if isfield(data, 'ssp')
    C3D = double(data.ssp);
    szC = size(C3D);
    ndC = ndims(C3D);

    % 根据 size 自动匹配维度：哪一维是 lon / lat / depth
    find_dim_index = @(sz, val) find(sz == val, 1, 'first');

    lon_idx   = find_dim_index(szC, numel(lon_vals));
    lat_idx   = find_dim_index(szC, numel(lat_vals));
    depth_idx = find_dim_index(szC, numel(depth_vals));

    if isempty(lon_idx) || isempty(lat_idx) || isempty(depth_idx)
        warning('Field "ssp" exists but cannot map dims to lon/lat/depth. Falling back to Temp/Sal path.');
    else
        idxC = repmat({':'}, 1, ndC);
        idxC{lon_idx}   = ilon;
        idxC{lat_idx}   = ilat;
        idxC{depth_idx} = 1:numel(depth_vals);

        try
            c_raw = squeeze(C3D(idxC{:}));
        catch ME
            warning('Indexing ssp failed: %s. Falling back to Temp/Sal path.', ME.message);
            c_raw = [];
        end

        if ~isempty(c_raw)
            % 处理为 depth 方向的一维向量
            if isvector(c_raw) && numel(c_raw) == numel(depth_vals)
                c_profile = double(c_raw(:));
            else
                ssz = size(c_raw);
                if any(ssz == numel(depth_vals))
                    dim = find(ssz == numel(depth_vals),1);
                    c_profile = double(permute(c_raw, [dim, setdiff(1:numel(ssz),dim)]));
                    c_profile = c_profile(:);
                else
                    ctmp = double(c_raw(:));
                    c_profile = interp1(linspace(0,1,numel(ctmp)), ctmp, ...
                        linspace(0,1,numel(depth_vals)), 'linear', 'extrap').';
                end
            end

            % NaN 处理
            if any(isnan(c_profile))
                ix = ~isnan(c_profile);
                if sum(ix) >= 2
                    c_profile = interp1(depth_vals(ix), c_profile(ix), depth_vals, 'linear', 'extrap');
                else
                    error('Insufficient valid c(z) points in ssp to interpolate.');
                end
            end

            % 插值到目标 depth_grid
            c_interp = interp1(depth_vals, c_profile, depth_grid, 'linear', 'extrap');

            % 去掉 NaN/Inf，避免构建失败
            valid = isfinite(depth_grid) & isfinite(c_interp);
            ssp.z = depth_grid(valid);
            ssp.c = c_interp(valid);
            if isempty(ssp.z)
                error('SSP interpolation resulted in empty profile after removing NaN/Inf.');
            end
            return;
        end
    end
end

error('No valid SSP data found in mat file; expected direct SSP data only.');

% ======================================================================
%  若没有 ssp 或使用失败，则退回到 Temp/Sal → Mackenzie 声速路径
% ======================================================================

% ---- 查找温度/盐度数组 ----
temp_cands = {'t','temp','temperature','t_an','t00','t01','t02','t03','t04','t05','t06','t07','t08','t09','t10','t11','t12','Temp'};
salt_cands = {'s','salt','salinity','s_an','s00','s01','s02','s03','s04','s05','s06','s07','s08','s09','s10','s11','s12','Sal'};

temp_field = '';
salt_field = '';
for i = 1:numel(fn)
    if any(strcmpi(fn{i}, temp_cands)), temp_field = fn{i}; end
    if any(strcmpi(fn{i}, salt_cands)), salt_field = fn{i}; end
end

% 若按名字找不到，则按维度+数值范围猜
if isempty(temp_field) || isempty(salt_field)
    for i=1:numel(fn)
        v = data.(fn{i});
        if isnumeric(v) && ndims(v) >= 2
            sz = size(v);
            if any(sz == numel(depth_vals)) && any(sz == numel(lat_vals)) && any(sz == numel(lon_vals))
                med = median(double(v(:)), 'omitnan');
                if isempty(temp_field) && med > -2 && med < 40
                    temp_field = fn{i};
                elseif isempty(salt_field) && med > 0 && med < 50
                    salt_field = fn{i};
                end
            end
        end
    end
end

if isempty(temp_field) || isempty(salt_field)
    warning('Could not robustly detect temp/salt fields by name/size. Attempting fallback: first two large arrays.');
    cnt = 0;
    for i=1:numel(fn)
        v = data.(fn{i});
        if isnumeric(v) && numel(v) > 100
            cnt = cnt + 1;
            if cnt==1 && isempty(temp_field), temp_field = fn{i}; end
            if cnt==2 && isempty(salt_field), salt_field = fn{i}; break; end
        end
    end
end

if isempty(temp_field) || isempty(salt_field)
    error('Failed to detect temperature or salinity arrays. Inspect your .mat contents.');
end

T = data.(temp_field);
S = data.(salt_field);

szT = size(T);
ndT = ndims(T);

find_dim_index = @(sz, val) find(sz == val, 1, 'first');

d_idx     = find_dim_index(szT, numel(depth_vals));
lat_idx_T = find_dim_index(szT, numel(lat_vals));
lon_idx_T = find_dim_index(szT, numel(lon_vals));
month_idx = find_dim_index(szT, 12);

% 如有缺失，用 S 的 size 再猜
if isempty(d_idx) || isempty(lat_idx_T) || isempty(lon_idx_T)
    szS = size(S);
    if isempty(d_idx),     d_idx     = find_dim_index(szS, numel(depth_vals)); end
    if isempty(lat_idx_T), lat_idx_T = find_dim_index(szS, numel(lat_vals));   end
    if isempty(lon_idx_T), lon_idx_T = find_dim_index(szS, numel(lon_vals));   end
    if isempty(month_idx), month_idx = find_dim_index(szS, 12);                end
end

if isempty(d_idx) || isempty(lat_idx_T) || isempty(lon_idx_T)
    error('Could not robustly map temp/salt array dimensions to depth/lat/lon in the .mat file. Inspect variable sizes.');
end

idxT = repmat({':'}, 1, ndT);
idxS = repmat({':'}, 1, ndims(S));

idxT{lat_idx_T} = ilat;
idxT{lon_idx_T} = ilon;
idxS{lat_idx_T} = ilat;
idxS{lon_idx_T} = ilon;

if month ~= 0 && ~isempty(month_idx)
    idxT{month_idx} = month;
    idxS{month_idx} = month;
else
    if ~isempty(month_idx)
        idxT{month_idx} = 1;
        idxS{month_idx} = 1;
    end
end

try
    T_profile_raw = squeeze(T(idxT{:}));
catch ME
    error('Failed to index temperature array with derived indices. Message: %s', ME.message);
end
try
    S_profile_raw = squeeze(S(idxS{:}));
catch ME
    error('Failed to index salinity array with derived indices. Message: %s', ME.message);
end

% ---- 调整为 depth 向量 ----
if isvector(T_profile_raw) && numel(T_profile_raw) == numel(depth_vals)
    T_profile = double(T_profile_raw(:));
else
    ssz = size(T_profile_raw);
    if any(ssz == numel(depth_vals))
        dim = find(ssz == numel(depth_vals),1);
        T_profile = double(permute(T_profile_raw, [dim, setdiff(1:numel(ssz),dim)]));
        T_profile = T_profile(:);
    else
        Ttmp = double(T_profile_raw(:));
        T_profile = interp1(linspace(0,1,numel(Ttmp)), Ttmp, linspace(0,1,numel(depth_vals)), 'linear', 'extrap')';
    end
end

if isvector(S_profile_raw) && numel(S_profile_raw) == numel(depth_vals)
    S_profile = double(S_profile_raw(:));
else
    ssz = size(S_profile_raw);
    if any(ssz == numel(depth_vals))
        dim = find(ssz == numel(depth_vals),1);
        S_profile = double(permute(S_profile_raw, [dim, setdiff(1:numel(ssz),dim)]));
        S_profile = S_profile(:);
    else
        Stmp = double(S_profile_raw(:));
        S_profile = interp1(linspace(0,1,numel(Stmp)), Stmp, linspace(0,1,numel(depth_vals)), 'linear', 'extrap')';
    end
end

% ---- NaN 处理 ----
if any(isnan(T_profile))
    ix = ~isnan(T_profile);
    if sum(ix) < 2
        error('Insufficient valid temperature points to interpolate.');
    end
    T_profile = interp1(depth_vals(ix), T_profile(ix), depth_vals, 'linear', 'extrap');
end
if any(isnan(S_profile))
    ix = ~isnan(S_profile);
    if sum(ix) < 2
        error('Insufficient valid salinity points to interpolate.');
    end
    S_profile = interp1(depth_vals(ix), S_profile(ix), depth_vals, 'linear', 'extrap');
end

% ---- 插值到目标 depth_grid ----
T_interp = interp1(depth_vals, T_profile, depth_grid, 'linear', 'extrap');
S_interp = interp1(depth_vals, S_profile, depth_grid, 'linear', 'extrap');

% ---- Mackenzie (1981) 声速公式（m/s） ----
Tg = T_interp(:);
Sg = S_interp(:);
Zg = depth_grid(:);

c = 1448.96 + 4.591.*Tg - 5.304e-2.*Tg.^2 + 2.374e-4.*Tg.^3 + 1.340.*(Sg - 35) ...
    + 1.630e-2.*Zg + 1.675e-7.*Zg.^2 - 1.025e-2.*Tg.*(Sg - 35) - 7.139e-13.*Tg.*(Zg.^3);

ssp.z = depth_grid(:);
ssp.c = c(:);

end
