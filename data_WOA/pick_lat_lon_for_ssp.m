function [lat, lon, month] = pick_lat_lon_for_ssp(ssp_type)
% pick_lat_lon_for_ssp  根据语义 ssp_type 返回代表的 (lat, lon, month)
% ssp_type: 字符串，例如 'munk','summer_shallow','isothermal','winter_shallow','deep_channel'
% 返回:
%   lat, lon (degrees), month (1..12; 0 表示年平均)

switch lower(ssp_type)
    case 'munk'
        % 深海 Munk 型（北太平洋中部示例）
        lat = 30.0; lon = -160.0; month = 0;   % 年平均
    case 'summer_shallow'
        % 夏季表层温跃层（近岸大陆架示例）
        lat = 36.6; lon = -122.0; month = 8;   % August (夏季)
    case 'isothermal'
        % 等温/低梯度（高纬或混合区）
        lat = 60.0; lon = -30.0; month = 0;    % 年平均
    case 'winter_shallow'
        % 冬季浅海（混合表层）
        lat = 34.0; lon = 125.0; month = 1;    % January (冬季)
    case 'deep_channel'
        % 深水声道示例
        lat = -20.0; lon = 160.0; month = 0;
    otherwise
        % 默认使用赤道附近
        lat = 0.0; lon = 0.0; month = 0;
end

end
