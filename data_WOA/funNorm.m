function y_out_norm = funNorm(y_out)
% 对信号做均值归一化处理
    y_out = y_out - mean(y_out); % 0均值处理
    
    % 归一化处理，让最大值为 0.9091 (1/1.1)
    % 避免 audioread/write 时的削波风险
    max_val = max(abs(y_out));
    if max_val > 0
        y_out_norm = y_out / max_val / 1.1; 
    else
        y_out_norm = y_out;
    end
end