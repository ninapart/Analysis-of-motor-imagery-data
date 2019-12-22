function time_vector = time_window(time_range,fs)
%time_window computes the time vector that correspond to a specific time
%range and sampling rate

time_vector = floor(time_range(1)*fs:time_range(2)*fs);
if time_vector(1)==0
    time_vector=time_vector(2:end);
end

