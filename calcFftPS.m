function PS = calcFftPS(signal)
% Calculate FFT and return power spectra
%
% PS = calcFft(signal)
%         Calculate the fft and power spectra like we've seen in the class
%         exercise.

    % FFt&PS calculations
    L = length(signal);
    Amp = abs(fft(signal));
    PS = Amp .^ 2;
    PS = [PS(1) 2*PS(2:floor(L/2))];
    
end

