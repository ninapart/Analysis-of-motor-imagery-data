function d = plotSpectrogram(mean_data, time_vect,fq_vect ,spect_title)
%plotSpectrogram receives as input the power spectrum and the time vector computed by
%spectrogram function, the frequency vector and the title of the
%spectrogram as string and plot it
    
    imagesc(time_vect,fq_vect, mean_data)
    c = colorbar;
    set(get(c,'label'),'string', 'Power/Frequency');
    colormap jet
    xlabel("Time [sec]")
    ylabel("Frequency [Hz]")
    set(gca, 'YDir','normal')
    title(spect_title);


