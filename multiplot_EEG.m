function multiplot_EEG(data, trial_indices, axis, channels, channel_txt, varargin)
%multiplot_wchannels Create figure and plot multiple signals, 
% showing both C3,C4 channels on the graph.
%   data - 3-D matrix of the EEG data - trials x signal x electrode
%   selected_trials - vector of indices to the trials to plot
%   axis - the signals' time axis (x-axis)
%   channels - the indexes of electrodes C3, C4 on the data matrix.
%   optional arguments: fig_name, fig_pos - figure parameters
    
    fig_name='default'; fig_pos='default';
    if ~isempty(varargin)
        fig_name = varargin{1};
        if length(varargin)>1
            fig_pos=varargin{2};
        end
    end
    
    figure('Name', fig_name, 'NumberTitle','off', 'Units', 'normalized', 'Position', fig_pos);
        
    legend_pos = [0.82 0.92 0.085 0.06];
    plots_amount = length(trial_indices);
    
    % determine subplot matrix dimensions
    m = ceil(sqrt(plots_amount)); n = plots_amount/m;
    % plot all
    for i = 1:plots_amount
        trial = squeeze(data(i,:,:));
        subplot(m,n,i);
        plot_lines = plot(axis, trial(:,channels(1)), axis, trial(:,channels(2)));
        title("trial #" + trial_indices(i));
        % set transperancy
        plot_lines(1).Color(4) = 0.5;
        plot_lines(2).Color(4) = 0.5;
        % show axes only on outer plots
        if mod(i,n) ~= 1
            set(gca, 'YTickLabels',[]);
        end
        if i+n <= plots_amount
            set(gca, 'XTickLabels',[]);
        end
    end
    suptitle(fig_name+" of "+plots_amount+" trials");
    suplabel('Frequency (Hz)','y');
    suplabel('Time (sec)','x');
    l = legend(plot_lines,channel_txt(1),channel_txt(2));
    l.Position = legend_pos;

end
