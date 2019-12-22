function features_analysis(features, left_trials, right_trials)
%feature_analysis Extra function that plots all features violin plots,
%histograms and scatter plots - so we could examin them.
%   features - structure of all features (freqs, times and channel)
%   left_trials, right_trials - matrices of the trials' energy in each feature.
    fts_count = size(features,1);
    lgnd = {"left","right"};
    
    % Violin vs Histogram
    figure('Name', "Histograms vs Violins", 'Units', 'normalized', ...
        'Position', [0.1 0.1 0.8 0.8], 'DefaultAxesPosition', [0.1, 0.1, 0.85, 0.85]);
    n=ceil(0.5*fts_count*2); m=ceil(fts_count*2/n);
    for feat = 1:fts_count
        %histograms
        subplot(m,n,feat*2-1);
        histogram(left_trials(:,feat)); hold on;
        histogram(right_trials(:,feat));
        legend(lgnd);
        xlabel('Energy (mV^2)');
        ylabel('Trials');
        title(feat);
        %violin
    	subplot(m,n,feat*2);
        violin({left_trials(:,feat),right_trials(:,feat)}, 'xlabel', lgnd);
        ylabel('Energy (mV^2)');
        title(feat);
    end
    suptitle("Features energy distribution");
    
    %scatter
    figure('Name', "All scatter plots", 'Units', 'normalized', 'Position', [0.2 0.1 0.6 0.8],...
         'NumberTitle','off', 'DefaultAxesPosition', [0.1, 0.1, 0.85, 0.85]);
	movegui('onscreen');
    sub=1;
    for j=1:fts_count
        for i=j+1:fts_count
            subplot(fts_count-1,fts_count-1,sub);
            scatter(right_trials(:,j),right_trials(:,i), '.');hold on;
            scatter(left_trials(:,j),left_trials(:,i), '.');
            title("Features "+j+"X"+i);
            xlabel("feature "+j);
            ylabel("feature "+i);
            sub=sub+1;
        end
        sub=sub+j;
    end
    legend(lgnd);
    suplabel('Energy (mV^2)','x');
    suplabel('Energy (mV^2)','y');
end

