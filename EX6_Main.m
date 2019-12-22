close all
clear 
clc


%% Visualization
motor_data = 'motor_imagery_data_2019.mat';
load(motor_data);  % variable P_C_S 
C3 = 1; % C3, C4 are the electrodes we are recording from
C4 = 2;
electrodes = [C3,C4];
classes_text = ["Left", "Right"];
elec_text = ["C3", "C4"];
fs = 128; % Hz
sampling_time = 6; %sec
vis_amount = 20; %visualize some trials to eyeball for difference
% visualization figures parameters
left_text = 'EEG: left hand imagery'; left_pos = [0.1 0.2 0.4 0.6];
right_text = 'EEG: right hand imagery'; right_pos = [0.5 0.2 0.4 0.6];

% get left&right attribute indices from the data
left_att_index = contains(P_C_S.attributename,'LEFT');
right_att_index = contains(P_C_S.attributename,'RIGHT');
sig_length = size(P_C_S.data,2);

left_trials = find(P_C_S.attribute(left_att_index,:)==1);
right_trials = find(P_C_S.attribute(right_att_index,:)==1);
trials_num = size(P_C_S.data,1);
side_trials_num = length(left_trials); % assuming we have same amount for left and right.


% Plot vis_amount random trials from each hand
time_axis = 0:1/fs:(sampling_time - 1/fs);
plot_EEG_sample(P_C_S.data(left_trials,:,electrodes),vis_amount, time_axis, electrodes, elec_text, left_text, left_pos);
plot_EEG_sample(P_C_S.data(right_trials,:,electrodes),vis_amount, time_axis, electrodes, elec_text,right_text, right_pos);

% clear section's temporary variables
clear motor_data left_att_index right_text left_text left_visualize ...
    right_att_index right_visualize vis_amount sampling_time time_axis right_pos left_pos


%% Analysis

imagination_window = [2 6]; % start and end time in seconds
time_vect = time_window(imagination_window, fs); % reduce the time window vector
L=length(time_vect); %length of the signal
fq = (1:floor(L/2))*fs/L; %frequency vector

%pwelch consts
window = 2*fs; % 2 times the sampling gap
overlap = [];

figure('Name', 'Power Spectra FFT/PWELCH - C3/C4', 'Units', 'normalized', 'Position', [0.25 0.2 0.4 0.6],...
     'NumberTitle','off', 'DefaultAxesPosition', [0.1, 0.15, 0.85, 0.8]);

% Plot fft/pwelch means- per class and electrode (8 means in total):
% compute fft and pwelch power spectrum for each trial, sum and calc the mean. 
% Then plot it!
subp_ind = 1;
for elec = elec_text
    % First, get each class (right/left) raw data.
    data.(classes_text(1)).(elec).raw = P_C_S.data(left_trials, :, eval(elec));
    data.(classes_text(2)).(elec).raw = P_C_S.data(right_trials, :, eval(elec));
    % Compute trials' power spectras for both methods- fft and pwelch
    for method = ["fft" "pwelch"]
        for class = classes_text
            data.(class).(elec).(method) = [];
            % calc and sum all trials' power spectra of the imagination time
            for i = 1:side_trials_num
                switch method
                    case "fft"
                        PS = calcFftPS(data.(class).(elec).raw(i,time_vect));
                    case "pwelch"
                        PS = pwelch(data.(class).(elec).raw(i,time_vect), window, overlap, fq, fs);
                end

                if isempty(data.(class).(elec).(method)) 
                    data.(class).(elec).(method) = PS;
                else
                    data.(class).(elec).(method) = data.(class).(elec).(method) + PS;
                end
            end
            % calc the mean 
            data.(class).(elec).(method) = data.(class).(elec).(method)/side_trials_num;
        end
        % plot the mean PS.
        % Each graph shows left and right spectra.
        subplot(2,2,subp_ind)
        plot(fq, data.(classes_text(1)).(elec).(method), fq, data.(classes_text(2)).(elec).(method));
        title(method+" - "+elec);
        legend(classes_text);
        xlim([0 30]);
        subp_ind = subp_ind + 2; % we want each method on different row
        switch method
            case "fft"
                ylabel("Magnitude (mV^2)");
            case "pwelch"
                ylabel("Magnitude (mV^2/Hz)");
        end
    end
    subp_ind = subp_ind - 3; % back to the first row
end
suptitle("Power Spectra Comparison")
suplabel('Frequency (Hz)','x');

       
% Spectrograms

% Take only frequencies in relevant range
fq = (1:floor(sig_length/2))*fs/sig_length;
f_range = fq <= 40 ;
fq = fq(f_range);

%spectrograms parameters
window = floor(fs/2);
overlap = floor(0.7*window); % Prettier than 1/2

figure('Name', "Spectrograms", 'Units', 'normalized', 'Position', [0.22 0.3 0.56 0.5], ...
     'NumberTitle','off', 'DefaultAxesPosition', [0.07, 0.1, 0.9, 0.85])

% update our beautiful data structure with the spectrograms.  
%   *we know this can be done in the fft&pwelch loop, but we wanted a
%    seperation in the code.
subp_ind = 1;
for elec = elec_text
	for class = classes_text
        % sum all spectograms
        data.(class).(elec).spectrogram = [];
        for i = 1:side_trials_num
            [~,~,time_vect,spect] = spectrogram(data.(class).(elec).raw(i,:),window,overlap,fq,fs,'yaxis');
            if isempty(data.(class).(elec).spectrogram)
                data.(class).(elec).spectrogram = spect;
            else
                data.(class).(elec).spectrogram = data.(class).(elec).spectrogram+spect;
            end
        end
        % calculate mean
        data.(class).(elec).spectrogram = data.(class).(elec).spectrogram/side_trials_num;
        
        % plot mean spectrogram. we're log'ing it because its clearer to
        % see the differences
        subplot(2,3,subp_ind);
        plotSpectrogram(log(data.(class).(elec).spectrogram), time_vect, fq, class+" - "+elec);
        subp_ind = subp_ind+1;
	end
    
    %plot the difference spectrograms for each electrode C3/C4
    subplot(2, 3, subp_ind);
    %log the spectrograms, substract and abs- for a clear representation.
    diff = log(data.(classes_text(1)).(elec).spectrogram) - log(data.(classes_text(2)).(elec).spectrogram);
    plotSpectrogram(diff, time_vect, fq, "Diff (left-right) "+elec);
    subp_ind = subp_ind+1;
end


clear L fq fft_data pwelch_data i imagination_window method overlap sig_length subpind tim_vect ...
    f_range spect time_vect


%% Extracted features analysis
% Our features:
%    freq range |  time  | channel
fts = {[14 18.5], [3.8 6],   C3;
       [14 18.5], [3.8 6],   C4;
       [16.5 20], [2 3],     C4;
       [8 13],    [4.5 6],   C4;
       [19 21],   [4 6],     C3;
       [30 38],   [3.5 6],   C4;
%        [27  30],   [5 6],     C4; % High perc alone, bad togather :(
%       [8.1 11.9],   [4 5.4],  C3; % Chance level
};
% chosen features to violin plot
violin_plot_fts = [1 2 4];

% Put our features in a structure we can work with
features = struct('freq_range', fts(:,1), 'time_range', fts(:,2), 'channel', fts(:,3));
% Initialize matrix of energy values for the trials x features
features_energy = zeros(trials_num, size(features,1)); 
trials_labels = zeros(trials_num, 1); % labels vector

window = fs/4;
overlap = [];
% for each trial x feature "cut" the specify time range in the raw
% data, check its label, computes its power spectrum and its energy
for i = 1: trials_num
    for feature_i = 1 : size(features,1)
        feature = features(feature_i);
        feature_data = P_C_S.data(i, time_window(feature.time_range, fs), feature.channel);
        trials_labels(i) = ismember(i, right_trials)*2-1; % set label as 1 if right, -1 if left
         
        [pxx, fq] = pwelch(feature_data, window, overlap, [], fs);
        features_energy(i, feature_i) = bandpower(pxx,fq,feature.freq_range,'psd');
    end
end


% Plot energy distribution 
if ~isempty(violin_plot_fts)
    figure('Name', "Energy distribution", 'Units', 'normalized', 'Position', [0.25 0.3 0.5 0.5],...
         'NumberTitle','off', 'DefaultAxesPosition', [0.1, 0.1, 0.85, 0.85]);
    % Plot each of the features stated in vector 'violin_plot_fts'
    for i = 1:length(violin_plot_fts)
        ft = violin_plot_fts(i);
        subplot(1,length(violin_plot_fts),i);
        violin({features_energy(left_trials,ft),features_energy(right_trials,ft)}, 'xlabel', classes_text);
        title("Feature "+ft);
    end
    suplabel('Energy (mV^2)','y');
end


% BONUS - all feature analysis. It's default in comment in order to not burden
% features_analysis(features,features_energy(left_trials,:), features_energy(right_trials,:));

clear i ft feature_i feature window violin_plot_fts subp_ind ...
    side_trials_num pxx overlap diff fq fts feature_data elec class

%% Classification
% Train on selected percentage of the trials
train_perc = 0.7;
train_amount = floor(trials_num*train_perc);
realizations = 70;
validation_succ_perc = zeros(realizations,1);%vector of the validation succes percentage
valid_feat = zeros(realizations, length(features));%vector of the validation success percentage per feature 

%realize the training and validation for many times to check the accuracy
%and standard deviation means of our features.
for i = 1 : realizations
    % randomly choose training set
    train_indices = datasample(1:trials_num, train_amount, 'Replace', false);
    train_trials = features_energy(train_indices,:);
    train_labels = trials_labels(train_indices);
    % validate on the rest of the trials
    val_indices = setdiff(1:trials_num, train_indices);
    val_labels = trials_labels(val_indices);
    validation_trials = features_energy(val_indices,:);
    
    % Train! using all features
    [W, b, y_pred] = LDA1(train_trials, train_labels);
    % calculate train error
    train_success = sum(y_pred == train_labels.'); 
    
    % note the realization's classification success rate
    validation_succ_perc(i) = LDA_validate(W, b, validation_trials, val_labels);

    % get each feature's success rate (using them seperately)
    for feature = 1:length(features)
        [W, b, ~] = LDA1(train_trials(:,feature), train_labels);
        valid_feat(i,feature) = LDA_validate(W, b,validation_trials(:,feature), val_labels); 
    end
end
% Calculate means and stds
total_accuracy  = mean(validation_succ_perc)*100;
total_features_std = std(validation_succ_perc)*100;

feature_accuracy = num2cell(mean(valid_feat)*100);
feature_std = num2cell(std(valid_feat)*100);
[features.accuracy] = feature_accuracy{:};
[features.std] = feature_std{:};

% print and save results
features_table = struct2table(features);
fprintf("Results for %d realizations\n"+ ...
 "Accuracy: %.3f, std: %.2f\n\n"+ ...
 "Features:\n", realizations, total_accuracy, total_features_std);
disp(features_table);

save('ex6_results_and_data.mat','features_energy','trials_labels','features_table', 'total_accuracy', 'total_features_std');



