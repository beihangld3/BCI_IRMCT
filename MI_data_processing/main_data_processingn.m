% processing in python:
% Dong Liu, IR&MCT, BUAA
clear; close all;clc;
<<<<<<< HEAD

<<<<<<< HEAD
dataPath = 'K:\EEG_Data\ygj\session2\fif';
=======
dataPath = 'D:\EEG_Data\qs9\session2\fif';
=======
tic;
dataPath = 'D:\EEG_Data\ld3\session2\fif';
>>>>>>> 2146926923a71a7bc8a0d7df4d120e375289c36c

>>>>>>> c6ecd40a8859c27a91a5d78c6315615326d67675
dur = 4; % time period after L/R go
SP_filter = 'Laplacian';

left_go = 11;
right_go= 9;
files = dir([dataPath '\*.mat']);
dataAll = [];
for run = 1:length(files)
    file = files(run).name;
    disp(['[MI_dataset] Loading: ' file]);
    filename = [dataPath '\' file];
    load(filename)
    dataAll = cat(1,dataAll, signals');
end

%% L:
trigger.l_go = gettrigger(dataAll(:,1) == left_go);
for tr=1:length(trigger.l_go)    
    start = trigger.l_go(tr);
    dur = 4*fs-1;    
    trialsL(tr,:,:) = dataAll(start:start+dur,2:17);    
end
for tr=1:size(trialsL,1)
    
    disp(['Processing trial ' num2str(tr)]);
    psdL(tr,:,:,:) = processTrial(squeeze(trialsL(tr,:,:)), SP_filter);    
end
lpsdL = log(psdL); % 80*49*16*23

% R
trigger.r_go = gettrigger(dataAll(:,1) == right_go);
for tr=1:length(trigger.r_go)    
    start = trigger.r_go(tr);
    dur = 4*fs-1;    
    trialsR(tr,:,:) = dataAll(start:start+dur,2:17);    
end
for tr=1:size(trialsR,1)
    
    disp(['Processing trial ' num2str(tr)]);
    psdR(tr,:,:,:) = processTrial(squeeze(trialsR(tr,:,:)), 'Laplacian');    
end
lpsdR = log(psdR);

%% feature analysis
fvec1 = reshape(lpsdL, [size(lpsdL,1)*size(lpsdL,2) size(lpsdL,3)*size(lpsdL,4)]);
% fvec1 = reshape(lpsdL, [80*49 16*23]);
lbl1 = ones(size(fvec1,1),1);
fvec2 = reshape(lpsdR, [size(lpsdR,1)*size(lpsdR,2) size(lpsdR,3)*size(lpsdR,4)]);
% fvec2 = reshape(lpsdR, [80*49 16*23]);
lbl2 = ones(size(fvec2,1),1)*2;
load('chanlocs16.mat')

DP12 = cva_tun_opt([fvec1;fvec2],[lbl1;lbl2]);
DPM12 = reshape(DP12,16,23);

figure(1)
imagesc(DPM12);
xlabel('Frequency band (Hz)');
ylabel('Channel label');
channel16 = importdata('channelLabels16.txt')'; 
set(gca,'YTick',1:16)
set(gca,'YTickLabel',channel16)
set(gca,'XTick',1:4:23)
set(gca,'XTickLabel',4:8:48)
set(gca, 'FontSize', 13)

figure(2)
topoplot(mean(DPM12(:,[4 5]),2),chanlocs16);

figure(3)
topoplot(mean(DPM12(:,[10 11]),2),chanlocs16);

% 1:1:23  -> 4:2:48, so x*2+2,  [4 5] -> [10 12] Hz alpha; [10 11] -> [22
% 24] Hz beta
%% feature selection and classification
K = 5;
featureNumber = 30; % replaced by grid search
n_trials = size(lpsdL,1) + size(lpsdR,1); % total trial numbers
cp = cvpartition(size(lpsdL,1), 'kfold', K);
testError_s = zeros(K,1); % sample-based test error
testError_t = zeros(n_trials, size(lpsdL,2), 2); % trial-based test error
testError_labels = [];

% 80 49 16*23
fvecL = reshape(lpsdL, [size(lpsdL,1) size(lpsdL,2) size(lpsdL,3)*size(lpsdL,4)]);
fvecR = reshape(lpsdR, [size(lpsdR,1) size(lpsdR,2) size(lpsdR,3)*size(lpsdR,4)]);

x = []; 
y = []; 
AUC = [];
for k = 1: K
    trainData1 = fvecL(find(training(cp, k)),:,:);
    trainData1 = reshape(trainData1, [size(trainData1,1)*size(trainData1,2),size(trainData1,3)]);
    trainLabels1 = ones(size(trainData1,1),1);
    
    trainData2 = fvecR(find(training(cp, k)),:,:);
    trainData2 = reshape(trainData2, size(trainData2,1)*size(trainData2,2),size(trainData2,3));    
    trainLabels2 = ones(size(trainData2,1),1)*2;
    
    DP12 = cva_tun_opt([trainData1;trainData2],[trainLabels1;trainLabels2]);
    
    [~, featSelect] = sort(DP12,'descend');
    IndSelFeat = featSelect(1:featureNumber);
    
    dataTrain = [trainData1;trainData2];
    dataTrain = dataTrain(:,IndSelFeat);
    labelTrain = [trainLabels1;trainLabels2];
    
    testData1  = fvecL(find(test(cp, k)),:,:);
    testData1_temp = testData1(:,:,IndSelFeat); % test Data-1 temporary
    testLabel1_temp = ones(size(testData1_temp,1),1); % test Label-1 temporary
    testData1 = reshape(testData1, size(testData1,1)*size(testData1,2),size(testData1,3));
    testLabels1 = ones(size(testData1,1),1);
    
    testData2  = fvecR(find(test(cp, k)),:,:);
    testData2_temp = testData2(:,:,IndSelFeat); % test Data-2 temporary
    testLabel2_temp = ones(size(testData2_temp,1),1)*2; % test Label-2 temporary
    testData2 = reshape(testData2, size(testData2,1)*size(testData2,2),size(testData2,3));
    testLabels2 = ones(size(testData2,1),1)*2;
    
    testData = [testData1; testData2]; %
    testData = testData(:, IndSelFeat); %
    testLabel = [testLabels1; testLabels2]; %
            
    [Class,~, post] = classify(testData, dataTrain, labelTrain); % relaced by SVM, RM, et al
    [testX, testY, T, AUC1] = perfcurve(testLabel, post(:,2), 2); 
    x = [x, testX]; 
    y = [y, testY]; 
    AUC = [AUC, AUC1];

    testError_s(k) = classerror(testLabel, Class);
    
    % trial-based collection
    testData_temp = cat(1, testData1_temp, testData2_temp);
    testLabel_temp = [testLabel1_temp; testLabel2_temp];
    testError_labels = [testError_labels; testLabel_temp];
    for i = 1: size(testData_temp, 1)
        for j = 1:size(testData_temp,2)
            temp = squeeze(testData_temp(i,j,:));
            [~,~,post] = classify(temp', dataTrain, labelTrain);
            testError_t(i+size(testData_temp,1)*(k-1), j, :) = post;
        end
    end
end

aveError = mean(testError_s);
disp(['Mean error is ' num2str(aveError)]);

meanAUC = mean(AUC);
disp(['Mean AUC is ' num2str(meanAUC)]);
chance1 = 0:0.01:1;
figure(4)
for i = 1: K 
plot(x(:,i),y(:,i)); hold on; 
end
plot(chance1, chance1,'r--','LineWidth',2);
set(gca, 'Xtick', 0:0.2:1);
xlabel('False Positive Rate');
set(gca, 'Ytick', 0:0.2:1);
ylabel('True Positive Rate');
title('S6');
s = num2str(meanAUC, '%10.3f');
str = {strcat('mean AUC = ',s)};
text(0.4, 0.1, str, 'Color','blue','FontSize',8); % 8,单独的时候换成12
set(gca, 'FontSize', 15);
%     grid on
%     grid minor
axis square
hold off;

%% BCI_command sending accuracy, with evidence accumulation
S0 = [0.5; 0.5];
alpha = 0.9; % pre-settings
threshold = 0.65; % threshold

S = zeros(n_trials, size(lpsdL,2), 2);

for k = 1:size(S,1) % all trials
    S(k,1,:) = alpha*S0 + (1-alpha)*squeeze(testError_t(k,1,:)); % post(1)
    for t = 2:size(S,2)
        S(k,t,:) = alpha*squeeze(S(k, t-1,:)) + (1-alpha)*squeeze(testError_t(k,t,:));
    end
end

% select one direction as positive, then make the decisions
S_L = squeeze(S(:,:,1));
test_decision = zeros(size(S_L,1),1);
for i = 1:size(S_L, 1)
    if max(S_L(i,:))>= threshold
        test_decision(i) = 1;
    elseif min(S_L(i,:)) <= 1-threshold
        test_decision(i) = 2;
    end
end

ACC1 = classerror(testError_labels, test_decision);
disp(['If the non-threshold-reached trials is removed: ' num2str(ACC1)]);
remove_ratio = sum(test_decision == 0)/length(test_decision);
disp(['The percentage of trials which were removed: ' num2str(remove_ratio*100), '%']);

% Don't remove the no-decision trials, just use the last sample as decision
for i = 1:length(test_decision)
    if test_decision(i) == 0
        temp = S_L(i,:);
        temp1 = temp(end);
        if temp1> 0.5
            test_decision(i)  = 1;
        elseif temp1 < 0.5
            test_decision(i) = 2;
        end
    end
end

ACC2 = classerror(testError_labels, test_decision);
disp(['If the non-threshold-reached trials is kept: ' num2str(ACC2)]);
% Not recommended, but during the online recording, this is the case!!
toc;

%% statistics on feature
% The 4-D matrix
pool = [7, 8, 10, 11];  % C1 C3 Cz C4 C2
significance = 0.0001; % 0.05
figure(5)
for i = 1:length(pool)
    ch = channel16(pool(i));
    gaL = squeeze(lpsdL(:,:,pool(i),:));
    gaR = squeeze(lpsdR(:,:,pool(i),:));
    gaL = reshape(gaL, size(gaL,1)*size(gaL,2), size(gaL,3));
    gaR = reshape(gaR, size(gaR,1)*size(gaR,2), size(gaR,3));
    p_value = [];
    for j = 1:size(gaL, 2) % channel
        [~,p] = ttest(gaL(:,j), gaR(:,j));
        p_value = [p_value p];
    end
        
    subplot(1, length(pool), i)
    plot(mean(gaL), 'r-.', 'LineWidth', 1.5);
    hold on
    plot(mean(gaR), 'b', 'LineWidth', 1.5);
    hold on
    for j = 1: length(p_value)
        if p_value(j) <= significance
            plot([j,j], [min(min(mean(gaL)), min(mean(gaR))),...
                         max(max(mean(gaL)),max(mean(gaR)))],'c', 'LineWidth', 1);
        end
    end
    set(gca,'XTick',1:4:23)
    set(gca,'XTickLabel',4:8:48)
    xlim([1,22]);
    ylim([min(min(mean(gaL)), min(mean(gaR))), max(max(mean(gaL)),max(mean(gaR)))]);
    title(ch)
    xlabel('Frequency (Hz)')
    ylabel('Magnitude (dB)')
    legend('L', 'R');
    axis square
    set(gca, 'FontSize', 10);
    hold off
end

