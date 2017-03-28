% Dong Liu, IR&MCT, BUAA
clear; close all;clc;
tic;
dataPath = 'K:\EEG_Data\qs9\session2\fif';

dur = 4; % time period after L/R go
SP_filter = 'CAR';

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
trigger.r_go = gettrigger(dataAll(:,1) == right_go);

dataEEG = dataAll(:,2:17);
dataEEG = eeg_car(dataEEG);
f_a = [0.1 1];
N = 4;  % filter order
Wn = f_a(1)*2/fs;
[a1, b1] = butter(N, Wn, 'high');
Wn = f_a(2)*2/fs;
[a2, b2] = butter(N, Wn, 'low');
dataEEG = filtfilt(a1, b1, dataEEG);  
dataEEG = filtfilt(a2, b2, dataEEG);

for tr=1:length(trigger.l_go)    
    start = trigger.l_go(tr);
    dur = 2*fs;    
    trialsL(tr,:,:) = dataEEG(start-dur+1:start+dur,:);    
end

for tr=1:length(trigger.r_go)    
    start = trigger.r_go(tr);
    dur = 2*fs;    
    trialsR(tr,:,:) = dataEEG(start-dur+1:start+dur,:);    
end
%%
figure(1)
Cz = squeeze(trialsL(:,:,9));
a = mean(Cz);
plot(a)
hold on
plot([1024 1024], [min(a) max(a)]);
hold off
ylim([min(a) max(a)]);

fvec1 = Cz(:,1:1024);
fvec2 = Cz(:,1025:2048);
lbl1 = ones(size(fvec1,1),1);
lbl2 = ones(size(fvec2,1),1)*2;

K = 5;
featureNumber = 10;
testError_s = zeros(K,1);
cp = cvpartition(size(fvec1, 1), 'kfold', K);

for k = 1: K
    trainData1 = fvec1(find(training(cp, k)),:);
    trainLabels1 = ones(size(trainData1,1),1);
    
    trainData2 = fvec2(find(training(cp, k)),:,:); 
    trainLabels2 = ones(size(trainData2,1),1)*2;
    
    DP12 = cva_tun_opt([trainData1;trainData2],[trainLabels1;trainLabels2]);
    
    [~, featSelect] = sort(DP12,'descend');
    IndSelFeat = featSelect(1:featureNumber);
    
    dataTrain = [trainData1;trainData2];
    dataTrain = dataTrain(:,IndSelFeat);
    labelTrain = [trainLabels1;trainLabels2];
    
    testData1  = fvec1(find(test(cp, k)),:);
    testLabels1 = ones(size(testData1,1),1);
    
    testData2  = fvec2(find(test(cp, k)),:);
    testLabels2 = ones(size(testData2,1),1)*2;
    
    testData = [testData1; testData2]; %
    testData = testData(:, IndSelFeat); %
    testLabel = [testLabels1; testLabels2]; %
            
    [Class,~, post] = classify(testData, dataTrain, labelTrain, 'diaglinear');
    testError_s(k) = classerror(testLabel, Class);
end

aveError = mean(testError_s);
disp(['L Mean error is ' num2str(aveError)]);

%% 
figure(2)
Cz = squeeze(trialsR(:,:,9));
a = mean(Cz);
plot(a)
hold on
plot([1024 1024], [min(a) max(a)]);
hold off
ylim([min(a) max(a)]);

fvec1 = Cz(:,1:1024);
fvec2 = Cz(:,1025:2048);
lbl1 = ones(size(fvec1,1),1);
lbl2 = ones(size(fvec2,1),1)*2;

testError_s = zeros(K,1);
cp = cvpartition(size(fvec1, 1), 'kfold', K);

for k = 1: K
    trainData1 = fvec1(find(training(cp, k)),:);
    trainLabels1 = ones(size(trainData1,1),1);
    
    trainData2 = fvec2(find(training(cp, k)),:,:); 
    trainLabels2 = ones(size(trainData2,1),1)*2;
    
    DP12 = cva_tun_opt([trainData1;trainData2],[trainLabels1;trainLabels2]);
    
    [~, featSelect] = sort(DP12,'descend');
    IndSelFeat = featSelect(1:featureNumber);
    
    dataTrain = [trainData1;trainData2];
    dataTrain = dataTrain(:,IndSelFeat);
    labelTrain = [trainLabels1;trainLabels2];
    
    testData1  = fvec1(find(test(cp, k)),:);
    testLabels1 = ones(size(testData1,1),1);
    
    testData2  = fvec2(find(test(cp, k)),:);
    testLabels2 = ones(size(testData2,1),1)*2;
    
    testData = [testData1; testData2]; %
    testData = testData(:, IndSelFeat); %
    testLabel = [testLabels1; testLabels2]; %
            
    [Class,~, post] = classify(testData, dataTrain, labelTrain, 'diaglinear');
    testError_s(k) = classerror(testLabel, Class);
end

aveError = mean(testError_s);
disp(['R Mean error is ' num2str(aveError)]);

