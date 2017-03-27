% processing in python:
clear; close all;clc;

dataPath = 'D:\EEG_Data\ld3\session2\fif';
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
figure(2)
subplot(1,2,1);topoplot(mean(DPM12(:,[4 5]),2),chanlocs16);
subplot(1,2,2);topoplot(mean(DPM12(:,[10 11]),2),chanlocs16);
% 1:1:23  -> 4:2:48, so x*2+2,  [4 5] -> [10 12] Hz alpha; [10 11] -> [22
% 24] Hz beta
%% feature selection and classification
K = 10;
featureNumber = 30; % replaced by grid search
cp = cvpartition(size(lpsdL,1), 'kfold', K);
testError_s = zeros(K,1); % sample-based test error

% 80 49 16*23
fvecL = reshape(lpsdL, [size(lpsdL,1) size(lpsdL,2) size(lpsdL,3)*size(lpsdL,4)]);
fvecR = reshape(lpsdR, [size(lpsdR,1) size(lpsdR,2) size(lpsdR,3)*size(lpsdR,4)]);

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
    testData1 = reshape(testData1, size(testData1,1)*size(testData1,2),size(testData1,3));
    testLabels1 = ones(size(testData1,1),1);
    
    testData2  = fvecR(find(test(cp, k)),:,:);
    testData2 = reshape(testData2, size(testData2,1)*size(testData2,2),size(testData2,3));
    testLabels2 = ones(size(testData2,1),1)*2;
    
    testData = [testData1; testData2]; %
    testData = testData(:, IndSelFeat); %
    testLabels = [testLabels1; testLabels2]; %
            
    Class = classify(testData, dataTrain, labelTrain); % relaced by SVM, RM, et al
    testError_s(k) = classerror(testLabels, Class);
    
end

aveError = mean(testError_s);
disp(['Mean error is ' num2str(aveError)]);

