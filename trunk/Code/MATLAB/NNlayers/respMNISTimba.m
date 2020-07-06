clear; clc;
%% Load MNIST data
%get file structure for prepared MNIST data
s = what('MNIST');
filename = fullfile(s.path,'\','MNISTdata.mat');

load(filename)

K=size(unique(Labels),1);
[N,w,h] = size(images2d);

%% Set PARAMETERS up for training
testSz = .33; %size of verification sample

%rng('shuffle');
rng(1218763585);
scurr = rng;
%run 1  5 feb 2020:[10,10,10,1,10,5,10,10,10,5];
%run 2/3  6 feb 2020:[1,7,10,1,10,5,2,8,5,4];
ratios = [0,30.1,17.6,12.5,9.7,7.9,6.7,5.8,5.1,4.6];%benford's law;
ratiosFull = ratios./sum(ratios);
if sum(ratios>0)~=K
    K=sum(ratios>0);
    ratios = ratiosFull(ratios>0);
end

%% Neural Network Layers
%layer setup
%possible different input layer
inputLayer = imageInputLayer([w h 1], 'Normalization','none');
baseLayers = [inputLayer
    convolution2dLayer(5,11)
    tanhLayer
    maxPooling2dLayer(2,'Stride',3)
    fullyConnectedLayer(K)
    softmaxLayer];

layers{1} = [baseLayers
    classificationLayer];

layers{2} = [baseLayers
    responsibilityLoss(K,4,'ratios',ratios')];

layers{3} = [baseLayers
    responsibilityLoss(K,4,'iterations',4, 'ratios',ratios')];

layers{4} = [baseLayers
    fixedRespLoss(K,4,'ratios',ratios')];

%% Filename info
% ensure working directory is parent of testresults before running!!
% alternatively, use what()
file = what('data2');
dir = strcat(file.path,'\');%strcat(pwd,'\testResults\data2\');

accFile = strcat(dir,...
    'acc_imba_MNIST_',num2str(K),'_',num2str(scurr.Seed),'.xlsx');
confusionFile = strcat(dir,...
    'confusion_imba_MNIST_',num2str(K),'_',num2str(scurr.Seed),'.xlsx');
pctFile = strcat(dir,...
    'tpr_fpr_imba_MNIST_',num2str(K),'_',num2str(scurr.Seed),'.xlsx');

numRuns = 3;
runSz = 40;
seeds = randi([1000000,90000000],1,numRuns*runSz);
W = waitbar(0,sprintf('Calculating Run %d of %d',1,numRuns));
for i=1:numRuns
    % Establish ratios
    rng('default')
    % Prepare Training Data
    [train_set,validate_set] =...
        unbalanced_MNIST(Labels,Images,40000,testSz,ratiosFull,w,h);
    %options setup
    options = trainingOptions('adam', ...
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropFactor',0.2, ...
        'LearnRateDropPeriod',5, ...
        'MaxEpochs',5, ...
        'MiniBatchSize',100, ...
        'Verbose',false, ...
        'ValidationData',{validate_set.data,validate_set.targets});
    %Create test set
    [test1,test2] =  unbalanced_MNIST(TestLabels,TestImages,...
                                      6600,.5,ratiosFull,w,h);
        
    waitbar(i/numRuns,W,sprintf('Calculating Sample %d of %d',i,numRuns))
    H = waitbar(0,sprintf('Training set %d of %d',0,runSz));
    tic
    for j = 1:runSz
        waitbar(j/runSz,H,sprintf('Training set %d of %d',j,runSz));
        %Train Nets
        [nets] = train_nets(layers,...
            train_set.data,...
            train_set.targets,...
            options,...
            seeds(j+runSz*(i-1)));
        toc
        %Test Nets
        [acc,confMat,pcts] = test_nets(nets,...
                                       test1.data,...
                                       double(string(test1.targets)));%+1);
                                   %only use +1 for datasets with all 10
                                   %digits
        [acc2,confMat2,pcts2] = test_nets(nets,...
                                       test2.data,...
                                       double(string(test2.targets)));%+1);
        %Write results to csv file
        cellIdx = 1+runSz*K*(i-1)+K*(j-1);
        cell = strcat('A',num2str(cellIdx));
        accCell = strcat('A',num2str(j+runSz*(i-1)));
        writematrix([acc,acc2],accFile,'Range',accCell)
        writematrix([confMat,confMat2],confusionFile,'Range',cell)
        writematrix([pcts,pcts2],pctFile,'Range',cell)
    end
    close(H)
end
close(W)
