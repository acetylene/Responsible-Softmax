clear;
clc;
%% Setup of basic parameters
rng(10072019);
%rng(91376491);
K = 5;%randi(16)+4;%number of classes
N = 8192;%Total sample size
TestSz = .3; %size of verification sample

%% GMM for sampling
%try two equdistributed samples with different means. Try different mixing.

% mu = randi(20,K,1); %means
mu = (linspace(0,K,K)*3)';
sigma(1,1,:) = (2:K+1)./((K:-1:1)*2*K); %variances
pStar = K:-1:1; 
pStar = pStar./sum(pStar);%mixture coefficients
% pStar = mean([ones(1,K)./K;pStar]);%pull the weights closer to even
gm = gmdistribution(mu, sigma, pStar);%gmm

[X,T] = random(gm,N);%total sample

%plot data here! Groups, joint sample.
histogram(X,'Normalization','pdf')


%% Create test and training set
c = cvpartition(T,"HoldOut",TestSz);%partition for validation
%NB: it might be good to create a C-fold validation set in the future
%% Make Validation Data
testIdx = test(c);
Xtest = X(testIdx);
Ttest = T(testIdx);

catTest = categorical(Ttest);
%make one hot encoded targets for validation set
vecTest = (full(ind2vec(Ttest')))';
testInputs(1,1,1,:) = Xtest;

%% Make Training Data
trainIdx = training(c);
Xtrain = X(trainIdx);
Ttrain = T(trainIdx);

catTrain = categorical(Ttrain);
%make one hot encoded targets for training set
vecTrain = (full(ind2vec(Ttrain')))';
trainInputs(1,1,1,:) = Xtrain;

%% Neural Network Training
%layer setup
inputLayer = imageInputLayer([1 1 1], 'Normalization','none');
layers1 = [inputLayer
    fullyConnectedLayer(K*4)
    tanhLayer
    fullyConnectedLayer(K)
    softmaxLayer
    classificationLayer];

layers2 = [inputLayer
    fullyConnectedLayer(K*4)
    tanhLayer
    fullyConnectedLayer(K)
    softmaxLayer
    responsibilityLoss(K,4)];

layers3 = [inputLayer
    fullyConnectedLayer(K*4)
    tanhLayer
    fullyConnectedLayer(K,'Bias',zeros(K,1),...
        'BiasLearnRateFactor',0,'WeightL2Factor',0)
    softmaxLayer
    responsibilityLoss(K,4)];

layers4 = [inputLayer
    fullyConnectedLayer(K*4)
    tanhLayer
    fullyConnectedLayer(K)
    softmaxLayer
    fixedRespLoss(K,4,'ratios',pStar')];
%options setup
optionsCat = trainingOptions('adam', ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.2, ...
    'LearnRateDropPeriod',5, ...
    'MaxEpochs',25, ...
    'MiniBatchSize',80, ...
    'Verbose',false,...
    'ValidationData',{testInputs,catTest});

optionsReg = trainingOptions('adam', ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.2, ...
    'LearnRateDropPeriod',5, ...
    'MaxEpochs',30, ...
    'MiniBatchSize',512, ...
    'Verbose',false,...
    'ValidationData',{testInputs,vecTest});
%neural net training

rng(1024241);
nets{1} = trainNetwork(trainInputs, catTrain, layers1, optionsCat);%need
%categorical targets for classification layers
rng(1024241);
nets{2} = trainNetwork(trainInputs, catTrain, layers2, optionsCat);%need 
%numerical targets for 'regression' layers
rng(1024241);
nets{3} = trainNetwork(trainInputs, catTrain, layers3, optionsCat);%need
%categorical targets for classification layers
rng('default');
nets{4} = trainNetwork(trainInputs, catTrain, layers4, optionsCat);%need
%categorical targets for classification layers
%% Look at predictions for a different set of data
rng('default')
[Y,C]=random(gm,N/4);
Ytest(1,1,1,:) = Y;

confTab = confusionTable(Ytest,C,nets,[2,2]);
% Chat1 = net1.predict(Ytest);
% Chat2 = net2.predict(Ytest);
% Chat3 = net3.predict(Ytest);
% Chat4 = net4.predict(Ytest);
% 
% [~,classHat1]= max(Chat1,[],2);
% [~,classHat2]= max(Chat2,[],2);
% [~,classHat3]= max(Chat3,[],2);
% [~,classHat4]= max(Chat4,[],2);
% 
% figure
% confusionchart(C,classHat1)
% figure
% confusionchart(C,classHat2)
% figure
% confusionchart(C,classHat3)
% figure
% confusionchart(C,classHat4)
