clear;
clc;
%% Setup of basic parameters
rng(10072019);
%rng(91376491);
K = 5;%randi(9)+1;%number of classes
N = 8192;%Total sample size
TestSz = .3; %size of verification sample

%% GMM for sampling
%try two equdistributed samples with different means. Try different mixing.

mu = linspace(-50*K,50*K,K); %means
% sigma = rand(1,1,K)/2+1; %variances
pStar = rand(K,1); 
pStar = pStar./sum(pStar);%mixture coefficients
% gm = gmdistribution(mu, sigma, pStar);%gmm
% 
% [X,T] = random(gm,N);%total sample

for ii = K:-1:1
    sample{ii} = randi([mu(ii)-10,mu(ii)+10],1,round(N*pStar(ii)));
    labels{ii} = ones(1,round(N*pStar(ii)))*ii;
end

X = cell2mat(sample);
T = cell2mat(labels);
%plot data here! Groups, joint sample.

histogram(X,K*20)

%% Create test and training set
c = cvpartition(T,"HoldOut",TestSz);%partition for validation
%NB: it might be good to create a C-fold validation set in the future
%% Make Validation Data
testIdx = test(c);
Xtest = X(testIdx);
Ttest = T(testIdx);

catTest = categorical(Ttest);
%make one hot encoded targets for validation set
vecTest = (full(ind2vec(Ttest)));
testInputs(1,1,1,:) = Xtest;

%% Make Training Data
trainIdx = training(c);
Xtrain = X(trainIdx);
Ttrain = T(trainIdx);

catTrain = categorical(Ttrain);
%make one hot encoded targets for training set
vecTrain = (full(ind2vec(Ttrain)));
trainInputs(1,1,1,:) = Xtrain;

%% Neural Network Training
%layer setup
layers1 = [imageInputLayer([1 1 1])
    fullyConnectedLayer(K*4)
    tanhLayer
    fullyConnectedLayer(K)
    softmaxLayer
    classificationLayer];

layers2 = [imageInputLayer([1 1 1])
    fullyConnectedLayer(K*4)
    tanhLayer
    fullyConnectedLayer(K,'Bias',zeros(K,1),...
        'BiasLearnRateFactor',0,'WeightL2Factor',0)
    responsibilityLayer(K,4)
    respCEntLayer];

layers3 = [imageInputLayer([1 1 1])
    fullyConnectedLayer(K*4)
    tanhLayer
    fullyConnectedLayer(K,'Bias',zeros(K,1),...
        'BiasLearnRateFactor',0,'WeightL2Factor',0)
    softmaxLayer
    responsibilityLoss(K,4)];

layers4 = [imageInputLayer([1 1 1])
    fullyConnectedLayer(K*4)
    tanhLayer
    fullyConnectedLayer(K)
    softmaxLayer
    fixedRespLoss(K,4,'ratios',pStar)];
%options setup
optionsCat = trainingOptions('adam', ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.2, ...
    'LearnRateDropPeriod',5, ...
    'MaxEpochs',25, ...
    'MiniBatchSize',64, ...
    'Plots','training-progress',...
    'Verbose',true,...
    'ValidationData',{testInputs,catTest});

optionsReg = trainingOptions('adam', ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.2, ...
    'LearnRateDropPeriod',5, ...
    'MaxEpochs',25, ...
    'MiniBatchSize',512, ...
    'Plots','training-progress',...
    'Verbose',true,...
    'ValidationData',{testInputs,vecTest'});
%neural net training

rng(1024241);
 net1 = trainNetwork(trainInputs, catTrain, layers1, optionsCat);%need
% %categorical targets for classification layers
rng(1024241);
 net2 = trainNetwork(trainInputs, vecTrain', layers2, optionsReg);%need 
% %numerical targets for 'regression' layers
rng(1024241);
 net3 = trainNetwork(trainInputs, catTrain, layers3, optionsCat);%need
%categorical targets for classification layers
rng('default');
 net4 = trainNetwork(trainInputs, catTrain, layers4, optionsCat);%need
%categorical targets for classification layers
%% Look at predictions for a different set of data
rng('default')
for ii = K:-1:1
    Ysample{ii} = randi([mu(ii)-10,mu(ii)+10],1,round(N/4*pStar(ii)));
    Ylabels{ii} = ones(1,round(N/4*pStar(ii)))*ii;
end

Y = cell2mat(Ysample);
C = cell2mat(Ylabels);

Ytest(1,1,1,:) = Y;
Chat1 = net1.predict(Ytest);
Chat2 = net2.predict(Ytest);
Chat3 = net3.predict(Ytest);
Chat4 = net4.predict(Ytest);

[~,classHat1]= max(Chat1,[],2);
[~,classHat2]= max(Chat2,[],2);
[~,classHat3]= max(Chat3,[],2);
[~,classHat4]= max(Chat4,[],2);

figure
confusionchart(C,classHat1)
figure
confusionchart(C,classHat2)
figure
confusionchart(C,classHat3)
figure
confusionchart(C,classHat4)
