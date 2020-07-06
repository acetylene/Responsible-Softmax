clear;
clc;
%% Setup of basic parameters
rng(10072019);
%rng(91376491);
K = 5;%randi(16)+4;%number of classes
N = 8192*16;%Total sample size
U = 15;
D = U^2;
TestSz = .3; %size of verification sample

%% GMM for sampling
mu = randi(20,K,D); %means
density = .4;
rc = .001;
for k=K:-1:1
    tmp = sprand(D,D,density,rc);
    sigma(:,:,k) = full(tmp*tmp')./2+.75; %variances
end
pStar = rand(1,K); 
pStar = pStar./sum(pStar);%mixture coefficients
gm = gmdistribution(mu, sigma, pStar);%gmm

[X,T] = random(gm,N);%total sample

%% Create test and training set
c = cvpartition(T,"HoldOut",TestSz);%partition for validation
%NB: it might be good to create a C-fold validation set in the future
%% Make Validation Data
testIdx = test(c);
Xtest = X(testIdx,:);
Ttest = T(testIdx);

catTest = categorical(Ttest);
%make one hot encoded targets for validation set
vecTest = (full(ind2vec(Ttest')))';
testInputs(:,:,1,:) = reshape(Xtest',U,U,[]);

%% Make Training Data
trainIdx = training(c);
Xtrain = X(trainIdx,:);
Ttrain = T(trainIdx);

catTrain = categorical(Ttrain);
%make one hot encoded targets for training set
vecTrain = (full(ind2vec(Ttrain')))';
trainInputs(:,:,1,:) = reshape(Xtrain',U,U,[]);

%% Neural Network Training
%layer setup
layers1 = [imageInputLayer([U U 1])
    convolution2dLayer(5,11)
    tanhLayer
    maxPooling2dLayer(2,'Stride',3)
    fullyConnectedLayer(K)
    softmaxLayer
    classificationLayer];

layers2 = [imageInputLayer([U U 1])
    convolution2dLayer(5,11)
    tanhLayer
    maxPooling2dLayer(2,'Stride',3)
    fullyConnectedLayer(K)
    softmaxLayer
    responsibilityLoss(K,4)];

layers3 = [imageInputLayer([U U 1])
    convolution2dLayer(5,11)
    tanhLayer
    maxPooling2dLayer(2,'Stride',3)
    fullyConnectedLayer(K,'Bias',zeros(K,1),...
        'BiasLearnRateFactor',0,'WeightL2Factor',0)
    softmaxLayer
    responsibilityLoss(K,4)];

layers4 = [imageInputLayer([U U 1])
    convolution2dLayer(5,11)
    tanhLayer
    maxPooling2dLayer(2,'Stride',3)
    fullyConnectedLayer(K,'Bias',zeros(K,1),...
        'BiasLearnRateFactor',0,'WeightL2Factor',0)
    softmaxLayer
    responsibilityLoss(K,4,'iterations',2)];
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

optionsReg = trainingOptions('sgdm', ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.2, ...
    'LearnRateDropPeriod',5, ...
    'MaxEpochs',25, ...
    'MiniBatchSize',512, ...
    'Plots','training-progress',...
    'Verbose',true,...
    'ValidationData',{testInputs,vecTest});
%neural net training

rng(1024241);
 net1 = trainNetwork(trainInputs, catTrain, layers1, optionsCat);%need
% %categorical targets for classification layers
rng(1024241);
 net2 = trainNetwork(trainInputs, catTrain, layers2, optionsCat);%need 
% %numerical targets for 'regression' layers
rng(1024241);
 net3 = trainNetwork(trainInputs, catTrain, layers3, optionsCat);%need
%categorical targets for classification layers
rng(1024241);
 net4 = trainNetwork(trainInputs, catTrain, layers4, optionsCat);%need
%categorical targets for classification layers


%% Look at predictions for a different set of data
rng('default')
[Y,C]=random(gm,N/4);
Ytest(:,:,1,:) = reshape(Y,U,U,[]);
Chat1 = net1.predict(Ytest);
Chat2 = net2.predict(Ytest);
Chat3 = net3.predict(Ytest);
Chat4 = net4.predict(Ytest);

[~,classHat1]= max(Chat1,[],2);
[~,classHat2]= max(Chat2,[],2);
[~,classHat3]= max(Chat3,[],2);
[~,classHat4]= max(Chat4,[],2);

confusionchart(C,classHat1)
figure
confusionchart(C,classHat2)
figure
confusionchart(C,classHat3)
figure
confusionchart(C,classHat4)
