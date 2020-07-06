clear;
clc;
%% Setup of basic parameters
rng(10072019);
% rng(91376491);
K = 5;%randi(16)+4;%number of classes
N = 10000;%Total sample size
D = 2;
TestSz = .3; %size of verification sample

%% GMM for sampling
mu = [linspace(-K,K,K)*3.5;linspace(0,K,K)]'; %means
density = 1;
rc = .9;
for k=K:-1:1
    tmp = sprand(D,D,density,rc);
    A = full(tmp)./2;
    if k == 3
        A = [0,1;-1,0];
    end
    x = rand();
%     if (x >.5)
%         evals = [randi(5,1)*x,x];
%     else
%         evals = [x,randi(5,1)*x];
%     end
    evals = [5*x,x];
    Sig = diag(evals);
    
    sigma(:,:,k) = A'*Sig*A ; %variances
end
pStar = rand(1,K); 
pStar = pStar./sum(pStar);%mixture coefficients
gm = gmdistribution(mu, sigma, pStar);%gmm

[X,T] = random(gm,N);%total sample

figure
scatter(X(:,1),X(:,2),10,T)

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
testInputs(:,:,1,:) = reshape(Xtest',D,1,[]); %do FIX THIS!!!!

%% Make Training Data
trainIdx = training(c);
Xtrain = X(trainIdx,:);
Ttrain = T(trainIdx);

catTrain = categorical(Ttrain);
%make one hot encoded targets for training set
vecTrain = (full(ind2vec(Ttrain')))';
trainInputs(:,:,1,:) = reshape(Xtrain',D,1,[]);

%% Neural Network Training
%layer setup
inputLayer = imageInputLayer([D 1 1], 'Normalization','none');
baseLayers = [inputLayer
    fullyConnectedLayer(K*4)
    tanhLayer
    fullyConnectedLayer(K)
    softmaxLayer];

layers1 = [baseLayers
    classificationLayer];

layers2 = [baseLayers
    responsibilityLoss(K,4)];

layers3 = [baseLayers
    responsibilityLoss(K,4,'iterations',8)];

layers4 = [baseLayers
    fixedRespLoss(K,4,'ratios',pStar')];
%options setup
optionsCat = trainingOptions('adam', ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.2, ...
    'LearnRateDropPeriod',5, ...
    'MaxEpochs',25, ...
    'MiniBatchSize',64, ...
    'Verbose',true,...
    'ValidationData',{testInputs,catTest});

optionsReg = trainingOptions('sgdm', ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.2, ...
    'LearnRateDropPeriod',5, ...
    'MaxEpochs',25, ...
    'MiniBatchSize',512, ...
    'Verbose',true,...
    'ValidationData',{testInputs,vecTest});
%neural net training

rng(1024241);
 nets{1} = trainNetwork(trainInputs, catTrain, layers1, optionsCat);%need
% %categorical targets for classification layers
rng(1024241);
 nets{2} = trainNetwork(trainInputs, catTrain, layers2, optionsCat);%need 
% %numerical targets for 'regression' layers
rng(1024241);
 nets{3} = trainNetwork(trainInputs, catTrain, layers3, optionsCat);%need
%categorical targets for classification layers
rng(1024241);
 nets{4} = trainNetwork(trainInputs, catTrain, layers4, optionsCat);%need
%categorical targets for classification layers


%% Look at predictions for a different set of data
rng('default')
[Y,C]=random(gm,N/4);
Ytest(:,:,1,:) = reshape(Y',D,1,[]);

confTab = confusionTable(Ytest,C,nets,[2,2]);
% Chat1 = net1.predict(Ytest);
% Chat2 = net2.predict(Ytest);
% Chat3 = net3.predict(Ytest);
% Chat4 = net4.predict(Ytest);
% 
% [~,classHat{1}]= max(Chat1,[],2);
% [~,classHat{2}]= max(Chat2,[],2);
% [~,classHat{3}]= max(Chat3,[],2);
% [~,classHat{4}]= max(Chat4,[],2);
% 
% figure
% for i = 1:4
%     subplot(2,2,i)
%     confusionchart(C,classHat{i})
%     title(['Net',num2str(i),' Confusion'])
% end
% confusionchart(C,classHat2)
% figure
% confusionchart(C,classHat3)
% figure
% confusionchart(C,classHat4)
