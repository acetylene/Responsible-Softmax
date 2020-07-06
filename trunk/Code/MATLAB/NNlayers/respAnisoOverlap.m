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
%rng(10072019);
mu = [linspace(-K,K,K)*3.5;zeros(1,K)]'; %means
density = 1;
rc = .9;
for k=K:-1:1
    x = rand();
    if mod(k,2) == 0 
        A = [0,-1;1,0];
        evals = [5*x,x];
    else
%         tmp = sprand(D,D,density,rc);
        A = -eye(D);%[1,-1;1,1]./sqrt(2);%full(tmp);
        evals = [19*x,x];
    end
    
    Sig = diag(evals);
    
    sigma(:,:,k) = A*Sig*A' ; %variances
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
    responsibilityLoss(K,4,'iterations',4,'ratios',pStar')];

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

for i = 4:-1:1
    [~,classHat{i}]= max(nets{i}.predict(Ytest),[],2);
end

figure
for i = 1:4
    subplot(2,2,i)
    confusionchart(C,classHat{i})
    title(['Net',num2str(i),' Confusion'])
end

xmin = floor(min(X(:,1)))-1;
xmax = ceil(max(X(:,1)))+1;
ymin = floor(min(X(:,2)))-1;
ymax = ceil(max(X(:,2)))+1;
x = linspace(xmin,xmax,200);
y = linspace(ymin,ymax,200);
[U,V]=meshgrid(x,y);

W = zeros(200*200,2);
for i = 1:200
    for j = 1:200
        W(i+200*(j-1),:)=[U(i,j);V(i,j)];
    end
end

Wtest(:,:,1,:) = reshape(W',D,1,[]);

for i = 4:-1:1
    [~,regions{i}]= max(nets{i}.predict(Wtest),[],2);
end

figure
for i = 1:4
    subplot(2,2,i)
    scatter(W(:,1),W(:,2),1,regions{i},'.')
    title(['Net',num2str(i),' Class Regions'])
end 

