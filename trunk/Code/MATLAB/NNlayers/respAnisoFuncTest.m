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
mu = [linspace(-K,K,K)*3.5;randi(4,1,K)-2]'; %means
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
sz = [2,1];
[train_set,validate_set] = prepare_training_set(gm,N,TestSz,sz,'categorical');

%% Neural Network Training
%layer setup
inputLayer = imageInputLayer([D 1 1], 'Normalization','none');
baseLayers = [inputLayer
    fullyConnectedLayer(K*4)
    tanhLayer
    fullyConnectedLayer(K)
    softmaxLayer];

layers{1} = [baseLayers
    classificationLayer];

layers{2} = [baseLayers
    responsibilityLoss(K,4,'ratios',pStar')];

layers{3} = [baseLayers
    responsibilityLoss(K,4,'iterations',8)];

layers{4} = [baseLayers
    fixedRespLoss(K,4,'ratios',pStar')];
%options setup
options = trainingOptions('adam', ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.2, ...
    'LearnRateDropPeriod',5, ...
    'MaxEpochs',15, ...
    'MiniBatchSize',64, ...
    'Verbose',true,...
    'ValidationData',{validate_set.data,validate_set.targets});
%neural net training

seed = 1024241;
[nets] = train_nets(layers,train_set.data,train_set.targets,options,seed);

%% Look at predictions for a different set of data
rng('default')
[Y,C]=random(gm,N/4);
Ytest(:,:,1,:) = reshape(Y',D,1,[]);

confTab = confusionTable(Ytest,C,nets,[2,2]);
[acc,confMat,pcts] = test_nets(nets, Ytest, C);

xmin = floor(min(X(:,1)))-1;
xmax = ceil(max(X(:,1)))+1;
ymin = floor(min(X(:,2)))-1;
ymax = ceil(max(X(:,2)))+1;
xSamples = 1000;
ySamples = 200;
x = linspace(xmin,xmax,xSamples);
y = linspace(ymin,ymax,ySamples);
[U,V]=meshgrid(x,y);

W = zeros(xSamples*ySamples,2);
for i = 1:ySamples
    for j = 1:xSamples
        W(i+ySamples*(j-1),:)=[U(i,j);V(i,j)];
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