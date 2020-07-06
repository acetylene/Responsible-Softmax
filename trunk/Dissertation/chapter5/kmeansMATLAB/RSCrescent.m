clear; clc;
%% Create a univariate GMM with 4 clusters two majority, two minority classes
K = 4;%Number of clusters
N = 5000;% number of samples
D = 2;
TestSz = .1;
sz = [2 1];
mu = linspace(-4,4,K)';
sigma = 1;
p = [.4 .15 .05 .4];

gm = gmdistribution(mu, sigma, p);
rng('default');%for repeatability
%% Data to train on and test on.
%ERR changes internal spread of clusters.
%When ERR is smaller more points lie on semicircle
err = 2e-1;
%SHIFT changes how close the clusters are together.
%Positive values of SHIFT move clusters closer, negative further.
shift = -7.3;
%[train_set,validate_set] = crescentDataRS(K,N,gm,err,shift);

[testData,testLabels] = crescentDataRS(K,N/10,gm,err,shift);

figure
scatter(testData(:,1),testData(:,2),15,testLabels,'filled')
axis equal

%% Layers to train
inputLayer = imageInputLayer([D 1 1], 'Normalization','none');
baseLayers = [inputLayer
    fullyConnectedLayer(K*4)
    tanhLayer
    fullyConnectedLayer(K)
    softmaxLayer];

layers{1} = [baseLayers
    classificationLayer];

layers{2} = [baseLayers
    responsibilityLoss(K,4,'ratios',p')];

layers{3} = [baseLayers
    responsibilityLoss(K,4,'iterations',4, 'ratios',p')];

layers{4} = [baseLayers
    fixedRespLoss(K,4,'ratios',p')];

%% Create Test and Validation Data


% Prepare Training Data
[train_set,validate_set] =...
    prep_train_crescents(gm,N,TestSz,sz,'categorical',err,shift);
%options setup
options = trainingOptions('adam', ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.2, ...
    'LearnRateDropPeriod',5, ...
    'MaxEpochs',15, ...
    'MiniBatchSize',100, ...
    'Verbose',false, ...
    'ValidationData',{validate_set.data,validate_set.targets});
%Create test set
rng('default')

Ytest(:,:,1,:) = reshape(testData',D,1,[]);

%Train Nets
[nets] = train_nets(layers,...
    train_set.data,...
    train_set.targets,...
    options,...
    rng);

%Test Nets
[acc,confMat,pcts] = test_nets(nets, Ytest, testLabels);
%plot confusion matrices
figure
for ii = 1:K
subplot(2,2,ii)
confusionchart(squeeze(confMat(:,:,ii)));
title(sprintf('Net %d Confusion',ii))
end

%prepare classification regions for plotting
xmin = floor(min(testData(:,1)))-1;
xmax = ceil(max(testData(:,1)))+1;
ymin = floor(min(testData(:,2)))-1;
ymax = ceil(max(testData(:,2)))+1;
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
%plot classification regions
colors = 'brgk';
figure
for i = 1:4
    subplot(2,2,i)
    scatter(W(:,1),W(:,2),1,regions{i},'.')
    hold on
    scatter(testData(:,1),testData(:,2),15,testLabels,'x')
    hold off
    title(['Net',num2str(i),' Class Regions'])
end 


%% Run EM clustering for comparison
opts = statset('Display','final','MaxIter',1500,'TolFun',1e-7);
emFit4= fitgmdist(testData,4,'Options',opts);
%also try one with 'correct' start??

H = figure;
g = gca;
scatter(testData(:,1),testData(:,2),10,emFit4.cluster(testData),'filled')
gmPDF4 = @(x1,x2)reshape(pdf(emFit4,[x1(:) x2(:)]),size(x1));
hold on
fcontour(gmPDF4,[g.XLim g.YLim])
hold off
axis equal

figure
scatter(W(:,1),W(:,2),1,emFit4.cluster(W),'.')
title('EM clustering regions')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% HELPER Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [train_set,validate_set] = prep_train_crescents(model,N,testPct,sz,mode,err,shift)
%PREPARE_TRAINING_SET Creates training and validation sets from Gaussian
%Mixture Models for neural network training.
%   Detailed explanation goes here
K = model.NumComponents;
[X,T] = crescentDataRS(K,N,model,err,shift);

%% Create test and training set
c = cvpartition(T,"HoldOut",testPct);%partition for validation
%NB: it might be good to create a C-fold validation set in the future
%% Make Validation Data
testIdx = test(c);
Xtest = X(testIdx,:);
Ttest = T(testIdx);

trainIdx = training(c);
Xtrain = X(trainIdx,:);
Ttrain = T(trainIdx);

switch mode
    case 'categorical'
        validate_set.targets = categorical(Ttest);
        train_set.targets = categorical(Ttrain);
    case 'numerical'
        validate_set.targets = (full(ind2vec(Ttest')))';
        train_set.targets = (full(ind2vec(Ttrain')))';
    otherwise
        error("The variable 'mode' must be either"+...
            " 'categorical' or 'numerical'")
end

%make one hot encoded targets for validation set

validate_set.data(:,:,1,:) = reshape(Xtest',sz(1),sz(2),[]); %do FIX THIS!!!!

%% Make Training Data

%make one hot encoded targets for training set

train_set.data(:,:,1,:) = reshape(Xtrain',sz(1),sz(2),[]);
end

function [data,labels] = crescentDataRS(K,N,gm,err,shift)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
[X,T] = random(gm,N);
for ii = K:-1:1
    clusters{ii} = X(T==ii);
end
sigma = gm.Sigma;
mu = gm.mu;
%% Transform the data for each cluster onto 4 semicircles
%centers on cluster means
radius = 3*sigma;

% Uses stereographic projection of the circle onto a line to create
% crescents
for ii = K:-1:1
   noise = randn(size(clusters{ii}))*err-shift;
   swap = (-1)^(ii-1);
   r = radius;
   t = clusters{ii}-mu(ii);
   d = (r.^2+t.^2);
   x = 2*r.^2.*t./d+mu(ii);
   y = swap*2*r.^3./d -swap.*noise+swap.*r;
   crescents{ii} = [x,y];
end

data = zeros(N,2);
labels =zeros(N,1);
index = 1;
for ii = 1:K
    last = size(crescents{ii},1);
    data(index:last+index-1,:) = crescents{ii};
    labels(index:last+index-1) = ii*ones(last,1);
    index = index+last;
end


end

