clear; clc;
%% Load MNIST data
%get file structure for prepared MNIST data
s = what('MNIST');
filename = fullfile(s.path,'\','MNISTdata.mat');

load(filename)

K=size(unique(Labels),1);
[N,w,h] = size(images2d);
b = randi(N-9);
figure
for i=1:9
    subplot(3,3,i)
    imagesc(squeeze(images2d(i+b,:,:)))
    title(['Label:',num2str(Labels(i+b))])
end

%Set data up for training
TestSz = .1; %size of verification sample
c = cvpartition(Labels,"HoldOut",TestSz);

%%% TODO:%%%
% Try a test and validation set with different distibution mixtures from 
% training. Intuition says that responsibility would do worse, but it might
% be okay.

X = permute(images2d,[2,3,1]);
T = Labels;
%% Make Validation Data
validIdx = test(c);
Xvalid = X(:,:,validIdx);
Tvalid = T(validIdx);

catValid = categorical(Tvalid);
%make one hot encoded targets for validation set
% vecValid = (full(ind2vec(Tvalid'+1)))';
validInputs(:,:,1,:) = Xvalid;

%% Make Training Data
trainIdx = training(c);
Xtrain = X(:,:,trainIdx);
Ttrain = T(trainIdx);

catTrain = categorical(Ttrain);
%make one hot encoded targets for training set
% vecTrain = (full(ind2vec(Ttrain'+1)))';
trainInputs(:,:,1,:) = Xtrain;

%% Neural Network Training
%layer setup
%possible different input layer
inputLayer = imageInputLayer([w h 1], 'Normalization','none');
baseLayers = [inputLayer
    convolution2dLayer(5,11)
    tanhLayer
    maxPooling2dLayer(2,'Stride',3)
    fullyConnectedLayer(K)
    softmaxLayer];

layers1 = [baseLayers
    classificationLayer];

layers2 = [baseLayers
    responsibilityLoss(K,4)];

layers3 = [baseLayers
    responsibilityLoss(K,4,'iterations',4)];

%Options for Training
optionsCat = trainingOptions('adam', ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.2, ...
    'LearnRateDropPeriod',5, ...
    'MaxEpochs',7, ...
    'MiniBatchSize',100, ...
    'Plots','training-progress',...
    'Verbose',true,...
    'ValidationData',{validInputs,catValid});

% rng(1024241);
rng(18931316);
 net1 = trainNetwork(trainInputs, catTrain, layers1, optionsCat);%need
% %categorical targets for classification layers
% rng(1024241);
rng(18931316);
 net2 = trainNetwork(trainInputs, catTrain, layers2, optionsCat);%need 
% %numerical targets for 'regression' layers
% rng(1024241);
rng(18931316);
 net3 = trainNetwork(trainInputs, catTrain, layers3, optionsCat);%need
%categorical targets for classification layers
%% Test the layers against test set
C = TestLabels;
Ytest(:,:,1,:) = permute(TestImages2d,[2,3,1]);

Chat1 = net1.predict(Ytest);
Chat2 = net2.predict(Ytest);
Chat3 = net3.predict(Ytest);
% Chat4 = net4.predict(Ytest);

[~,classHat1]= max(Chat1,[],2);
[~,classHat2]= max(Chat2,[],2);
[~,classHat3]= max(Chat3,[],2);
% [~,classHat4]= max(Chat4,[],2);

figure
confusionchart(C+1,classHat1)
figure
confusionchart(C+1,classHat2)
figure
confusionchart(C+1,classHat3)
% figure
% confusionchart(C,classHat4)
