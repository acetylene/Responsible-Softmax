function [train_set,validate_set] = prepare_training_set(model,N,testPct,sz,mode)
%PREPARE_TRAINING_SET Creates training and validation sets from Gaussian
%Mixture Models for neural network training.
%   Detailed explanation goes here


[X,T] = random(model,N);

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

