function [train_set,validate_set] = ...
    unbalanced_MNIST(labels,data,subsamplesize,pct,ratios,W,H)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
numSamples = length(labels);
weights = zeros(numSamples,1);
for i=1:numSamples
    weights(i) = ratios(labels(i)+1);
end

[labelsub,subIdx] = datasample(labels,subsamplesize,'Weights',weights);
datasub = data(subIdx,:);

c = cvpartition(labelsub,"HoldOut",pct);

testIdx = test(c);
trainIdx = training(c);

labelTest = labelsub(testIdx);
dataTest = datasub(testIdx,:);

labelTrain = labelsub(trainIdx);
dataTrain = datasub(trainIdx,:);

train_set.targets = categorical(labelTrain);
validate_set.targets = categorical(labelTest);

train_set.data(:,:,1,:) = reshape(dataTrain',W,H,[]);
validate_set.data(:,:,1,:) = reshape(dataTest',W,H,[]);

end

