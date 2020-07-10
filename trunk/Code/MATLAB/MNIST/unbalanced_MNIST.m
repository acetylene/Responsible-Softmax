function [train_set,validate_set] = ...
    unbalanced_MNIST(labels,data,subsamplesize,pct,ratios,W,H)
%UNBALANCED_MNIST Creates training and validation sets from the MNIST data
%  set. The sets produced have samples reweighted according to the convex
%  mixture given in RATIOS.
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

