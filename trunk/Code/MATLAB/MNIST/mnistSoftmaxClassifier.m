function [pctCorrect, assignments, maxPct, maxAssign, confidence, pHat, pStar, weights, badIndices ] = ...
    mnistSoftmaxClassifier(trainSample, trainLabels, testSample, testLabels, digits, maxIter)
%MNISTSOFTMAXCLASSIFIER Uses softmax (multliclass logistic) regression to
%classify mnist images from a given test set, TESTSAMPLE. Training
%(regression) is perfomed on the set TRAINSAMPLE, with targets given by
%TRAINLABELS.  Results are evaluated based on the classification given in
%TESTLABELS.
%   Detailed explanation goes here
disp(['Digits to classify are: ', num2str(digits)])
K = length(digits);

%% Isolate digits to study. One training set, one test set.

%Training Set
[trainSample, trainLabels, ~]=MNISTDigitSelect(trainSample, trainLabels, digits);
%MNISTDigitSelect naturally sorts the sample and the label.
N = length(trainLabels);
change=randperm(N);
trainSample = trainSample(change,:);
trainLabels = trainLabels(change);

%Test Set
[testSample, testLabels, testIndices]=MNISTDigitSelect(testSample, testLabels, digits);
%MNISTDigitSelect naturally sorts the sample and the label.
M = length(testLabels);
change = randperm(M);
testSample = testSample(change,:);
testLabels = testLabels(change);


%% Minimize objective function for weights on training set.
T = classLabeler(trainLabels);
X = [ones(N,1),trainSample];

weights = gradSoftmax(X, T, maxIter);

%% Calculate the parameter matrix F, use that to get a pHat
X = [ones(M,1),testSample];
F = sftmax(weights*X');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
p = ones(K,1)./K;
pHat = stablepoint(F,p,13,'diff');

T = classLabeler(testLabels);
pStar=sum(T)./M;

%% Get the item assignments from F and pHat
assignments = classAssignment(F,pHat);
% Pitfall: must make sure all elements of class assignment at this point
% are not the same as any possible labels

assignments=assignments-(K+1);

% Adjust the assignment labels to be the same as the initial labels.
for i=-K:-1
    assignments(assignments==i)=digits(-i);
end

% Potential here to compare to just rounding the sigmoidfunction to get
% assignments? (should be almost the same).
[confidence,maxAssign] = max(F);
maxAssign = maxAssign - (K+1);

for i=-K:-1
    maxAssign(maxAssign==i)=digits(-i);
end

%% Compare the item assignments to the correct labels
numCorrect = sum(assignments(K,:)'==testLabels);
pctCorrect = numCorrect./M;

numMax = sum(maxAssign'==testLabels);
maxPct = numMax./M;

badIndices = testIndices(assignments(K,:)'~=testLabels);

end

