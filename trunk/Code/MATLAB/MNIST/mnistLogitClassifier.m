function [pctCorrect, assignments, maxPct, maxAssign, pHat, pStar, weights, badIndices ] = mnistLogitClassifier(trainSample, trainLabels, testSample, testLabels, label1, label2, maxIter)

%% Isolate two digits to study. One training set, one test set.
fprintf('Labels are %d and %d\n',label1,label2)
% Training set
[trainSample, trainLabels, ~]=MNISTDigitSelect(trainSample, trainLabels, [label1, label2]);%~ could be trainIndices
 %= trainSample(trainIndices,:);
fprintf('Size of training sample is %d by %d\n',size(trainSample))
 %= trainLabels(trainIndices);
fprintf('Size of training labels is %d by %d\n',size(trainLabels))

% Test Set
[testSample, testLabels, testIndices]=MNISTDigitSelect(testSample, testLabels, [label1, label2]);
% = testSample(testIndices,:);
% = testLabels(testIndices);

%% Minimize objective function for weights on training set.
initTheta = zeros(size(trainSample,2)+1,1);
assert(length(initTheta)==785)

fnLabels = abs((trainLabels-label1)./(label2-label1));

options = optimoptions('fminunc','SpecifyObjectiveGradient',true,'MaxIterations',maxIter);
% TODO: implement/borrow code to run multiclass minimization to get weights
weights = fminunc(@(t)logisticCost(t,trainSample, fnLabels),initTheta,options);



%% Calculate the parameter matrix F, use that to get a pHat
W=weights(2:length(weights));
B=weights(1);
 
F=zeros(2,length(testIndices));
F(1,:)=sigmf(testSample*W+B,[1 0]); %replace later with exp(X)./sum(exp(X))?
F(2,:)=1-F(1,:);
p = ones(2,1)./2;
pHat = stablepoint(F,p,13,'diff');

P=sum(testLabels==label1)./length(testLabels);
pStar=[P;1-P];

%% Get the item assignments from F and pHat
assignments = classAssignment(F,pHat);
% Pitfall: must make sure all elements of class assignment at this point
% are not the same as any possible labels

assignments=assignments-3;% How to do this with more than 2 classes?

% Adjust the assignment labels to be the same as the initial labels.
assignments(assignments==-2)=label1;
assignments(assignments==-1)=label2;

% Potential here to compare to just rounding the sigmoidfunction to get
% assignments? (should be almost the same).
maxAssign = round(F(1,:))-2;
maxAssign(maxAssign==-2)=label1;
maxAssign(maxAssign==-1)=label2;

%% Compare the item assignments to the correct labels
numCorrect = sum(assignments(2,:)'==testLabels);
pctCorrect = numCorrect./length(testLabels);

numMax = sum(maxAssign'==testLabels);
maxPct = numMax./length(testLabels);

badIndices = testIndices(assignments(1,:)'==testLabels);
end