%% Load the MNIST files
addpath("C:\Users\costco\Dropbox\grad\RyanC\Clustering\MATLAB",'-end');
if exist('logitFitsMATLAB.mat','file')
  clearvars;
  load('logitFitsMATLAB.mat')
  loaded=true;
else
    clearvars;
    save('logitFitsMATLAB.mat')
    % Loads and deskews 60000 training MNIST images
    loadMISTtraining;
    
    % Loads and deskews 10000 testing MNIST images
    loadMISTtesting;

    save('logitFitsMATLAB.mat','Images','Labels','images2d', 'deskewImg2d', '-append')
    save('logitFitsMATLAB.mat','TestImages','TestLabels','TestImages2d', 'deskewTestImg2d', '-append')
    loaded = false;
end

%% Run Logistic regression for several (all) pairs of digits
C=nchoosek(0:9,2);
len = size(C,1);

trainDeskew = reshape(deskewImg2d,60000,784);
testDeskew = reshape(deskewTestImg2d,10000,784);
    
if ~loaded       
    rng('default');
    rng(25583671);
    
    h=waitbar(0/len,"Calculating weights");
    
    for i=len:-1:1
        waitbar((len-i)/len,h)
        %tic
        A = C(i,1);
        B = C(i,2);
        [ fullSampleClassifier(i).percent, ...
            fullSampleClassifier(i).assignments, ...
            fullSampleClassifier(i).maxPct, ...
            fullSampleClassifier(i).maxAssign, ...
            fullSampleClassifier(i).phat, ...
            fullSampleClassifier(i).pstar, ...
            fullSampleClassifier(i).weights, ...
            fullSampleClassifier(i).badIndices] = ...
            mnistLogitClassifier(trainDeskew, ...
            Labels, testDeskew,...
            TestLabels, A, B, 2500);
        %toc
    end
    save('logitFitsMATLAB.mat','fullSampleClassifier', '-append')
end

%% Compare results above to those given by 'usual' logit binary classification
%percents = zeros(1,length(C));
for i=45:-1:1
percents(i)=fullSampleClassifier(i).percent;
end

for i=45:-1:1
maxPercents(i)=fullSampleClassifier(i).maxPct;
end

diffIdxs=find(abs(percents-maxPercents)>10^-6);
D=C(diffIdxs,:);


%% Use resampling to compare the same ideas

% Look at norm(piHat-piStar) for several samples

%% Choose pairs of digits to sample based on the fullsample classifier

%% Train a pattern net or perceptron on each pair of digits.
%% Create 4 'unbalanced' test data sets for each pair of digits
%% For each test set, perform both max classification and recursive gradient classification
% return pct error, pct different classifications, and K-L div.  Bootstrap 1000(?) times
%% Write data to CSV?
%% Find ways to decide which is correct most often! 