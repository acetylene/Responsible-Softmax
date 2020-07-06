%% Load pre-processed data
if exist('logitFitsMATLAB.mat','file')
  clearvars;
  load('logitFitsMATLAB.mat')
else 
  assert(false, 'The file logitFitsMATLAB.mat could not be found.  Please make it available and try again')
end
%% Choose pairs of digits to sample based on the fullsample classifier

%% Train a pattern net or perceptron on each pair of digits.
%% Create 4 'unbalanced' test data sets for each pair of digits
%% For each test set, perform both max classification and recursive gradient classification
% return pct error, pct different classifications, and K-L div.  Bootstrap 1000(?) times
%% Write data to CSV?
%% Find ways to decide which is correct most often!