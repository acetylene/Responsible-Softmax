function [classMat] = classLabeler(Labels)
%CLASSLABELER Creates dummy variables for softmaxt classification
%algorithms.  It takes in a set of N labels and outputs an N by K matrix of
%0s and 1s where K is the number of unique labels in the set of labels.
%   Detailed explanation goes here
classes = unique(Labels);

N = length(Labels);
K = length(classes);

classMat = zeros(N,K);

for i=1:K
    classMat(:,i)=(Labels == classes(i));
end

end

