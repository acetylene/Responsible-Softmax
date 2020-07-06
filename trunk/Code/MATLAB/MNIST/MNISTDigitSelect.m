function [newSample, newLabels, indices] = MNISTDigitSelect(sample,labels,digits)
%MNISTDIGITSELECT selects a subset of an MNIST sample based on the
% DIGITS to be selected
%   SAMPLE - The larger sample from which a subset is to be given. A N by D
%   times D matrix,  where D is the number of pixels per side of the image.
%   LABELS - Labels for the images. A 1 by N vector. Used for selection.
%   DIGITS - A 1 by M vector of the digits to be selected.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Should be good now 10/24/18
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
assert(size(digits,1)==1,'The argument digits must be a 1 by M vector')
M=size(digits,2);
D=size(sample,2);

samplesize=zeros(1,M);

%% Isolate the desired digits in a struct
for i=M:-1:1
    dataStruct(i).indices = find(labels==digits(i));
    dataStruct(i).sample = sample(dataStruct(i).indices,:);
    dataStruct(i).labels = labels(dataStruct(i).indices);
    samplesize(i) = length(dataStruct(i).indices);
end

newSample=zeros(sum(samplesize),D);
newLabels=zeros(sum(samplesize),1);
indices=newLabels;

prev = 1;

%% Extract data from the struct as matrices and vectors
for i=1:M
    newSample(prev:(prev+samplesize(i)-1),:)=dataStruct(i).sample;
    newLabels(prev:(prev+samplesize(i)-1))=dataStruct(i).labels;
    indices(prev:(prev+samplesize(i)-1))=dataStruct(i).indices;
    prev = prev+samplesize(i);
end

end

