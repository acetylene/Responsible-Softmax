function [samples, indices, means] = mnistDataProc(imgsPath,lablPath,sampleSize,offset,sort)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

%import data into struct
[trainImgs,trainLbls]=readMNIST(imgsPath,lablPath,sampleSize,offset);

for i=sampleSize:-1:1
    mnistImgs(i).image=trainImgs(:,:,i);
    mnistImgs(i).label=trainLbls(i);
end

%sort data
mnistImgsSort=nestedSortStruct(mnistImgs,'label');

means = zeros(10,28,28);
indices = zeros(10,1);

%calculate means and indices
label = 0;
idx = 1;
for i=1:10
    while label == i-1
        means(i,:,:) = squeeze(means(i,:,:)) + mnistImgsSort(idx).image;
        idx = idx+1;
        if idx<=sampleSize
            label = mnistImgsSort(idx).label;
        else
            label=0;
        end
    end
    indices(i)=idx-1;
end

%use means to finally calculate means
means(1,:,:) = means(1,:,:)./indices(1);
for i=2:10
means(i,:,:) =means(i,:,:)./(indices(i)-indices(i-1));
end

if sort
    samples = mnistImgsSort;
else
    samples = mnistImgs;
end

end

