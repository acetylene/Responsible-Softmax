%% Load the MNIST images
clearvars;

addpath("readMNIST","hugheylab-nestedSortStruct-4b61c0f",'-end');

prefix=pwd;
mnistImgs="train-images.idx3-ubyte";
mnistLbls="train-labels.idx1-ubyte";
imFile=strcat(prefix,"\",mnistImgs);
lblFile=strcat(prefix,"\",mnistLbls);

offset=0;%randi(7500);
numberSamples=50000;

[samples,changeIdx,trainMeans]=mnistDataProc(imFile, lblFile, numberSamples, offset, true);

for i=numberSamples:-1:1
unrotSamples(i).label=samples(i).label;
unrotSamples(i).image=mnistunrotate(samples(i).image);
end

numerTest = 25;
indices=randi(numberSamples,1,numerTest);

for i=1:numerTest
imshow(unrotSamples(indices(i)).image,'InitialMagnification','fit')
waitforbuttonpress;
disp([samples(indices(i)).label,indices(i)])
end