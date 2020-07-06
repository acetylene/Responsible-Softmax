%clearvars;
    
addpath("C:\Users\costco\Dropbox\grad\RyanC\Clustering\MATLAB",'-end');
addpath("readMNIST","hugheylab-nestedSortStruct-4b61c0f",'-end');    

prefix=pwd;
mnistImgs="train-images.idx3-ubyte";
mnistLbls="train-labels.idx1-ubyte";
imFile=strcat(prefix,"\",mnistImgs);
lblFile=strcat(prefix,"\",mnistLbls);
[Images,Labels]=processMNISTdata(imFile,lblFile);
images2d=reshape(Images,60000,28,28);

deskewImg2d=zeros(size(images2d));
h=waitbar(0/60000,"Deskewing Training Images");
for i=1:length(images2d)
deskewImg2d(i,:,:)=mnistunrotate(squeeze(images2d(i,:,:))./255);
waitbar(i/60000,h,"Deskewing Training Images")
end

close(h)