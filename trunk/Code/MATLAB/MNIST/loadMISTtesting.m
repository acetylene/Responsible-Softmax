addpath("C:\Users\costco\Dropbox\grad\RyanC\Clustering\MATLAB",'-end');
addpath("readMNIST","hugheylab-nestedSortStruct-4b61c0f",'-end');    

prefix=pwd;
mnistImgs="t10k-images.idx3-ubyte";
mnistLbls="t10k-labels.idx1-ubyte";
imFile=strcat(prefix,"\",mnistImgs);
lblFile=strcat(prefix,"\",mnistLbls);
[TestImages,TestLabels]=processMNISTdata(imFile,lblFile);
TestImages2d=reshape(TestImages,10000,28,28);

deskewTestImg2d=zeros(size(TestImages2d));
h=waitbar(0/10000,"Deskewing Test Images");
for i=1:length(TestImages2d)
deskewTestImg2d(i,:,:)=mnistunrotate(squeeze(TestImages2d(i,:,:))./255);
waitbar(i/10000,h,"Deskewing Test Images")
end
close(h)
