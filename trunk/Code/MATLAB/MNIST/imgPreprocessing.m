[trainImgs,trainLbls]=readMNIST("C:\Users\costco\Dropbox\grad\RyanC\Clustering\MATLAB\MNIST\train-images.idx3-ubyte", "C:\Users\costco\Dropbox\grad\RyanC\Clustering\MATLAB\MNIST\train-labels.idx1-ubyte",150,0);

zeroIdx=find(trainLbls==0);
onesIdx=find(trainLbls==1)
twosIdx=find(trainLbls==2)
threesIdx=find(trainLbls==3)
foursIdx=find(trainLbls==4);
fivesIdx=find(trainLbls==5);
sixIdx=find(trainLbls==6);
sevenIdx=find(trainLbls==7);
eightIdx=find(trainLbls==8);
ninesIdx=find(trainLbls==9);

for i=0:9
tmp=find(trainLbls == i);
len=length(tmp);
lblIdxs(i+1,1:len)=tmp';
end
lblIdxs=lblIdxs(:,1:22);
lblIdxs(1,:)
find(lblIdxs(1,:)~=0)
length(find(lblIdxs(1,:)~=0))
len=18
for i=1:len
imshow(trainImgs(:,:,lblIdxs(1,i)));
waitforbuttonpress;
end