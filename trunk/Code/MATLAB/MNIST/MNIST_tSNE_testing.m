%% Load the MNIST images
%TODO check for file tsneclassification.mat, if it's there, load it, else do
%below
if exist('tsneclassification.mat','file')
    load('tsneclassification.mat')
else
    clearvars;
    
    addpath("C:\Users\costco\Dropbox\grad\RyanC\Clustering\MATLAB",'-end');
    addpath("readMNIST","hugheylab-nestedSortStruct-4b61c0f",'-end');
    
    prefix=pwd;
    mnistImgs="train-images.idx3-ubyte";
    mnistLbls="train-labels.idx1-ubyte";
    imFile=strcat(prefix,"\",mnistImgs);
    lblFile=strcat(prefix,"\",mnistLbls);
    
    offset=randi(5000);
    numberSamples=50000;
    
    [mnistTrainImgs,changeIdx,trainMeans]=mnistDataProc(imFile, lblFile, numberSamples, offset, true);
    
    test=true;
    
    if test
        testImgs="t10k-images.idx3-ubyte";
        testLbls="t10k-labels.idx1-ubyte";
        testImFile=strcat(prefix,"\",testImgs);
        testLblFile=strcat(prefix,"\",testLbls);
        [mnistTestImgs,~,~]=mnistDataProc(testImFile,testLblFile,9000,0,false);
    end
    
    
    %% Do tSNE identification of digits
    len=length(mnistTrainImgs);
    imSize=size(mnistTrainImgs(1).image);
    imLen=numel(mnistTrainImgs(1).image);
    images=zeros(len,imLen);
    labels=zeros(1,len);
    for i=1:len
        images(i,:) = reshape(mnistTrainImgs(i).image,1,imLen);
        labels(i)=mnistTrainImgs(i).label;
    end
    
    %The following takes a long time. Maybe have it do a progress bar, fewer pca modes?
    [mnistTsne3,loss3d] = tsne(images,'Distance','euclidean',...
        'NumDimensions',3,...
        'NumPCAComponents',75,...
        'Standardize',true);
    
%end

[zeroTsne,zeroMu,zeroSig] = mnistTsneProcessing(mnistTsne3,labels,0);
[oneTsne,oneMu,oneSig] = mnistTsneProcessing(mnistTsne3,labels,1);
[twoTsne,twoMu,twoSig] = mnistTsneProcessing(mnistTsne3,labels,2);
[sixTsne,sixMu,sixSig] = mnistTsneProcessing(mnistTsne3,labels,6);

testLabels = zeros(1,9000);
for i=1:9000
    testLabels(i)=mnistTestImgs(i).label;
end

indices=find(testLabels == 0);
for i=[1,2,6]
    indices=[indices,find(testLabels == i)];
end

testSamples=zeros(numel(indices),28^2);

for i=1:numel(indices)
    testSamples(i,:)=reshape(mnistTestImgs(indices(i)).image,1,28^2);
end

mnistTsneTest = tsne(testSamples,'Distance','euclidean',...
        'NumDimensions',3,...
        'NumPCAComponents',75,...
        'Standardize',true);
end

means=[zeroMu;oneMu;twoMu;sixMu];
Sigs=[zeroSig;oneSig;twoSig;sixSig];
Sigs=reshape(Sigs',3,3,4);

F=zeros(4,length(indices));
for i=1:4
    F(i,:)=mvnpdf(mnistTsneTest,means(i,:),Sigs(:,:,i))';
end

p=ones(4,1)./4;

[phat,orbit]=stablepoint(F,p,13,'diff');

[digitAssignments,costs]=classAssignment(F,phat);
digitAssignments(digitAssignments==1)=0;
digitAssignments(digitAssignments==2)=1;
digitAssignments(digitAssignments==3)=2;
digitAssignments(digitAssignments==4)=6;

realLabels=ones(1,length(indices))./length(indices);
for i=1:length(indices)
    realLabels(i)=mnistTestImgs(indices(i)).label;
end

fracCorr = zeros(1,4);
for i=1:4
    fracCorr(i)=sum(digitAssignments(i,:)==realLabels)./length(indices);
end

fprintf('Fraction of correct assignments is: %6.4f \n',fracCorr)

numZero=sum(realLabels == 0);
numOne=sum(realLabels == 1);
numTwo=sum(realLabels == 2);
numSix=sum(realLabels == 6);

pstar=[numZero;numOne;numTwo;numSix]./length(indices);

disp('phat is: ')
disp(phat)
disp('pstar is: ')
disp(pstar)
%% Calculate covariances of individual images
% The means from the training set are removed... might give better results?
% using Delta removes all entries less than delta form the covariance of an individual picture

% THIS WAS TAKING TOO LONG. ALSO, IMAGE COV. MATS WEREN'T SPARSE.  MAYBE JUST DO
% VARIANCE DIAG MATRIX?

% delta =10^-4;

% % Zeros images
% for i=changeIdx(1)-1:-1:1
% zeroCovs(i).label=mnistTrainImgs(i).label;
% zeroCovs(i).covar=singleImageCov(mnistTrainImgs(i).image,delta);
% end
% % Ones images
% for i=changeIdx(2)-1:-1:changeIdx(1)
% oneCovs(i-changeIdx(1)+1).label=mnistTrainImgs(i).label;
% oneCovs(i-changeIdx(1)+1).covar=singleImageCov(mnistTrainImgs(i).image,delta);
% end
% % Twos images
% for i=changeIdx(3)-1:-1:changeIdx(2)
% twoCovs(i-changeIdx(2)+1).label=mnistTrainImgs(i).label;
% twoCovs(i-changeIdx(2)+1).covar=singleImageCov(mnistTrainImgs(i).image,delta);
% end
%
% % Rearrange the samples to calculate a covariance
% tic
%  [zeroCovSamp,zeroCovMean]=covStruct2sample(zeroCovs);
% toc
% tic
% [oneCovSamp,oneCovMean]=covStruct2sample(oneCovs);
% toc
% tic
% [twoCovSamp,twoCovMean]=covStruct2sample(twoCovs);
% toc
%
% tic
% zeroSamp = zeroCovSamp-zeroCovMean;
% oneSamp = oneCovSamp-oneCovMean;
% twoSamp = twoCovSamp-twoCovMean;
% toc

%% Function definitions
function [sample,sampmean]=covStruct2sample(covstruct)
covsize = numel(covstruct(1).covar);
numsamples = numel(covstruct);
indices=zeros(3,covsize*numsamples);
currPos=1;
for k=1:numsamples%extract indices for non zero elements of covars reshaped as row variables
    currCov=covstruct(k).covar;
    currLin=reshape(currCov,1,numel(currCov));%ideally numel(currCov)==covsize
    [i,j,v]=find(currLin);
    oldPos=currPos;
    currPos=oldPos+numel(i)-1;
    %  fprintf('currPos is: %d \n oldPos is: %d \n length of i is: %d\n',currPos,oldPos,numel(i))
    indices(:,oldPos:currPos)=[k*i;j;v];
end
I=find(indices(1,:),1,'last');
J=find(indices(2,:),1,'last');
V=find(indices(3,:),1,'last');

lastpos = max([I,J,V]);
indices=indices(:,1:lastpos);

sample=sparse(indices(1,:),indices(2,:),indices(3,:),numsamples,covsize);
sampmean=mean(sample);
end

function [sample,mu,sigma]=mnistTsneProcessing(tsneData,labels,value)
idx=find(labels == value);
sample = tsneData(idx,:);
mu=mean(sample);
sigma = robustcov(sample-mu);
end

%M=[200,.32,.11,.44,.43;1000,2.33,1.06,2.18,.85;10000,45.22,18.93,23.98,19.77];
