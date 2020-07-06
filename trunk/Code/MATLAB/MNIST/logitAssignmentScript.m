addpath("C:\Users\costco\Dropbox\grad\RyanC\Clustering\MATLAB",'-end');
addpath("readMNIST","hugheylab-nestedSortStruct-4b61c0f",'-end');
load('logitFitsMATLAB.mat')
W=zeroOneFit.Beta;
C=zeroOneFit.Bias;
% loadMISTtesting
% h=waitbar(0/10000,"unrotating images");
% for i=1:length(TestImages2d)
% unrotTestImg2d(i,:,:)=mnistunrotate(squeeze(TestImages2d(i,:,:))./255);
% waitbar(i/10000,h,"unrotating images")
% end
testIndices=find((TestLabels==1)|(TestLabels==0));
testSample=unrotTestImg2d(testIndices,:,:);
testSample=reshape(testSample,2115,784);
testGuess=round(sigmf(testSample*W+C,[1 0]));
testSampleLabels=TestLabels(testIndices);
sum(testSampleLabels==testGuess)
sum(testSampleLabels==testGuess)./2115
F=zeros(2,2115);
F(1,:)=sigmf(testSample*W+C,[1 0]);
F(2,:)=1-F(1,:);
p=ones(2,1)./2;
[phat,orbit]=stablepoint(F,p,13,'diff');
pstar=[sum(testSampleLabels)./2115;0];
pstar(2)=1-pstar(1);
norm(phat-pstar)
sum(abs(phat-pstar))
[testAssign,costs]=classAssignment(F,phat);
testAssign;
testAssign=testAssign-1;

%imshow(reshape(testSample(441,:),28,28),'InitialMagnification','fit')
W2=zeroOneFitLogOdds.Beta;
C2=zeroOneFitLogOdds.Bias;
F2=zeros(2,2115);

logOddTestSample=log(testSample./(1-testSample+eps)+eps);
F2(1,:)=sigmf(logOddTestSample*W2+C2,[1 0]);
F2(2,:)=1-F2(1,:);
[phat2,orbit2]=stablepoint(F2,p,13,'diff');
norm(phat2-pstar)
sum(abs(phat2-pstar))
[testAssign2,costs2]=classAssignment(F2,phat2);
testAssign2=testAssign2-1;
sum(testAssign(2,:)'==testSampleLabels)
sum(testAssign(2,:)'==testSampleLabels)./2115
sum(testAssign2(2,:)'==testSampleLabels)
sum(testAssign2(2,:)'==testSampleLabels)./2115

find(testAssign(1,:)'==testSampleLabels)
sum(testAssign(2,:)'==testSampleLabels)
find(testAssign(1,:)'==testSampleLabels)
find(testAssign2(1,:)'==testSampleLabels)
find(testGuess==testSampleLabels)
find(testGuess~=testSampleLabels)
find(testAssign2(1,:)'==testSampleLabels)
find(testAssign(1,:)'==testSampleLabels)


badIndices = unique([find(testAssign(1,:)'==testSampleLabels); find(testAssign2(1,:)'==testSampleLabels); find(testGuess~=testSampleLabels)]);
toughImages=testIndices(badIndices);
toughImages
for i=1:9
imshow(squeeze(unrotTestImg2d(toughImages(i),:,:)),'InitialMagnification','fit')
waitforbuttonpress
end