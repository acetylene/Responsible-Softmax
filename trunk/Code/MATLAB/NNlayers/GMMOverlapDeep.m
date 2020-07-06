clear;
clc;

rng(1085456391); %for testing
%rng('shuffle');
% scurr = rng;

K = 5;%randi(16)+4;%number of classes
N = 10000;%Total sample size
D = 2;
TestSz = .3; %size of verification sample

%% GMM for sampling
mu = [linspace(-K,K,K)*4;randi(4,1,K)-2]'; %means
density = 1;
rc = .9;
for k=K:-1:1
    x = rand();
    if mod(k,2) == 0 
        A = [0,-1;1,0];
        evals = [5*x,x];
    else
%         tmp = sprand(D,D,density,rc);
        A = -eye(D);%[1,-1;1,1]./sqrt(2);%full(tmp);
        evals = [19*x,x];
    end
    
    Sig = diag(evals);
    
    sigma(:,:,k) = A*Sig*A' ; %variances
end

%adjust for specific example
mu(2,1) = -9;
mu(4,1) = 9;
sigma(:,:,4) = sigma(:,:,2);
sigma(:,:,5) = sigma(:,:,1);

idx = [1,5];%unique(randi(K,1,K-1));
pStar = rand(1,K);
pStar(idx) = pStar(idx)+10;

pStar = pStar./sum(pStar);%mixture coefficients
gm = gmdistribution(mu, sigma, pStar);%gmm

[X,T] = random(gm,N);%total sample

figure
scatter(X(:,1),X(:,2),10,T)

%% Layers to train

inputLayer = imageInputLayer([D 1 1], 'Normalization','none');
baseLayers = [inputLayer
    fullyConnectedLayer(K*512)
    tanhLayer
    fullyConnectedLayer(K*32)
%     %dropoutLayer(.25)
%     tanhLayer
%     fullyConnectedLayer(K*16)
    tanhLayer
    fullyConnectedLayer(K)
    softmaxLayer];

layers{1} = [baseLayers
    classificationLayer];

layers{2} = [baseLayers
    responsibilityLoss(K,4,'ratios',pStar')];

layers{3} = [baseLayers
    responsibilityLoss(K,4,'iterations',4, 'ratios',pStar')];

layers{4} = [baseLayers
    responsibilityLoss(K,4,'iterations',8, 'ratios',pStar')];

layers{5} = [baseLayers
    responsibilityLoss(K,4,'iterations',16, 'ratios',pStar')];

layers{6} = [baseLayers
    fixedRespLoss(K,4,'ratios',pStar')];

%% Filename info
% ensure working directory is parent of testresults before running!!
% alternatively, use what()
rng('shuffle');
scurr = rng;

file = what('data2');
dir = strcat(file.path,'\');%strcat(pwd,'\testResults\data2\');

accFile = strcat(dir,...
    'acc_GMM_ovrlp_',num2str(K),'_',num2str(scurr.Seed),'.xlsx');
confusionFile = strcat(dir,...
    'confusion_GMM_ovrlp_',num2str(K),'_',num2str(scurr.Seed),'.xlsx');
pctFile = strcat(dir,...
    'tpr_GMM_ovrlp_',num2str(K),'_',num2str(scurr.Seed),'.xlsx');

%% Create Test and Validation Data

sz = [2,1];
numRuns = 5;
runSz = 32;
seeds = randi([1000000,90000000],1,numRuns*runSz);
W = waitbar(0,sprintf('Calculating Run %d of %d',1,numRuns));
for i=1:numRuns
    % Prepare Training Data
    [train_set,validate_set] =...
        prepare_training_set(gm,N,TestSz,sz,'categorical');
    %options setup
    options = trainingOptions('adam', ...
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropFactor',0.2, ...
        'LearnRateDropPeriod',5, ...
        'MaxEpochs',15, ...
        'MiniBatchSize',100, ...
        'Verbose',false, ...
        'ValidationData',{validate_set.data,validate_set.targets});
    %Create test set
    rng('default')
    [Y,C]=random(gm,N/4);
    Ytest(:,:,1,:) = reshape(Y',D,1,[]);
        
    waitbar(i/numRuns,W,sprintf('Calculating Sample %d of %d',i,numRuns))
    H = waitbar(0,sprintf('Training set %d of %d',0,runSz));
    tic
    for j = 1:runSz
        waitbar(j/runSz,H,sprintf('Training set %d of %d',j,runSz));
        %Train Nets
        [nets] = train_nets(layers,...
            train_set.data,...
            train_set.targets,...
            options,...
            seeds(j+runSz*(i-1)));
        toc
        %Test Nets
        [acc,confMat,pcts] = test_nets(nets, Ytest, C);
        %Write results to csv file
        %future iteration: look @ weights and biases, on average. Is there
        %a trend? Am I really changing piHat?
        cellIdx = 1+runSz*K*(i-1)+K*(j-1);
        cell = strcat('A',num2str(cellIdx));
        accCell = strcat('A',num2str(j+runSz*(i-1)));
        writematrix(acc,accFile,'Range',accCell)
        writematrix(confMat,confusionFile,'Range',cell)
        writematrix(pcts,pctFile,'Range',cell)
    end
    close(H)
end
close(W)

for i=K:-1:1
f{i} = @(x) mvnpdf(x,mu(i,:),squeeze(sigma(:,:,i)));
end
g=@(x) [f{1}(x),f{2}(x),f{3}(x),f{4}(x),f{5}(x)];
d=@(p,x) ((p.*g(x))./(g(x)*p'));

[X,T] = random(gm,N);%total sample

xmin = floor(min(X(:,1)))-1;
xmax = ceil(max(X(:,1)))+1;
ymin = floor(min(X(:,2)))-1;
ymax = ceil(max(X(:,2)))+1;
xSamples = 1000;
ySamples = 200;
x = linspace(xmin,xmax,xSamples);
y = linspace(ymin,ymax,ySamples);
[U,V]=meshgrid(x,y);

W = zeros(xSamples*ySamples,2);
for i = 1:ySamples
    for j = 1:xSamples
        W(i+ySamples*(j-1),:)=[U(i,j);V(i,j)];
    end
end

Wtest(:,:,1,:) = reshape(W',D,1,[]);

numNets = size(nets,2);
for i = numNets:-1:1
    [~,regions{i}]= max(nets{i}.predict(Wtest),[],2);
end

numRows = ceil(numNets/2);
figure
for i = 1:numNets
    subplot(numRows,2,i)
    scatter(W(:,1),W(:,2),1,regions{i},'.')
    title(['Net',num2str(i),' Class Regions'])
end 

[~,trueclass] = max(d(pStar,W),[],2);

figure
scatter(W(:,1),W(:,2),1,trueclass,'.')
figure
scatter(X(:,1),X(:,2),10,T,'filled')