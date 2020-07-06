clear;
clc;

rng(1085456391); %for testing
%rng('shuffle');
% scurr = rng;

K = 5;%randi(16)+4;%number of classes
N = 500;%Total sample size
D = 2;
TestSz = .3; %size of verification sample

%% GMM for sampling
mu = [linspace(-K,K,K)*3.5;randi(4,1,K)-2]'; %means
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

idx = [1,3,5];%unique(randi(K,1,K-1));
pStar = rand(1,K);
pStar(idx) = pStar(idx)+10;

pStar = pStar./sum(pStar);%mixture coefficients
gm = gmdistribution(mu, sigma, pStar);%gmm

[X,T] = random(gm,N);%total sample

xmin = floor(min(X(:,1)))-1;
xmax = ceil(max(X(:,1)))+1;
ymin = floor(min(X(:,2)))-1;
ymax = ceil(max(X(:,2)))+1;

xlimits = [xmin xmax];
ylimits = [ymin ymax];

video = false;
emAlg = true;
classregions = false;

for i=K:-1:1
    f{i} = @(x) mvnpdf(x,mu(i,:),squeeze(sigma(:,:,i)));
end
g3=@(x) [f{1}(x),f{2}(x),f{3}(x),f{4}(x),f{5}(x)];
d=@(p,x) ((p.*g3(x))./(g3(x)*p'));

h=gcf;
a = colormap('lines');
c3 = a([1 7 4],:);%needs 3 colors
c5 = a([1 5 7 3 4],:);%needs 5 colors
close(h)

if video
    %% Create videomaker objects to capture K-means iterations
    vidmaker3 = VideoMaker('VideoTitle','KmeansGMM3',...
        'FrameRate',1);
    vidmaker5 = VideoMaker('VideoTitle','KmeansGMM5',...
        'FrameRate',1);
    %% Run k means with 3 and 5 clusters
    %Kmeans with 3 clusters
    C = [zeros(K-2,1),(-.1:.1:.1)'];
    figure
    scatter(X(:,1),X(:,2),10,'k','filled')
    hold on
    scatter(C(:,1),C(:,2),100,'xr','LineWidth',2)
    hold off
    axis image
    ax = gca;
    ax.XLim=xlimits;
    ax.YLim=ylimits;
    ax.Colormap = colormap('lines');
    ax.ColorScale = 'log';
    % vidmaker3.capture_frame(8);
    
    [idx,C] =kmeans(X,K-2,...
        'Display','iter',...
        'EmptyAction','drop',...
        'MaxIter',1,...
        'Start',C);
    
    figure
    scatter(X(:,1),X(:,2),10,idx,'filled')
    hold on
    scatter(C(:,1),C(:,2),100,'xr','LineWidth',1)
    hold off
    axis image
    ax = gca;
    ax.XLim=xlimits;
    ax.YLim=ylimits;
    ax.Colormap = colormap('lines');
    ax.ColorScale = 'log';
    vidmaker3.capture_frame(4);
    
    for ii = 1:8
        [idx,C] =kmeans(X,K-2,...
            'Display','iter',...
            'EmptyAction','drop',...
            'MaxIter',1,...
            'Start',C);
        
        scatter(X(:,1),X(:,2),10,idx,'filled')
        hold on
        scatter(C(:,1),C(:,2),100,'xr','LineWidth',2)
        hold off
        axis image
        ax = gca;
        ax.XLim=xlimits;
        ax.YLim=ylimits;
        ax.Colormap = colormap('lines');
        ax.ColorScale = 'log';
        vidmaker3.capture_frame(4);
        
        % waitforbuttonpress;
    end
    close(vidmaker3);
    idx3 =idx;
    
    %Kmeans with 5 clusters
    C = [zeros(K,1),(-2:1:2)'];
    figure
    scatter(X(:,1),X(:,2),10,T,'filled')
    axis image
    ax = gca;
    ax.XLim=xlimits;
    ax.YLim=ylimits;
    
    figure
    scatter(X(:,1),X(:,2),10,'k','filled')
    hold on
    scatter(C(:,1),C(:,2),100,'xr','LineWidth',2)
    hold off
    axis image
    ax = gca;
    ax.XLim=xlimits;
    ax.YLim=ylimits;
    ax.Colormap = colormap('lines');
    ax.ColorScale = 'log';
    
    vidmaker5.capture_frame(8);
    
    [idx,C] =kmeans(X,K,...
        'Display','iter',...
        'EmptyAction','drop',...
        'MaxIter',1,...
        'Start',C);
    
    figure
    scatter(X(:,1),X(:,2),10,idx,'filled')
    hold on
    scatter(C(:,1),C(:,2),100,'xr','LineWidth',2)
    hold off
    axis image
    ax = gca;
    ax.XLim=xlimits;
    ax.YLim=ylimits;
    ax.Colormap = colormap('lines');
    ax.ColorScale = 'log';
    vidmaker5.capture_frame(4);
    
    for ii = 1:15
        [idx,C] =kmeans(X,K,...
            'Display','iter',...
            'EmptyAction','drop',...
            'MaxIter',1,...
            'Start',C);
        
        scatter(X(:,1),X(:,2),10,idx,'filled')
        hold on
        scatter(C(:,1),C(:,2),100,'xr','LineWidth',2)
        hold off
        axis image
        % waitforbuttonpress;
        ax = gca;
        ax.XLim=xlimits;
        ax.YLim=ylimits;
        ax.Colormap = colormap('lines');
        ax.ColorScale = 'log';
        vidmaker5.capture_frame(4);
        
    end
    close(vidmaker5);
end
%% fitGMdist with 3 and 5 clusters model select with BIC?
% not that this may be more of a criticism of BIC than of EM itself.
if emAlg
    [Y,T2] = random(gm,N);
    opts = statset('Display','final','MaxIter',1500,'TolFun',1e-7);
    emFit5= fitgmdist(X,5,'Options',opts,'Start',T);
    emFit3= fitgmdist(X,3,'Options',opts,'Start','plus');
    emFit5.BIC
    emFit3.BIC
    emFit5.BIC-emFit3.BIC
    
    h3= figure;
    g3= gca;
    scatter(Y(:,1),Y(:,2),10,cluster(emFit3,Y),'filled')
    gmPDF3 = @(x1,x2)reshape(pdf(emFit3,[x1(:) x2(:)]),size(x1));
    hold on
    s3 = fcontour(gmPDF3,[g3.XLim g3.YLim],':b');
    hold off
    axis image
    g3.TickLabelInterpreter = 'latex';
    g3.Colormap = c3;
    g3.XLim=xlimits;
    g3.YLim=ylimits;
    
    h5= figure;
    g5= gca;
    scatter(Y(:,1),Y(:,2),10,cluster(emFit5,Y),'filled')
    gmPDF5 = @(x1,x2)reshape(pdf(emFit5,[x1(:) x2(:)]),size(x1));
    hold on
    s5 = fcontour(gmPDF5,[g5.XLim g5.YLim],':b');
    hold off
    axis image
    g5.TickLabelInterpreter = 'latex';
    g5.Colormap = c5;
    g5.XLim=xlimits;
    g5.YLim=ylimits;
end

if classregions
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
    
    [~,trueclass] = max(d(pStar,W),[],2);
    
    figure
    scatter(W(:,1),W(:,2),1,trueclass,'.')
    a=gca;
    axis image
    
    
    h = findall(groot,'Type','figure');
    
    for ii = 2:7
        h(ii).Children.XLim=xlimits;
        h(ii).Children.YLim=ylimits;
        h(ii).Children.Colormap = colormap('lines');
        h(ii).Children.ColorScale = 'log';
    end
    
    img = figure;
    imagesc(reshape(trueclass,200,1000))
    ax1 = img.Children;
    ax2 = h(1).Children;
    ax1.Colormap = colormap('lines');
    ax1.ColorScale = 'log';
    ax1.YDir = 'normal';
    ax1.XTick = linRescale(ax2.XLim,ax1.XLim,ax2.XTick);
    ax1.YTick = linRescale(ax2.YLim,ax1.YLim,ax2.YTick);
    ax1.XTickLabel = ax2.XTickLabel;
    ax1.YTickLabel = ax2.YTickLabel;
    axis image
end

function [newLoc] = linRescale(source,target,locations)
l = @(x) (x-source(1)).*(target(2)-target(1))./(source(2)-source(1))+target(1);
newLoc = linspace(l(locations(1)), l(locations(end)),length(locations));
end