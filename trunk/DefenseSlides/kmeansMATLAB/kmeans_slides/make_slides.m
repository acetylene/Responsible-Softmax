clear;
clc;
close all;

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

% for i=K:-1:1
% f{i} = @(x) mvnpdf(x,mu(i,:),squeeze(sigma(:,:,i)));
% end
% g=@(x) [f{1}(x),f{2}(x),f{3}(x),f{4}(x),f{5}(x)];
% d=@(p,x) ((p.*g(x))./(g(x)*p'));

%% Setup colormap
%These colormaps are chosen to give Decent bold contrast, but may be
%adjusted some.
h=gcf;
a = colormap('lines');
close(h)
c3 = a([1 7 4],:);%needs 3 colors
c5 = a([1 5 7 3 4],:);%needs 5 colors
%% Run k means with 3 and 5 clusters
%Kmeans with 3 clusters
numIter3 = 7;
C_old = [zeros(K-2,1),(-1:1:1)'];
plt = plot_clusters(X,C_old,'k');
ax = plt.Children;
adjust_Axis(ax,c3,xlimits,ylimits);

for ii = 1:numIter3
    %two new plots produced or for loop cycle.
    [idx,C_new] =kmeans(X,K-2,...
        'Display','iter',...
        'EmptyAction','drop',...
        'MaxIter',1,...
        'Start',C_old);
    %plot with new assignments, old cluster centers
    plt = plot_clusters(X,C_old,idx);
    ax = plt.Children;
    adjust_Axis(ax,c3,xlimits,ylimits);
    
    %plot with new assignments, new centers
    C_old = C_new;
    plt = plot_clusters(X,C_old,idx);
    ax = plt.Children;
    adjust_Axis(ax,c3,xlimits,ylimits);
    
end
%plot voronoi diagram for final image
hold on
voronoi(C_new(:,1),C_new(:,2))
hold off

%Kmeans with 5 clusters
numIter5 = 8;
C_old = [zeros(K,1),(-2:1:2)'];
plt = plot_clusters(X,C_old,'k');
ax = plt.Children;
adjust_Axis(ax,c5,xlimits,ylimits);

for ii = 1:numIter5
    %two new plots produced or for loop cycle.
    [idx,C_new] =kmeans(X,K,...
        'Display','iter',...
        'EmptyAction','drop',...
        'MaxIter',1,...
        'Start',C_old);
    %plot with new assignments, old cluster centers
    plt = plot_clusters(X,C_old,idx);
    ax = plt.Children;
    adjust_Axis(ax,c5,xlimits,ylimits);
    
    %plot with new assignments, new centers
    C_old = C_new;
    plt = plot_clusters(X,C_old,idx);
    ax = plt.Children;
    adjust_Axis(ax,c5,xlimits,ylimits);
    
end

%plot voronoi diagram for final image
hold on
voronoi(C_new(:,1),C_new(:,2))
hold off
%% Grab all the figures made,
% This grabs them in REVERSE order from when
% they were made. i.e. h(1) is the newest figure.
h = findall(groot,'Type','figure');

export2tikz = true;
if export2tikz == true
    %create filenames
    fnames{length(h)} = '';
    for ii= 1:length(h)
        if ii> 1+2*numIter5
            %filenames for 3 cluster graphs
            fnames{ii} = sprintf('Kmeans3_%d.tex',1+2*numIter3-(ii-1-2*numIter5)+1);
        else
            %filenames for 5 cluster graphs
            fnames{ii} = sprintf('Kmeans5_%d.tex',1+2*numIter5-ii+1);
        end
    end
    
    %Use matlab2tikz to write out the figures to tex files.
    for ii= 1:length(h)
        if ii> 1+2*numIter5
            %create tex files for 3 cluster graphs
            set(groot,'CurrentFigure',h(ii));
            colormap(c3);
            cleanfigure('handle',h(ii));
            matlab2tikz('filename',fnames{ii},...
                'figurehandle',h(ii),...
                'colormap',c3,...
                'showInfo',false,...
                'showWarnings',false,...
                'extraTikzpictureOptions','scale=.9',...
                'floatFormat','%.5g');
        else
            %create tex files for 5 cluster graphs
            set(groot,'CurrentFigure',h(ii));
            colormap(c5);
            cleanfigure('handle',h(ii));
            matlab2tikz('filename',fnames{ii},...
                'figurehandle',h(ii),...
                'colormap',c5,...
                'showInfo',false,...
                'showWarnings',false,...
                'extraTikzpictureOptions','scale=.9',...
                'floatFormat','%.5g');
        end
    end
end
%return settings to defaults.
colormap('default');
rng('default');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Helper functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ax] = adjust_Axis(ax,colors,xlimits,ylimits)
%Adjusts axis limits and colormap
axis(ax,'image')
ax.XLim=xlimits;
ax.YLim=ylimits;
ax.Colormap = colors;
ax.TickLabelInterpreter = 'latex';
end

function [plt] = plot_clusters(X,C,idx)
%Form a scatterplot of X and C on same axis. X is colored wrt idx.
plt = figure();
scatter(X(:,1),X(:,2),10,idx,'filled')
hold on
scatter(C(:,1),C(:,2),150,'xr','LineWidth',3)
hold off
end

