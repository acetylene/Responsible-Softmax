clear; clc;
%% Create a univariate GMM with 4 clusters two majority, two minority classes
K = 4;%Number of clusters
N = 500;% number of samples
mu = linspace(-4,4,K)';
sigma = 1;
p = [.4 .15 .05 .4];

gm = gmdistribution(mu, sigma, p);
rng('default');%for repeatability
%ERR changes internal spread of clusters. 
%When ERR is smaller more points lie on semicircle
err = 2e-1; 
%SHIFT changes how close the clusters are together.
%Positive values of SHIFT move clusters closer, negative further.
shift = -7.3;
[data,labels] = crescentData(K,N,gm,err,shift);

rng('default');
[X,T] = random(gm,N);
for ii = K:-1:1
    clusters{ii} = X(T==ii);
end
%% Transform the data for each cluster onto 4 semicircles
% % centers on cluster means
% radius = 3*sigma;
% 
% for ii = K:-1:1
%    noise = randn(size(clusters{ii}))*err-shift;
%    swap = (-1)^(ii-1);
%    r = radius;
%    t = clusters{ii}-mu(ii);
%    d = (r.^2+t.^2);
%    x = 2*r.^2.*t./d+mu(ii);
%    y = swap*2*r.^3./d -swap.*noise+swap.*r;
%    crescents{ii} = [x,y];
% end

%% Make graphs to visualize clusters
colors = 'brgk';
figure
hold on
for ii = 1:K
    points = crescents{ii};
    scatter(points(:,1),points(:,2),15,colors(ii),'filled');
end
scatter(data(:,1),data(:,2),15,labels','x');
hold off
axis equal