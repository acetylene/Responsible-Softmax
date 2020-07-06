clear; clc;
%% Create a univariate GMM with 4 clusters two majority, two minority classes
K = 4;%Number of clusters
N = 500;% number of samples
mu = linspace(-4,4,K)';
sigma = 1;
p = [.4 .15 .05 .4];

gm = gmdistribution(mu, sigma, p);
rng('default');%for repeatability
[X,T] = random(gm,N);
for ii = K:-1:1
    clusters{ii} = X(T==ii);
end
%% Transform the data for each cluster onto 4 semicircles
%centers on cluster means
radius = 3.6*sigma;
%(data greater than 3 sigma away discard? wrap? increase radius!)
for ii = K:-1:1
theta{ii} = acos((clusters{ii}-mu(ii))/radius);
end

%one up minority, one down minority
%ERR changes internal spread of clusters. 
%When ERR is smaller more points lie on semicircle
err = 3e-2; 
%SHIFT changes how close the clusters are together.
%Positive values of SHIFT move clusters closer, negative further.
shift = .7;
for ii = K:-1:1
   noise = randn(size(theta{ii}))*err-shift;
   swap = (-1)^(ii-1);
   crescents{ii} = [radius/4*cos(theta{ii})+mu(ii)/4,...
                    swap*(radius/4*sin(theta{ii})+noise)];
end

%% Make graphs to visualize clusters
colors = 'brgk';
figure
hold on
for ii = 1:K
    points = crescents{ii};
    scatter(points(:,1),points(:,2),15,colors(ii),'filled');
end
hold off
axis equal

%% combine cluster into single data matrix
Data = zeros(N,2);
labels =zeros(N);
index = 1;
for ii = 1:K
    last = size(crescents{ii},1);
    Data(index:last+index-1,:) = crescents{ii};
    labels(index:last+index-1) = ii*ones(last,1);
    index = index+last;
end

figure
scatter(Data(:,1),Data(:,2),15,'k','filled')
axis equal

%% Try EM algorithm (fitgmdist) for clustering
opts = statset('Display','final','MaxIter',1500,'TolFun',1e-7);
emFit4= fitgmdist(Data,4,'Options',opts);
emFit3= fitgmdist(Data,3,'Options',opts);
emFit4.BIC
emFit3.BIC
emFit4.BIC-emFit3.BIC

H = figure;
g = gca;
scatter(Data(:,1),Data(:,2),10,emFit3.cluster(Data),'filled')
gmPDF3 = @(x1,x2)reshape(pdf(emFit3,[x1(:) x2(:)]),size(x1));
hold on
fcontour(gmPDF3,[g.XLim g.YLim])
hold off
axis equal

gmPDF4 = @(x1,x2)reshape(pdf(emFit4,[x1(:) x2(:)]),size(x1));
H = figure;
g = gca;
scatter(Data(:,1),Data(:,2),10,emFit4.cluster(Data),'filled')
hold on
fcontour(gmPDF4,[g.XLim g.YLim])
hold off
axis equal

S.mu = [mu/4,[.2;-.2;.2;-.2]];
for ii = 1:K
S.Sigma(:,:,ii) = eye(2);
end
S.ComponentProportion = p;

emFit4_Starts = fitgmdist(Data,4,'Options',opts,'Start',S);
gmPDF4_start = @(x1,x2)reshape(pdf(emFit4_Starts,[x1(:) x2(:)]),size(x1));
H = figure;
g = gca;
scatter(Data(:,1),Data(:,2),10,emFit4_Starts.cluster(Data),'filled')
hold on
fcontour(gmPDF4_start,[g.XLim g.YLim])
hold off
axis equal
    
%% Second option: 4 2-d clusters, drop datapoints "too close" to mean?