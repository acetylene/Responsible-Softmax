%% Create a GMM to sample from. values for mu and sigma are arbitrary
sig(1,1,:) = [9;.04];
gm = gmdistribution([0;2.5],sig,[.8,.2]);

%% Store parameter of GMM for later use
pi = gm.ComponentProportion;
mu = gm.mu;
s = sqrt(squeeze(gm.Sigma));
numsamples = 100000;

%% Generate a sample
[X,T] = random(gm,numsamples);
pHat = [sum(T==1)/numsamples;sum(T==2)/numsamples];
%sum(pHat)

%% Plot histogram of sample compared to PDF of GMM
h_1 = figure;
histogram(X,'Normalization','pdf');
hold on;
low = min(X)-.1;
high = max(X)+.1;
numsteps = 1000;
dx = (high-low)./numsteps;
x = linspace(low,high,numsteps);
f = @(x) (pi(1).*normpdf(x,mu(1),s(1)) + pi(2).*normpdf(x,mu(2),s(2)));
g = @(x) (pHat(1).*normpdf(x,mu(1),s(1)) + pHat(2).*normpdf(x,mu(2),s(2)));
plot(x,f(x));
plot(x,g(x));
hold off;
%sum(f(x)*dx)

%% Create videomaker
%currDir = pwd;
vidtitle = 'gmm_K_2_varyingMean_N_5to250';

vidMaker = VideoMaker('VideoTitle',vidtitle,...
    'Pause',0,...
    'FrameRate',10);

%% Create video
rng('default');
seed = rng;
h_2 = figure;
make_title_frame({'Looking at Fixed Points'
    'While \mu_2 and N vary. K=2'})
vidMaker.capture_frame(50);
set(0,'CurrentFigure',h_1)
vidMaker.capture_frame(50);

close(h_2);
close(h_1);
seeds = randi(20000,1,561)+10000;
hold on
for i=1:10
    h_3 = stablepointExplorationGMM(gm,5,seeds(i),1,1,0,0);
    vidMaker.capture_frame(1);
    clf(h_3)
end
for n = 1:50
    for i=1:10
        h_3 = stablepointExplorationGMM(gm,5*n,seeds(5*n+i),1,1,0,0);
        vidMaker.capture_frame(1);
        clf(h_3)
    end
end

h_3 = stablepointExplorationGMM(gm,5000,seeds(561),1,1,0,0);
vidMaker.capture_frame(20);

vidMaker.close;