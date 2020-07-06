%% This gives an example of how the simplex S_K contracts 
% through iterations of R_F.
clear;
clc;
rng(1085456391); %for testing
%rng('shuffle');
% scurr = rng;

K = 3;%randi(16)+4;%number of classes
N = 50;%Total sample size
D = 2;

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

idx = [1 3];%unique(randi(K,1,K-1));
pStar = rand(1,K);
pStar(idx) = pStar(idx)+5;

pStar = pStar./sum(pStar);%mixture coefficients
gm = gmdistribution(mu, sigma, pStar);%gmm

[X,T] = random(gm,N);%total sample

figure
scatter(X(:,1),X(:,2),10,T)

%% Setup of various functions to be used
G = @(F,p) F./(p'*F);
H_ell = @(F,p,N) -1/N.*G(F,p)*G(F,p)';

options = {12, 'diff', true};
stabPt = @(F,p) stablepoint(F,p,options{:});

for i=K:-1:1
f{i} = @(x) mvnpdf(x,mu(i,:),squeeze(sigma(:,:,i)));
end
F=@(x) [f{1}(x),f{2}(x),f{3}(x)]';

%% Set up the mesh we will be using
fullImg=true;
% V will be the 3 dimensional simplex (option 1)
resolution=.01;

if(fullImg)
    %can the following be vectorized!?!
    v=zeros(K,1000);
    k=1;
    for i=0:resolution:1
        for j=0:resolution:1-i
            v(:,k)=[i;j;abs(1-i-j)];
            k=k+1;
        end
    end
% V as the line between 2 points in the simplex(option 2)
else
    t=0:resolution:1;
    a=[.95;.05;0];
    b=[0;.05;.95];
%     a=[0;.25;.75];
%     b=[.36;.64;0];
    s=(ones(1,size(t,2))-t);
    u=t.*a+s.*b;
    e1=t.*[1;0;0]+s.*[0;1;0];
    e2=t.*[0;1;0]+s.*[0;0;1];
    e3=t.*[0;0;1]+s.*[1;0;0];
    v=[u,e1,e2,e3];
end
%extract the x, y, and z coordinates for plotting
x=v(1,:);
y=v(2,:);
z=v(3,:);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% For image understanding
% Draw lines to convergent points
% [img,~]=MeshMapK(F,v,10,'false');
% b=img(:,4);
% t=0:resolution:1;
% s=(ones(1,size(t,2))-t);
% line1=t.*img(:,1)+s.*b;
% line2=t.*img(:,2)+s.*b;
% line3=t.*img(:,3)+s.*b;
% line4=t.*img(:,5)+s.*b;
% line5=t.*img(:,6)+s.*b;
% line7=t.*img(:,7)+s.*b;
% lines=[line1,line2,line3,line4,line5,line7];

%% Make frames
p_0 = ones(K,1)./K;
[pHat,orbit] = stabPt(F(X),p_0);
numIter = size(orbit,1);
mesh = v;

for ii = numIter*2:-1:1
[img,colors]=SingleMeshMapK(F(X),mesh,8,'diff');

figures2(ii)=figure;
scatter3(img(1,:),img(2,:),img(3,:),36,1:size(img,2),'^','filled');

mesh = img;
end

for ii=1:numIter*2
axes2 = figures2(ii).Children;
view(axes2,[135 25]);
grid(axes2,'on');
%colormap(axes2,'jet');
end
%%write 
[~,~,finalcolors,~] = MeshMap3(F(X),.01,8,'diff',false);
%% this fails in the loop
figure1=figure;
scatter3(x,y,z,5,finalcolors,'filled');
axes1 = figure1.Children;
view(axes1,[135 25]);
grid(axes1,'on');
%colormap(axes1,'jet');

