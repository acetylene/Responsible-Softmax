mu = [3,-6;0,0;-4,6];

sigma(:,:,1) = [1,.05;.05,1];
sigma(:,:,2) = [1,-2;-2,5];
sigma(:,:,3)=[1.5,-.5;-.5,.95];
gm = gmdistribution(mu, sigma, [.3,.2,.5]);
for i=1:3 F{i} = @(x) mvnpdf(x, mu(i,:), squeeze(sigma(:,:,i)));
end
X = random(gm,1000);
for i=1:3
G(i,:)=F{i}(X);
end
[pNewt,iterNewt]=stablepoint(G,ones(3,1)./3,10,'newton');
[pDiff,iterDiff]=stablepoint(G,ones(3,1)./3,10,'diff');
[pRatio,iterRatio]=stablepoint(G,ones(3,1)./3,10,'ratio');
p=pDiff; 

for i=1:9
[HP{i},DP{i}]=lDifferentials(G,iterDiff(:,i));
RDP{i}=HP{i}*diag(iterDiff(:,i))+diag(DP{i});
end