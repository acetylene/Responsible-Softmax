K=7; N=300;
G = rand(K-3,N);
a=.5; b=.5;
F = [eye(K-3);[diag([a,b,abs(1-a-b)]), zeros(3,K-6)]]*G ;
p = ones(K,1)./K;
pHat = stablepoint(F,p,12,"diff",false);

[Hl,dl]=lDifferentials(F,pHat);
dRdPi = Hl.*pHat+diag(dl);

eig(-Hl)
sum(dRdPi)
eig(dRdPi)
pHat
