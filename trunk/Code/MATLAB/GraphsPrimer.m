s = [1 1 2 2 3 3 4 4 5 5 6 7];
t = [2 3 4 5 4 5 6 7 6 7 8 8];
[F,p]=GMMData(2,3,2,233);
phat = stablepoint(F,p',8,'diff');
cost = F.*phat;
B=ones(1,2);%(1,K)
tmp=kron(cost,B);
W=reshape([tmp(:,2:6),B'],1,numel(tmp));%tmp(:,2:2N)
w=log(W);
G=graph(s, t, w);
classAssignment(F,phat)
mod(shortestpath(G,1,8)+1,2)