function [ new_p ] = lagrange_map( f )
%Lagrange_Map Summary of this function goes here
%   Detailed explanation goes here
[K,N]=size(f);
P=sym('p',[1,K]);
syms l;
%init=1/K*ones(1,K);
denoms=sum(sym(f).*repmat(conj(P'),1,N),1);
%size(denoms)
step= sum(sym(f)./repmat(conj(denoms'),1,N));
S=size(step)
eqns1=  step == l*ones(S);
eqns2= sum(P)==1;
ret=solve([eqns1, eqns2],P);
new_p={P,ret};
end

