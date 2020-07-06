function [dYdF] = derivYF(F,p,denoms)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
[K,N]=size(F);
errmsg=sprintf("The vector p must have the same number of columns as F." +...
    "F has size %d by %d, p has %d columns", K, N, size(p,1));
assert(K==size(p,1),errmsg)

%denoms = 1./p'*F;%perhaps this should be calculated elsewhere and passed? 
%It's not a good idea to just pass variables though, but most of the places
%that I need output from this function also need this.

dYdF = zeros(K,N,K,N);

for i=1:K
    for j=1:N
        for n=1:N
            for k=1:K
                if n==j
                    if k==i
                        dYdF(i,j,k,n)=(p(i).*denoms(n))-(p(i).*p(k).*F(i,j).*denoms(n)^2);
                    else
                        dYdF(i,j,k,n)=-p(i)*p(k)*F(i,j).*denoms(n)^2;
                    end
                else
                    dYdF(i,j,k,n)=0;%This is here to help with clarity
                end
            end
        end
    end
end

end
