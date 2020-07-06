function [dYdPi] = derivYPi(F,p,denoms)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
[K,N]=size(F);
errmsg=sprintf("The vector p must have the same number of columns as F." +...
    "F has size %d by %d, p has %d columns", K, N, size(p,1));
assert(K==size(p,1),errmsg)

%denoms = 1./p'*F;%perhaps this should be calculated elsewhere and passed? 
%It's not a good idea to just pass variables though, but most of the places
%that I need output from this function also need this.

dYdPi = zeros(K,N,K);
%TODO write unit test to compare to first differences!
for i=1:K
    for j=1:N
        for k=1:K
            if k==i
                dYdPi(i,j,k)=F(i,j)*denoms(j)-F(i,j)*F(k,j)*p(i)*denoms(j)^2;
            else
                dYdPi(i,j,k)=-F(i,j)*F(k,j)*p(i)*denoms(j)^2;
            end
        end
    end
end


end