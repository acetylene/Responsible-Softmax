function [dRdF] = derivRF(F,p)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
[K,N]=size(F);
errmsg=sprintf("The vector p must have the same number of columns as F." +...
    "F has size %d by %d, p has %d columns", K, N, size(p,1));
assert(K==size(p,1),errmsg)

denoms = 1./(p'*F);%perhaps this should be calculated elsewhere and passed? 
%It's not a good idea to just pass variables though, but most of the places
%that I need output from this function also need this.

dRdF = zeros(K,N,K);

for i=1:K
    for j=1:N
        for k=1:K
            dRdF(i:j:k)=0;
        end
    end
end

end

%% TODO this is a stub!?!