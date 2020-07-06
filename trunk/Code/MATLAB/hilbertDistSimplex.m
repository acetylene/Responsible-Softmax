function [ dist,dim ] = hilbertDistSimplex( P,Q,dim )
%UNTITLED2 Summary of this function goes here
%   Implements a version of the hilbert metric on the positive orthant
%   in R^(dim)
assert(length(P)==dim);
assert(length(Q)==dim);
div=P./Q;
M=max(div);
m=min(div);
dist=log(M./m);
end

