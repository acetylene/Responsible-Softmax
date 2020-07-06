function [DpRh] = derivRpVec(F,p,h)
%DERIVYFVEC is a vectorized version of the derivative of Y as described in
% dissertation algorithm 
%   F - A K by N matrix, K is the number of classes, N is the number of
%   samples
%   p - A K by 1 vector of probabilities representing cumulative
%   likelihoods of given classes. Requires sum(p)=1.
%   h - A K by 1 matrix representing a small change in p. requires
%   sum(h)=0.
[~,N] = size(F);
DpYh=derivYpVec(F,p,h);
DpRh = 1/N*sum(DpYh,2);
end