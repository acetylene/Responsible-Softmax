function [DFRH] = derivRFvec(F,p,H)
%DERIVYFVEC is a vectorized version of the derivative of Y as described in
% dissertation algorithm 
%   F - A K by N matrix, K is the number of classes, N is the number of
%   samples
%   p - A K by 1 vector of probabilities representing cumulative
%   likelihoods of given classes
%   H - A K by N matrix representing a small change in F
[~,N] = size(F);
DFYH=derivYFvec(F,p,H);
DFRH = 1/N*sum(DFYH,2);
end