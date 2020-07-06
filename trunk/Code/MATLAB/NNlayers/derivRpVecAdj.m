function [DpRhadj] = derivRpVecAdj(F,p,h)
%DERIVYFVEC is a vectorized version of the derivative of Y as described in
% dissertation algorithm 
%   F - A K by N matrix, K is the number of classes, N is the number of
%   samples
%   p - A K by 1 vector of probabilities representing cumulative
%   likelihoods of given classes
%   H - A K by N matrix representing a small change in F
[~,N] = size(F);
H = 1/N.*(h*ones(1,N));
DpRhadj = derivYpVecAdj(F,p,H); 
end