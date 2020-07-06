function [DFRhadj] = derivRFvecAdj(F,p,h)
%DERIVYFVEC is a vectorized version of the derivative of Y as described in
% dissertation algorithm 
%   F - A K by N matrix, K is the number of classes, N is the number of
%   samples
%   p - A K by 1 vector of probabilities representing cumulative
%   likelihoods of given classes
%   h - A K by 1 matrix representing a small change in p
[~,N] = size(F);
%sprintf('N is %d',N);
%sprintf('The size of h is %d by %d', size(h));
H = 1/N.*(h*ones(1,N));
DFRhadj = derivYFvecAdj(F,p,H); 
end