function [DpYh] = derivYpVec(F,p,h)
%DERIVYFVEC is a vectorized version of the derivative of Y as described in
% dissertation algorithm 
%   F - A K by N matrix, K is the number of classes, N is the number of
%   samples
%   p - A K by 1 vector of probabilities representing cumulative
%   likelihoods of given classes
%   h - A K by 1 matrix representing a small change in p
Pbar=1./(p'*F);
DpYh = h.*F.*Pbar-p.*F.*(Pbar.^2).*(h'*F);
end

