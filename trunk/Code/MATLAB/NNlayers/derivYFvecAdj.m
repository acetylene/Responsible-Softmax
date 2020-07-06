function [DFYHadj] = derivYFvecAdj(F,p,H)
%DERIVYFVEC is a vectorized version of the derivative of Y as described in
% dissertation algorithm 
%   F - A K by N matrix, K is the number of classes, N is the number of
%   samples
%   p - A K by 1 vector of probabilities representing cumulative
%   likelihoods of given classes
%   H - A K by N matrix representing a small change in F
Pbar=1./(p'*F);
K = length(p);

DFYHadj = p.*H.*Pbar-(p*ones(1,K))*(p.*F.*(Pbar.^2).*(H));
end