function [DFYH] = derivYFvec(F,p,H)
%DERIVYFVEC is a vectorized version of the derivative of Y as described in
% dissertation algorithm 
%   F - A K by N matrix, K is the number of classes, N is the number of
%   samples
%   p - A K by 1 vector of probabilities representing cumulative
%   likelihoods of given classes
%   H - A K by N matrix representing a small change in F
Pbar=1./(p'*F);
DFYH = p.*H.*Pbar-p.*F.*(Pbar.^2).*(p'*H);
end

