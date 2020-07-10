function [ new_p ] = simplex_map(  f_dist, old_p, method )
%SIMPLEX_MAP Take in the K by N paramaters f_dist and K by 1 input old_p to create new_p.
% new_p will be a K by 1 vector.
% The purpose of this map is to apply a nonlinear function defined by the
% parameters F_DIST, and apply it to the inputs OLD_P.  This function is
% useful in the study of K-means clustering.
%  F_DIST is an N by K matrix of values taken from K probability
%   distributions on N samples.  The only requirement on F_DIST is that the
%   entries are positive.
%  OLD_P is a 1 by K vector such that the sum of the entries is equal to 1.
%  NEW_P is a 1 by K vector such that the sum of the entries is equal to 1.


[K, N]=size(f_dist);
assert(length(old_p)==K);

if strcmp(method,'ratio')
    prods=bsxfun(@times,f_dist,old_p);
    sums=sum(prods,1);
    assert(length(sums)==N);
    ratios=bsxfun(@rdivide,prods,sums);
    new_p=sum(ratios, 2)/N;
    
elseif strcmp(method,'diff')
    %Worries about underflow!
    denoms=1/N*(1./(f_dist'*old_p));%good here 2/22
    dl=f_dist*denoms;%good here!2/22
    new_p=dl.*old_p;
    
else
    sprintf('You must enter a method of diff or ratio');
end

%% TODO:

% Update tolerance handling to deal with different types of numerical
% values: single, double, int etc.

% Double check for NaN situations in the code.  One way this happens is if
% some entry of denoms is Inf (we divide by zero).  In
% this case should probably assign a zero in the right place. Lookup Matlab
% help at https://www.mathworks.com/help/matlab/matlab_prog/infinity-and-nan.html
% Can some sort of numerical L'Hopital's rule help?

