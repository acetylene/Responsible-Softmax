function [pi_n,orbit] = iteratedSimplexMap(F,pi_0,n)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
[K,~]= size(F);
assert(K == size(pi_0,1),'F and pi_0 should both have dimension %d.',K)
F = F./sum(F); %can probably remove this with a flag.
pi_old = pi_0;
orbit = zeros(K,n+1);
for ii = 2:n+1
   pi_new = simplex_map(F,pi_old,'diff');
   orbit(:,ii-1) = pi_old;
   pi_old = pi_new;
end
pi_n = pi_old;
end

