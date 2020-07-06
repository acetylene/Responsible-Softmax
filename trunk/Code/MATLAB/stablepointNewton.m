function [p_star,iterates] = stablepointNewton(F,p,err)
%STABLEPOINTNEWTON A cheap and easy way to determine a stable point of
%SIMPLEX_MAP
%   F Is a K by N matrix representing the values of the K
%      distributions over the N sample points. The parameters that
%      determine the SIMPLEX_MAP
%   P Is a K by 1 vector representing initial Mixing probabilities
%   ERR is an error term deciding when to stop.  Should not be bigger than
%   about 12 or 14.
%
%   P_STAR is the stable point to which the iteration of SIMPLEX_MAP
%   converges with the given starting parameters.
%   ITERATES is the orbit of P_DIST under iterations of the Newton method
K=size(F,1);
dx=ones(K,1);
newp=p;
iterates=p;
while (norm(dx)>10^-err*norm(p))
    [Hl,dl]=lDifferentials(F,newp);
    BHl=[Hl,ones(K,1);ones(1,K),0];
    dlf=[dl;0];
    dx=BHl^-1*dlf;
    dx=dx(1:K);
    newp=abs(newp-dx); %must stay positive
    newp=newp/sum(newp); %keep it on the simplex
    iterates=[iterates,newp];
end
p_star=newp;
end
%% TODO: See todos for simplex_map.m
