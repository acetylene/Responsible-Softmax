function [stable_dist, orbit] = stablepoint(f_dist, p_dist, err,...
                                            method, orbitON)
%STABLEPOINT  Implements iteration of simplex_map to convergence.
% Different computation methods are available.
%   F_DIST Is a K by N matrix representing the values of the K
%      distributions over the N sample points. The parameters that
%      determine the SIMPLEX_MAP
%   P_DIST Is a K by 1 vector representing initial Mixing probabilities
%   ERR is an error term deciding when to stop.  Should not be bigger than
%   about 12 or 14. This is because log10(eps())-4 = 12 for doubles. For
%   single floating points, this should not be bigger than 4 or 6.
%
%   STABLE_DIST is the stable point to which the iteration of SIMPLEX_MAP
%   converges with the given starting parameters.
%   ORBIT is the orbit of P_DIST under iterations of SIMPLEX_MAP


%% TODO:defaults and varargin handling
stop=2*10^(-err);
p=p_dist;
if orbitON
    orbit=zeros(size(f_dist,1),10^(ceil(err/2)));
end

if strcmp(method,'newton')
    [stable_dist,orbit]=stablepointNewton(f_dist,p,err);
else
    i=1;
    new=simplex_map(f_dist,p,method);
    %Current implemetation uses absolute toleranve. Relative tolerance may
    %be considered at a future date.
    while (sum(abs(p-new))> stop)
        if orbitON
            orbit(:,i)=p;
        end
        i=i+1;
        p=new;
        new=simplex_map(f_dist,p,method);
    end
    stable_dist=new;
end
default = [p_dist,stable_dist];
if orbitON
    orbit=orbit(:,any(orbit,1));
    if isempty(orbit)
        orbit=default;
    end
else
    orbit=default;
end
end
%% TODO: see simplex_map.m
