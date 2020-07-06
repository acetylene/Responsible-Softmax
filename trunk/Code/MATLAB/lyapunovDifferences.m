maxiter = 140;
maxK = 20;

differences = zeros(maxK-1,maxiter);
paramCell{maxK-1} = [];

for K = 2:maxK
    
    N = 15*K;
    F = rand(K,N);
    p = ones(K,1)./K;
    
    paramCell{K-1} = F;
    
    differences(K-1,:) = lyapConv(F,p,maxiter);
end

hold on
for K=floor(maxK/2):-1:1
   a(K) = plot(differences(K,1:40));
   M(K) = "K = " + int2str(K);
end
legend(a,M);
hold off
clearvars a M

figure

hold on
for K=maxK-1:-1:(floor(maxK/2))+1
   a(K-floor(maxK/2)) = plot(differences(K,:));
   M(K-floor(maxK/2)) = "K = " + int2str(K);
end
legend(a,M);
hold off


%% TODO:Analysis of convergence WRT K?
function [differences] = lyapConv(F, p, max)
    l = @(F,p) -mean(log(p'*F));
    differences = zeros(1,max);
    for i=1:max
        q = p;
        p = simplex_map(F,q,'diff');
        differences(i) = l(F,q)-l(F,p);
    end
end