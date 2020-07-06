figure
ax1 = axes;
hold on
figure
ax2 = axes;
h = waitbar(0/100,sprintf('On iteration %d of 100',0));
hold on
for J=100:-1:1
    waitbar((100-J+1)/100,h,sprintf('On iteration %d of 100',(100-J+1)))
    [X,T] = random(gm,N);%total sample
    
    for i=K:-1:1
        F(i,:)=normpdf(X,mu(i),sigma(1,1,i));
    end
    p_0 = ones(K,1)./K;
    [pHat,fullOrbit] = stablepoint(F,p_0,8,'diff',true);
    
    % For verification that the finite iterated method does the same as the
    % 'infinitely' iterated method.
   % [finitePi,orbitFinite] = iteratedSimplexMap(F,p_0,900);     
    M = size(fullOrbit,2);
    distancesHilb = zeros(1,M);
    distancesEuc = zeros(1,M);
    for n=(M-1):-1:1
        distancesHilb(n) = hilbertDistSimplex(fullOrbit(:,n),fullOrbit(:,n+1),K);
    end
    
    for n=(M-1):-1:1
        distancesEuc(n) = norm(fullOrbit(:,n)-fullOrbit(:,n+1));
    end
    plot(ax1,1:M,log(distancesHilb))
    plot(ax2,1:M,log(distancesEuc))
    
end


for J=10000:-1:1
    [X,T] = random(gm,N);%total sample
    
    for i=K:-1:1
        F(i,:)=normpdf(X,mu(i),sigma(1,1,i));
    end
    svals(J,:) = svd(F);
end
