%% SETUP
rng(2147483647)
K=5;
N=25;
p0 = ones(K,1)./K;
F = rand(K,N);
piHat = stablepoint(F,p0,12,"diff",false);

%% Calculate 
for err = 39:-1:0
    dF = rand(K,N)./(2^err);
    results{err+1}.dF = dF;
    H = F+dF;
    results{err+1}.H = H;    
    results{err+1}.piHatH = stablepoint(H,p0,12,"diff",false);
    results{err+1}.dPidF = derivPiHatvec(F,piHat,dF);
end
%% Display
len = length(results);
X=zeros(1,len);
Y=zeros(1,len);
for i=1:length(results)
    X(i) = norm(results{i}.dF);
    tmp = results{i}.piHatH - piHat - results{i}.dPidF;
    Y(i) = norm(tmp);
end

plot(X(30:len),Y(30:len))
hold on
scatter(X(30:len),Y(30:len))
hold off
figure
plot(Y(10:len)./X(10:len),'b-o')%% do log scale plots!