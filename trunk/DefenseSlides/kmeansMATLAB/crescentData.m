function [data,labels] = crescentData(K,N,gm,err,shift)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
[X,T] = random(gm,N);
for ii = K:-1:1
    clusters{ii} = X(T==ii);
end
sigma = gm.Sigma;
mu = gm.mu;
%% Transform the data for each cluster onto 4 semicircles
%centers on cluster means
radius = 3*sigma;

% Uses stereographic projection of the circle onto a line to create
% crescents
for ii = K:-1:1
   noise = randn(size(clusters{ii}))*err-shift;
   swap = (-1)^(ii-1);
   r = radius;
   t = clusters{ii}-mu(ii);
   d = (r.^2+t.^2);
   x = 2*r.^2.*t./d+mu(ii);
   y = swap*2*r.^3./d -swap.*noise+swap.*r;
   crescents{ii} = [x,y];
end

data = zeros(N,2);
labels =zeros(N,1);
index = 1;
for ii = 1:K
    last = size(crescents{ii},1);
    data(index:last+index-1,:) = crescents{ii};
    labels(index:last+index-1) = ii*ones(last,1);
    index = index+last;
end


end

