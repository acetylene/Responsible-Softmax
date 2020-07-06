function [ results ] = test_map( K, N, varargin  )
%TEST_MAP Summary of this function goes here
%   Detailed explanation goes here
switch nargin
    case 3
        numtrials=varargin{1};
        precision=8;
    case 4
        numtrials=varargin{1};
        precision=varargin{2};
    otherwise
        numtrials=100;
        precision=8;
end



results=zeros(K,N+2,numtrials);

F=rand(K,N);%TODO Make it so that we use non uniform dist...
P=ones(K,1);

for i=1:numtrials %TODO add in collection of the distributions F
    tic
    results(:,1:N,i)=F;
    results(:,N+1,i)=stablepoint(F,P,precision);
    results(:,N+2,i)=toc;

    F=rand(K,N);
    P=ones(K,1);
end

end

