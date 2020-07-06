K = 2;

N = 3;

T=ind2vec(randi(K,1,N));

F = rand(K,N)*2;
p = stablepoint(F,ones(K,1)./K,12,'diff',false);

%% TODO: figure out how to test gradients and derivatives in MATLAB.
%   Both autodiff and finite element methods are worth trying.
%   make sure to at least Test DR, Dell, DY and DLoss