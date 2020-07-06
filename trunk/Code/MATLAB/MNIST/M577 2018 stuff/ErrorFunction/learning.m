%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%% M-file : learning.m
%% Author: Marek Rychlik (8-22-2018)
%% 
%% In this script, we generate new sample on every epoch.
%% We also generated a fixed sample for testing.
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N = 100;                                % Number of samples per batch
NEpochs = 100;                          % Number of epochs
eta = .5;				% Learning rate
alpha = 0.2;				% Regularizer constant

sigma=.5;				% Sigma of the distribution


Wn = [];
Gn = [];

D = 2;					% Dimension of samples


W = random('normal',0,3,[1,D]);         % Starting weihgts

[X0,T0] = gen_sample(N, D, sigma);      % Test sample

for epoch = 1:NEpochs

    [X,T] = gen_sample(N, D, sigma);    % Generate minibatch sample.

    % pause();

    Y = sigmoid(X * W');                % Compute activations
    E = T - Y;				% Errors
    DW = - E'*X + alpha * W;
    W = W - eta * DW;
    Wn = [Wn;W];

    G = loss(W,X0,T0,alpha);		        % Test on the original sample
    Gn = [Gn,G];


    %% Limit weight history to 10
    if size(Wn,1) == 11
        Wn = Wn(2:11, :);
    end

    % Visualize  learning

    subplot(2,2,1), plot(Wn(:,1),Wn(:,2),'o-'), % Weights
    pbaspect([1 1 1]),
    title(['Current weight vector: ',num2str(W)]),

    C = 1 - T;                          % Color codes per sample
    subplot(2,2,2), scatter(X(:,1),X(:,2),5,C),% Scatter plot samples
    pbaspect([1 1 1]),
    % w1 * x1 + w2 * x2 = 0 => x2 = -w2/w1 * x1
    [S,L] = bounds(X(:,1));
    x1 = [S,L];
    x2 = -W(2)/W(1) * x1;
    line([x1(1),x2(1)],[x1(2),x2(2)],'Linewidth',2,'Color','Red'),
    title('Separation'),

    subplot(2,2,[3,4]), plot(Gn,'-o'),
    title('Learning'),

    drawnow,pause(.1);

end


function [L] = loss(W,X,T,alpha)
    Y = sigmoid(X * W');                % Compute activations/activity
    L = -sum(T .* log(Y) + (1-T) .* log(1-Y),1);
    L = L +  alpha * (W * W');		% Regularize
end


function y = sigmoid(x)
    y = 1./(1+exp(-x));
end

function [X,T] = gen_sample(N, D, sigma)
    mu=1;					% Mean of normal
    distribution='normal';			% The distribution


    % Draw a new sample from the mixture
    X0 = random(distribution, mu, sigma, [N,D]);
    T0 = zeros(N,1);

    X1 = random(distribution, -mu, sigma, [N,D]);
    T1 = ones(N,1);

    X = [X0;X1];				% Mixture
    T = [T0;T1];				% Targets
    
    % Randomly permute samples
    P = randperm(2*N)';
    X=X(P,:);
    T=T(P,:);
end