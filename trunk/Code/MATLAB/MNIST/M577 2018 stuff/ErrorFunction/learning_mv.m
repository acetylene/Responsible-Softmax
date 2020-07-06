%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%% M-file : learning.m
%% Author: Marek Rychlik (8-25-2018)
%% 
%% This script illustrates drawing samples from multidimensional
%% Gaussian distribution. Also, it uses multi-dimensional
%% normal when drawing the initial weight (prior of weights).
%%
%% Example from the manual:
%%        mu = [2 3];
%%        sigma = [1 1.5; 1.5 3];
%%        rng default  % For reproducibility
%%        R = mvnrnd(mu,sigma,100);
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N = 100;                                % Number of samples per batch
E = 100;                                % Number of epochs
eta = 0.5;				% Learning rate
alpha = 0.2;				% Regularizer constant

%% Covariance matrix of the normal distribution;
%% NOTE: Can be either a matrix or a row vector (diagonal)
global Sigma;
Sigma = [.6, 0.2; 0.2,.2];     

Wn = [];
Gn = [];

global D;
D = 2;					% Dimension


%% Generate weights according to Gaussian prior
%% with density exp(-alpha * W' * W). The density
%% of Gaussian in general is proportional to 
%%   exp (-1/2 * (x-mu) * Sigma^(-1) * (x-mu) )
%% where x = W, mu = 0 for our case. Matching we
%% need alpha * W'*W = W' * Sigma^(-1) * W. Thus
%% Sigma^(-1) is diagonal: 2 * alpha * eye(D).
%% Hence:
SigmaW = (1 / (2 * alpha)) * eye(D);
W = mvnrnd([0,0], SigmaW);    % Starting weihgts

[X0,T0] = gen_sample(N,D,Sigma);        % Test sample

for epoch = 1:E

  [X,T] = gen_sample(N,D,Sigma);

  % pause();

  Y = sigmoid(X * W');			% Compute activations
  E = T - Y;				% Errors
  DW = - E'*X + alpha * W;
  W = W - eta * DW;
  Wn = [Wn;W];

  G = loss(W,X0,T0,alpha);		% Test on the original sample
  Gn = [Gn,G];

  %% Limit weight history to 10
  if size(Wn,1) == 11
      Wn = Wn(2:11, :);
  end

  % Visualize  learning
  subplot(2,2,1), plot(Wn(:,1),Wn(:,2),'o-'), % Weights
  pbaspect([1 1 1]),
  title(['Current weight vector: ',num2str(W)]),

  C = 1-T;                              % Color codes per sample
  subplot(2,2,2), scatter(X(:,1),X(:,2),5,C),% Scatter plot samples
  pbaspect([1 1 1]),		
  % w1 * x1 + w2 * x2 = 0 => x2 = -w2/w1 * x1
  [S, L] = bounds(X(:,1));
  x1 = [S, L];
  x2 = -W(2)/W(1) * x1;
  line([x1(1),x2(1)],[x1(2),x2(2)],'Linewidth',2,'Color','Red'),
  title('Separation'),

  subplot(2,2,[3,4]), plot(Gn,'-o'), 
  title('Learning'),

  drawnow;
  pause(.1);

end


function [M] = loss(W,X,T,alpha)
  Y = sigmoid(X * W');			% Compute activations/activity
  G = -sum(T .* log(Y) + (1-T) .* log(1-Y),1);
  M = G +  alpha * (W * W');		% Regularize
end


function y = sigmoid(x)
  y = 1 ./ (1 + exp(-x));
end

function [X,T] = gen_sample(N,D,Sigma)
  mu = [1, 2];                          % Mean of normal

  %% Draw a new sample from the mixture
  X0 = mvnrnd(mu, Sigma, N);            % Multi-dim. Gaussian
  T0 = zeros(N,1);

  X1 = mvnrnd(-mu, Sigma, N);           % Multi-dim. Gaussian
  T1 = ones(N,1);

  X = [X0;X1];				% Mixture
  T = [T0;T1];				% Targets
  
  % Randomly permute samples
  P = randperm(2*N)';
  X=X(P,:);
  T=T(P,:);
end