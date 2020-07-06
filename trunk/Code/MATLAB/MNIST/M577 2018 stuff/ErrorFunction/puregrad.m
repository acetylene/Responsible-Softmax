%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%% M-file : puregrad.m
%% Author: Marek Rychlik (8-28-2018)
%% 
%% Use a single sample and apply pure (non-stochastic) steepest
%% descent method.
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N=1000;					% Number of samples
NEpochs=100;                            % Number of epochs
eta = 1e-3;				% Learning rate
alpha = 0.1;				% Regularizer constant
min_eta = 1e-8;                         % Stop if learning rate drops below

sigma=.5;				% Sigma of the distribution
D=2;					% Dimension

[X,T] = gen_sample(N, D, sigma);        % The sample

W = random('normal',0,1./sqrt(2*alpha),[1,D]);% Starting weights

Y = sigmoid(X * W');                    % Compute activations
E = T - Y;				% Errors
DW = - E'*X + alpha * W;                % Gradient
G = loss(W,X,T,alpha);                  % Starting loss
Wn = [W];                               % Initial list of W
Gn = [G];                               % Initial list of G

for epoch = 1:NEpochs
    disp(['Epoch: ',num2str(epoch)]);
    W_old = W;
    W = W - eta * DW;                   % Update weights
    Wn = [Wn;W];                        % Add new weight
    Y = sigmoid(X * W');                % Compute activations
    E = T - Y;				% Errors
    DW_old = DW;                        % Save old gradient
    DW = - E'*X + alpha * W;            % Gradient of loss
    G = loss(W,X,T,alpha);              % Compute the loss
    disp(['Loss: ',num2str(G)]);
    Gn = [Gn;G];                        % Add new loss

    % Adjust learning rate according to Barzilai-Borwein
    eta = ((W - W_old) * (DW - DW_old)') ...
          ./ (eps + norm(DW - DW_old).^2);

    eta = min(eta, 1);
    disp(['Learning rate:',num2str(eta)]);

    if eta < min_eta
        disp('Learning rate threshold met, stopping...');        
        break;
    end

    %% Limit weight history to 10
    if size(Wn,1) == 11
        Wn = Wn(2:11, :);
    end



    %% Limit weight history to 10
    % if size(Wn,1) == 11
    %   Wn = Wn(2:11, :);
    % end

    % Visualize  learning

    subplot(2,2,1), plot(Wn(:,1),Wn(:,2),'o-'),; % Weights
    pbaspect([1 1 1]),;
    title(['Current weight vector: ',num2str(W)]),;

    subplot(2,2,2), scatter(X(:,1),X(:,2),3,T),;% Scatter plot samples
    pbaspect([1 1 1]),
    % w1 * x1 + w2 * x2 = 0 => x2 = -w2/w1 * x1
    [S,L] = bounds(X(:,1));
    x1 = [S,L];
    x2 = -W(2)/W(1) * x1;
    line([x1(1),x2(1)], [x1(2),x2(2)], 'Linewidth', 2, 'Color', 'Red'),
    title('Separation'),

    subplot(2,2,[3,4]), plot(Gn,'-o'), 
    title('Learning'),

    drawnow, pause(.2);

end

function [L] = loss(W,X,T,alpha)
    Y = sigmoid(X * W');                % Compute activations/activity
    L = -sum(T .* log( eps + Y) + (1-T) .* log( eps + (1-Y) ), 1);
    L = L + alpha * (W*W');             % Add regularization term
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