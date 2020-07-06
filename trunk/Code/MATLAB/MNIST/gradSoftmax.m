function [weights] = gradSoftmax(samples,labels,NEpochs)%,eta0)
% GRADSOFTMAX implements a gradient descent of the softmax function with
% Barzilai-Borwein learning rate.  Make sure your sample has a column of
% ones at the beginning to account for a bias!

%% Use with a bias!
N=size(samples,1);					% Number of samples
%% The below may become inputs
%NEpochs=1000;                            % Number of epochs
eta = 1e-3;				% Learning rate
alpha = 0.075;				% Regularizer constant
min_eta = 1e-8;                         % Stop if learning rate drops below
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

D=size(samples,2);					% Dimension

X=samples';
T=labels';

K=size(labels,2);

W = rand(K,D)-0.5;% Starting weights 
W = W - mean(W);

Y = sftmax(W * X);                      % Compute activations
disp(num2str(size(Y)))
E = T - Y;		                		% Errors
DW = - E*X' + alpha * W;                % Gradient
G = softmLoss(W,X,T,alpha);                  % Starting loss
disp(['Starting Loss: ',num2str(G)]);
%Wn = [W];                               % Initial list of W
%Gn = [G];                               % Initial list of G

for epoch = 1:NEpochs
    disp(['Epoch: ',num2str(epoch)]);
    W_old = W;
    W = W - eta * DW;                   % Update weights
    
    if mod(epoch, 10)==0                % Ensure that mean(W) is 0.   
        W = W - mean(W);
    end
    
    %Wn = [Wn;W];                        % Add new weight
    Y = sftmax(W * X);                % Compute activations
    E = T - Y;				% Errors
    DW_old = DW;                        % Save old gradient
    DW = - E*X' + alpha * W;            % Gradient of loss
    G = softmLoss(W,X,T,alpha);              % Compute the loss
    disp(['Loss: ',num2str(G)]);
%    Gn = [Gn;G];                        % Add new loss
    
    % Adjust learning rate according to Barzilai-Borwein
    etaFrob = trace((W - W_old) * (DW - DW_old)') ...
        ./ (eps + norm(DW - DW_old).^2);
    
    eta = ((W(:) - W_old(:))' * (DW(:) - DW_old(:))) ...
        ./ (eps + norm(DW(:) - DW_old(:)).^2);
    
    eta = min(eta, 1);
       
    disp(['Learning rate:',num2str(eta)]);
    disp(['Alternate Learning rate:',num2str(etaFrob)]);
    
    if eta < min_eta
        disp('Learning rate threshold met, stopping...');
        break;
    end
    
    %% Limit weight history to 10
%      if size(Wn,1) == 11
%          Wn = Wn(2:11, :);
%      end
     
    pause(.01)
    
    %% Limit weight history to 10
    % if size(Wn,1) == 11
    %   Wn = Wn(2:11, :);
    % end
    
    % Visualize  learning
    
%     subplot(2,2,1), plot(Wn(:,1),Wn(:,2),'o-'),; % Weights
%     pbaspect([1 1 1]),;
%     title(['Current weight vector: ',num2str(W)]),;
%     
%     subplot(2,2,2), scatter(X(:,1),X(:,2),3,T),;% Scatter plot samples
%     pbaspect([1 1 1]),
%     % w1 * x1 + w2 * x2 = 0 => x2 = -w2/w1 * x1
%     [S,L] = bounds(X(:,1));
%     x1 = [S,L];
%     x2 = -W(2)/W(1) * x1;
%     line([x1(1),x2(1)], [x1(2),x2(2)], 'Linewidth', 2, 'Color', 'Red'),
%     title('Separation'),
%     
%     subplot(2,2,[3,4]), plot(Gn,'-o'),
%     title('Learning'),
%     
%     drawnow, pause(.2);
    
end
% n=size(Wn,1);
weights = W;

end