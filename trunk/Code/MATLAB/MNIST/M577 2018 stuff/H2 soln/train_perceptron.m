function [Y,NErrors,W] = train_perceptron(X,T, num_epochs)
% TRAIN_PERCEPTRON - train a single perceptron.
%
%    [Y,NERRORS,W] = TRAIN_PERCEPTRON(X,T, NUM_EPOCHS) 
%    accepts these inputs:
%
%    X            the training set; an N-by-D matrix, with data
%                 laid out in rows; 
%
%    T            the target vector; an N-by-1 matrix of 0's and 1's;
%
%    NUM_EPOCHS   the number of epochs; an integer specifying the number of
%                 iterations.
%
%    Every epoch uses all data to compute the gradient, and uses
%    Barzilai-Borwein method to control the learning rate (the step of
%    conventional gradient method). If the learning rate drops below a
%    treshold, the training is stopped before reaching NUM_EPOCHS.
%
%    The output Y is the final output of the neuron. NERRORS is the number
%    of errors if Y is used for maximum likelihood decoding. The number of
%    errors is thus the number of 1's in the vector T~=round(Y).
%
%    The output W is the 1-by-D row vector of weights at the end of
%    training.
    min_eta = 1e-5;                     % Stop if learning rate drops below
    alpha = 1e-1;                       % Regularizer constant
    D = size(X, 2);
    N = size(X, 1);

    SigmaW = (1 / (2 * alpha)) * eye(D);
    W = mvnrnd(zeros([1,D]), SigmaW);   % Starting weihgts
    Y = sigmoid(X * W');                % Compute activations
    E = T - Y;				% Initial Errors
    DW = - E'*X + alpha * W;            % Initial gradient
    eta = 1 /(eps + norm(DW));          % Initial learning rate
    G = cross_entropy(W,X,T);		% Initial cross entropy
    Wn = [W];
    Gn = [G];
    LearningHandle = figure;
    for epoch = 1:num_epochs
        if mod(epoch,10) == 0 ; disp(['Epoch: ',num2str(epoch)]); end;
        W_old = W;                          % Save old weight
        W = W - eta * DW;                   % Update weight
        Y = sigmoid(X * W');                % Compute activations
        E = T - Y;				% Errors
        DW_old = DW;                        % Save old gradient
        DW = - E'*X + alpha * W;            % Update gradient
        Wn = [Wn;W];
        G = cross_entropy(W,X,T);		% Test on the original sample
        G = G +  alpha * (W * W');		% Regularize
        Gn = [Gn,G];

        % Adjust learning rate according to Barzilai-Borwein
        eta = ((W - W_old) * (DW - DW_old)') ...
              ./ (eps + norm(DW - DW_old).^2);

        %eta = min(eta, 3);
        
        disp(['Learning rate: ',num2str(eta)]);

        if eta < min_eta
            disp('Learning rate threshold met, stopping...');        
            break;
        end

        %% Limit weight history to 10
        if size(Wn,1) == 11
            Wn = Wn(2:11, :);
        end
        %% Limit the history to 100
        if length(Gn) == 101
            Gn = Gn(2:101);
        end

        % Visualize  learning
        set(0, 'CurrentFigure', LearningHandle)
        subplot(2,2,[1,2]),
        bar(W),
        title('Weights'),
        subplot(2,2,[3,4]),
        plot(Gn,'-o'), 
        title('Learning'),
        if mod(epoch,10)==0; drawnow; end;
        %pause(.1);
    end

    NErrors = length(find(round(Y)~=T));
    disp(['Number of errors: ', num2str(NErrors)]);

end

function [Z] = cross_entropy(W,X,T)
    Y = sigmoid(X * W');			% Compute activations/activity
    Z = -sum(T .* log(Y+eps) + (1-T) .* log((1-Y)+eps),1);
end

function y = sigmoid(x)
    y = 1./(1+exp(-x));
end

