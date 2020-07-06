%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%% M-file : puregrad.m
%% Author: Marek Rychlik (9-9-2018)
%% 
%% A minimalistic object-oriented implementation of the
%% perceptron training algorithm. Essentially equivalent
%% to puregrad.m.
%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
classdef perceptron
    properties
        N;                              % Number of samples
        D;                              % Input dimension (number of inputs)
        num_epochs = 10000;             % Number of epochs (could be fewer)
        eta = 1e-3;                     % Learning rate
        alpha = 0.1;                    % Regularizer constant
        min_eta = 1e-8;                 % Stop if learning rate drops below
        vis_period=10;                  % Visualize every this many steps
    end;

    properties(SetAccess = 'private', Hidden = false)
        X = [];                         % Input data
        T = [];                         % Target vector
        W;                              % Current weights
        Y;                              % Activation
        E;                              % Error
        DW;                             % Current gradient
        G;                              % Current loss
        epoch = 0;
    end;

    properties(SetAccess = 'private', Hidden = true)
        DW_old;                         % Saved gradient
        W_old;                          % Saved weights
        Gn = [];                        % Loss history
        Wn = [];                        % Weight history
    end;

    methods
        function obj = perceptron(varargin)
        end;

        function obj = train(obj, X, T)
            obj.X = X;
            obj.T = T;
            obj.N = size(obj.X, 1);
            obj.D = size(obj.X, 2);
            obj.W = random('normal', 0, 1 ./ sqrt(2* obj.alpha),...
                           [1,obj.D]); 

            obj.Y = perceptron.sigmoid( obj.X * obj.W' );% Compute activations
            obj.E = obj.T - obj.Y;      
            obj.DW = - obj.E' * obj.X + obj.alpha * obj.W;
            obj.G = loss(obj);  
            obj.Wn = [obj.Wn; obj.W];               
            obj.Gn = [obj.Gn; obj.G];                           

            obj = train_more(obj);
        end

        function obj = train_more(obj)
            if isempty(obj.X)
                error('Call ''train'' first.');
            end
            while obj.epoch < obj.num_epochs
                obj.epoch = obj.epoch + 1;
                disp(['Epoch: ',num2str(obj.epoch)]);
                obj.W_old = obj.W;
                obj.W = obj.W - obj.eta * obj.DW;
                obj.Wn = [obj.Wn; obj.W];   
                obj.Y = perceptron.sigmoid(obj.X * obj.W');   
                obj.E = obj.T - obj.Y;			
                obj.DW_old = obj.DW;
                obj.DW = - obj.E' * obj.X + obj.alpha * obj.W;
                obj.G = loss(obj);

                disp(['Loss: ',num2str(obj.G)]);
                obj.Gn = [obj.Gn; obj.G];

                % Adjust learning rate according to Barzilai-Borwein
                obj.eta = ((obj.W - obj.W_old) * (obj.DW - obj.DW_old)') ...
                          ./ (eps + norm(obj.DW - obj.DW_old).^2);

                obj.eta = min(obj.eta, 1);
                disp(['Learning rate: ',num2str(obj.eta)]);

                if obj.eta < obj.min_eta
                    disp(['Learning rate threshold ', num2str(obj.min_eta), ...
                          ' met, stopping...']);
                    break;
                end

                %% Limit weight history to 10
                if size(obj.Wn, 1) == 11
                    obj.Wn = obj.Wn(2:11, :);
                end

                % Run visualization every so often
                if mod(obj.epoch, obj.vis_period) == 0
                    visualize_learning(obj);
                end
            end;
            % Visualize at least once
            visualize_learning(obj);
        end;

        function [] = visualize_learning(obj)
            subplot(2,2,1), plot(obj.Wn(:,1), obj.Wn(:,2),'o-'); % Weights
            pbaspect([1 1 1]);
            title(['Current weight vector: ',num2str(obj.W)]);
            
            subplot(2,2,2), scatter(obj.X(:,1), obj.X(:,2), 3, obj.T);% Scatter plot samples
            pbaspect([1 1 1]);

            % w1 * x1 + w2 * x2 = 0 => x2 = -w2/w1 * x1
            [S, L] = bounds(obj.X(:,1));
            x1 = [S,L];
            x2 = -obj.W(2)/obj.W(1) * x1;
            line([x1(1),x2(1)], [x1(2),x2(2)], 'Linewidth', 2, 'Color', 'Red'),
            title('Separation'),

            subplot(2,2,[3,4]), plot(obj.Gn,'-o'), 
            title('Learning'),
            drawnow;
        end;

        function [L] = loss(obj)
            obj.Y = perceptron.sigmoid( obj.X * obj.W' );       
            L = -sum(obj.T .* log( eps + obj.Y) + (1-obj.T) .* log( eps + (1-obj.Y) ), 1);
            L = L + obj.alpha * (obj.W * obj.W');
        end;

    end;

    methods(Static)

        function y = sigmoid(x)
            y = 1./(1+exp(-x));
        end

        function [X,T] = gen_sample(N, D, sigma)
        % [X,T] = GEN_SAMPLE(N, D, SIGMA) generates
        % a test data X and target vector T.
            mu=1;                       % Mean of normal
            distribution='normal';      % The distribution


            % Draw a new sample from the mixture
            X0 = random(distribution, mu, sigma, [N,D]);
            T0 = zeros(N,1);

            X1 = random(distribution, -mu, sigma, [N,D]);
            T1 = ones(N,1);

            X = [X0;X1];				% Mixture
            T = [T0;T1];				% Targets
    
            % Randomly permute samples
            P = randperm(2*N)';
            X = X(P,:);
            T = T(P,:);
        end
    end
end