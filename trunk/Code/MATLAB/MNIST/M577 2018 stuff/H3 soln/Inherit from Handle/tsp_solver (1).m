%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%% M-file : tsp_solver.m
%% Author: Marek Rychlik (10-9-2018)
%% 
% Implements Hopfield-Tank model from the 1985 paper.
% In this version, we use an ODE solver to solve the
% continuous system for Hopfield-Tank.
%
% Object-oriented version
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

classdef tsp_solver < handle

    properties

        d;                              % cost matrix
        beta;                           % Inverse temp
        initial_beta;                   % Beta to start withh
        delta_beta;                     % Annealing - inverse temp increment
        gamma;                          % lower bound on optimal cost
        n;
        R;
        num_runs;
        tau;
        visualize;                      % Turn visualization on/off
        num_epochs;                     % Number of epochs (= intervals)
        E_threshold;                    % Minimum change of energy

        gradient_delta = 1e-5;          % Delta for numerical gradient
        gradient_err_threshold = 1e-3;  % Gradient validation constant
        done = false;                   % Stop if this is set
    end

    properties(Constant)
        initial_best = struct('E',Inf, 'x', [], 'count', -1);
    end;

    properties(SetObservable, AbortSet)
        best = tsp_solver.initial_best;
    end


    methods
        function obj = tsp_solver(d, varargin)
            p = inputParser;
            validCostMatrix = @(x) size(x,1)==size(x,2);
            addRequired(p,'d',validCostMatrix);

            defaultBeta = 1;                    % Inverse temp.
            validBeta = @(x) isnumeric(x) && isscalar(x) && (x > 0);
            addParameter(p, 'beta', defaultBeta, validBeta);

            defaultDeltaBeta = .1;
            validDeltaBeta = @(x) isnumeric(x) && isscalar(x) && (x > 0);
            addParameter(p, 'betaIncrement', defaultDeltaBeta, validDeltaBeta);

            defaultTau = 1;
            validTau = @(x) isnumeric(x) && isscalar(x) && (x > 0);
            addParameter(p, 'tau', defaultTau, validTau);

            defaultNumRuns = 10;
            validNumRuns = @(x) isnumeric(x) && isscalar(x) && (x > 0);
            addParameter(p, 'numRuns', defaultNumRuns, validNumRuns);

            defaultNumEpochs = 1000;
            validNumEpochs = @(x) isnumeric(x) && isscalar(x) && (x > 0);
            addParameter(p, 'numEpochs', defaultNumEpochs, validNumEpochs);

            defaultEnergyTreshold = 1e-3;
            validEnergyThreshold = @(x) isnumeric(x) && isscalar(x) && (x > 0);
            addParameter(p, 'energyThreshold', defaultEnergyTreshold, ...
                         validEnergyThreshold);

            defaultVisualize = false;
            validVisualize = @(x) islogical(x) && isscalar(x);
            addParameter(p, 'visualize', defaultVisualize, validVisualize);

            parse(p, d, varargin{:});

            obj.d = p.Results.d;
            obj.initial_beta = p.Results.beta;
            obj.delta_beta = p.Results.betaIncrement;
            obj.tau = p.Results.tau;
            obj.num_runs = p.Results.numRuns;
            obj.visualize = p.Results.visualize;
            obj.num_epochs = p.Results.numEpochs;
            obj.E_threshold = p.Results.energyThreshold;

            obj.n = size(obj.d, 1);
            obj.R = obj.n;
            obj.gamma = min(sum(max(obj.d,[],1)), sum(max(obj.d,[],2)));
        end

        function obj = sim(obj)
            obj.best  = tsp_solver.initial_best;
            for run = 1:obj.num_runs
                disp(sprintf('Run: %d', run));
                obj.beta = obj.initial_beta;
                [t, x, E] = seek_equilibrium(obj);
                E_final = E(end);

                % Round the final solution to the nearest vertex
                x_optimal = round(squeeze(x(end,:,:)));
                E_optimal = energy(obj, x_optimal);
                if obj.visualize
                    disp('Nearest vertex');
                    display(x_optimal);    
                    disp(sprintf('Energy at nearest vertex: %6.3f', E_optimal));
                end
                if E_optimal < obj.best.E
                    obj.best = struct('E', E_optimal, ...
                                      'x', x_optimal, ...
                                      'count', 1);
                elseif E_optimal == obj.best.E
                    obj.best.count = obj.best.count + 1;
                end

                if obj.done
                    disp('Done!!!');
                    break;
                end

                disp(sprintf('Best cost: %g, current: %g', obj.best.E, E_optimal));
            end
            disp('---------------- RESULTS ----------------');
            disp(sprintf('Best cost: %g', obj.best.E));
            disp(sprintf('Best configuration (seen %d/%d  times)', ...
                         obj.best.count, obj.num_runs));
            disp(obj.best.x);
        end

        function [tn, xn, En] = seek_equilibrium(obj)
        %Seek equilibrium state of the Hopfield-Tank model
        % [X, E] = SEEK_EQUILIBRIUM(P) accepts parameters P of the
        % Hopfield-Tank model and returns optimual equilibrium configuration
        % X. The second value is the energy of the state X.


            x0 = rand([obj.n, obj.n]);
            t0 = 0;
            y0 = x0(:);

            tn = [];
            yn = [];
            xn = [];    
            En = [];

            for epoch = 1:obj.num_epochs
                [t, y] = ode23(@(y,t)vector_field(obj,y,t), [t0, t0+1], y0);
                %[t, y] = ode45(@(y,t)vector_field(obj,y,t), [t0, t0+1], y0);                
                y0 = y(end,:);
                t0 = t(end);

                x = reshape(y, [size(y,1), obj.n, obj.n]);
                xx = squeeze( x(end,:,:) );

                E = zeros(size(x,1),1);
                for j=1:size(x,1)
                    E(j) = energy(obj, squeeze(x(j,:,:)) );
                end

                % Gather results
                xn = [xn;x];
                tn=[tn;t];
                yn=[yn;y];        
                En = [En;E];

                % Visualization
                if mod(epoch, 2) == 0 && obj.visualize
                    subplot(2,2,[1,2]),plot(tn,En),
                    title(sprintf('Epoch: %3d, learning: %6.3g, beta: %6.3g', ...
                                  epoch, E(end), obj.beta)),
                    subplot(2,2,3),imagesc(xx),
                    title('Matrix'),
                    subplot(2,2,4),plot(tn,yn),
                    title('Entries vs. time'),
                    drawnow;
                end

                if range(E) < obj.E_threshold
                    if obj.visualize
                        disp(sprintf(['Stopping in epoch %3d on threshold ' ...
                                      'met.'], epoch));
                    end
                    break;
                end
                obj.beta = obj.beta + obj.delta_beta;
            end
        end

        function E = energy(obj, x)
        %Computes Hopfield-Tank energy
        % E = ENERGY(X) takes an N-by-N matrix X of values in (0,1)
        % and returns energy E according to the Hopfield-Tank Model
            E = sum(obj.d .* (x * circshift(x, -1, 2)'), 'all') ...
                + obj.gamma .* sum( (sum(x, 1) - 1) .^ 2) ...
                + obj.gamma .* sum( (sum(x, 2) - 1) .^ 2 ) ...
                + obj.R .* sum(x .* (1 - x), 'all') ...
                + obj.gamma .* (x(1,1)-1)^2;
            ;
        end


        function g = energy_gradient(obj, x)
        %Computes the gradient of the Hopfield-Tank energy
        % G = ENERGY_GRADIENT(X) takes an N-by-N matrix X of values in (0,1)
        % and returns the matrix G of the same shape as X, which is
        % the gradient of the Hopfield-Tank energy.
            g = obj.d  * circshift(x,  -1, 2)  + ...
                obj.d' * circshift(x, 1, 2) ...
                + 2 .* obj.gamma .* ( (sum(x,2) - 1) * ones(1, obj.n) ) ...
                + 2 .* obj.gamma .* ( ones(obj.n, 1) * (sum(x,1) - 1) ) ...
                + obj.R .* (ones([obj.n, obj.n]) - 2 .* x);
            g(1,1) = g(1,1) + 2 .* obj.gamma .* (x(1,1)-1);
        end    

        function g = energy_gradient_est(obj, x)
        %Computes the gradient of the Hopfield-Tank energy
        % G = ENERGY_GRADIENT(X) takes an N-by-N matrix X of values in (0,1)
        % and returns the matrix G of the same shape as X, which is
        % the gradient of the Hopfield-Tank energy.
            g = zeros([obj.n,obj.n]);
            for r = 1:obj.n
                for c = 1:obj.n
                    x1 = x; x2 = x;
                    x1(r,c) = x1(r,c) + obj.gradient_delta;
                    x2(r,c) = x2(r,c) - obj.gradient_delta;
                    g(r,c) = ( energy(obj, x1) - energy(obj, x2) ) ./...
                             (2 .* obj.gradient_delta);
                                                                      
                end
            end
        end    

        function dydt = vector_field(obj, t, y)
        %Computes the vector field
            x = reshape(y, [obj.n, obj.n]);% Translate to 2D array
            g = energy_gradient(obj, x);% Find gradient
            b = tsp_solver.sigmoid( -obj.beta .* g );% find activity
            dxdt = - (x - b) ./ obj.tau;% Define vector field
            dydt = dxdt(:);             % Translate to 1D array
        end

        function r = validate(obj)
            x = rand([obj.n, obj.n]);
            g1 = energy_gradient_est(obj, x);
            g = energy_gradient(obj, x);
            err = norm(g1(:) - g(:))
            if err < obj.gradient_err_threshold
                r = true;
            else
                r = false;
            end
        end
    end;                                

    methods(Static)
        function y = sigmoid(x)
            y = 1 ./ (1 + exp(-x));
        end
    end

end