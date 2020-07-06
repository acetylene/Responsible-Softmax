classdef respCEntLayer < nnet.layer.RegressionLayer
        
    properties
        % (Optional) Layer properties.
            
    end
 
    methods
        function layer = respCEntLayer()           
            % (Optional) Create a myClassificationLayer.
                layer.Name = 'Responsibility Cross Entropy Classification Layer';
                layer.Description = 'Classification Output';
        end

        function loss = forwardLoss(layer, Y, T)
            % Return the loss between the predictions Y and the training 
            % targets T.  This is a version of CE 
            %
            % Inputs:
            %         layer - Output layer
            %         Y     – Predictions made by network (1 x 1 x K x N)
            %         T     – Training targets (1 x 1 x K x N)
            %
            % Output:
            %         loss  - Loss between Y and T
            N = size(T,4);
            Y = squeeze(Y);
            T = squeeze(T);
            loss = -sum(T.*log(Y),'all')/N;
        end
        
        %Backward Loss does not need to be implemented here!
        %but it causes trouble to do autodiff!
        function dLdY = backwardLoss(layer, Y, T)
            % (Optional) Backward propagate the derivative of the loss 
            % function.
            %
            % Inputs:
            %         layer - Output layer
            %         Y     – Predictions made by network
            %         T     – Training targets
            %
            % Output:
            %         dLdY  - Derivative of the loss with respect to the 
            %                 predictions Y
            N = size(T,4);
            err = eps(class(Y));
            dLdY = -1/N*(T./(Y+err));
        end
    end
end