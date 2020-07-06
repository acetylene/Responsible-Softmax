classdef objectClassLayer < nnet.layer.ClassificationLayer
        
    properties
        % (Optional) Layer properties.
        op;
        % Layer properties go here.
    end
 
    methods
        function layer = objectClassLayer(numClasses)           
            % (Optional) Create a myClassificationLayer.
                tol = tolCheckerEuc(1);                              
                layer.op = responsibilityOperator(numClasses,tol);
            
        end

        function loss = forwardLoss(layer, Y, T)
            % Return the loss between the predictions Y and the training 
            % targets T.
            %
            % Inputs:
            %         layer - Output layer
            %         Y     – Predictions made by network
            %         T     – Training targets
            %
            % Output:
            %         loss  - Loss between Y and T

            % Layer forward loss function goes here.
            sprintf('In forward. Operator dim is %d',layer.op.dim)
            if layer.op.dim == size(T,3)+1
                M = layer.op.dim;
            else
                M = size(T,3);
            end
            layer.op.setDim(M+1);
            sprintf('In forward. Operator dim is now %d',layer.op.dim)
            loss = cast(1,'like',Y);
        end
        
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

            % Layer backward loss function goes here.
            sprintf('In backward. Operator dim is %d',layer.op.dim)
            if layer.op.dim == size(T,4)+1
                M = layer.op.dim;
            else
                M = size(T,4);
            end
            layer.op.setDim(M+1);
            sprintf('In backward. Operator dim is now %d',layer.op.dim)
            dLdY = cast(zeros(size(Y)),'like',Y);
        end
    end
end