classdef myExpLayer < nnet.layer.Layer

    properties
        % (Optional) Layer properties.

        % Layer properties go here.
    end

    properties (Learnable)
        % (Optional) Layer learnable parameters.
            
        % Layer learnable parameters go here.
        Alpha
    end
    
    methods
        function layer = myExpLayer(numChannels, name)
            % (Optional) Create a myLayer.`
            % This function must have the same name as the layer.

            % Layer constructor function goes here.
            % Set layer name.
            layer.Name = name;
            
            % Set Layer description.
            layer.Description = "Exponential Layer with " + numChannels + " channels";
            
            % Initialize scaling coefficient.
            layer.Alpha = rand([1 1 numChannels]);
        end
        
        function Z = predict(layer, X)
            % Forward input data through the layer at prediction time and
            % output the result.
            %
            % Inputs:
            %         layer    -    Layer to forward propagate through
            %         X        -    Input data
            % Output:
            %         Z        -    Output of layer forward function
            
            % Layer forward function for prediction goes here.
            Z=exp(layer.Alpha.*X);
        end

%           NOT NEEDED FOR THIS LAYER - May include if I want to do
%           attention at a later date
%
%         function [Z, memory] = forward(layer, X)
%             % (Optional) Forward input data through the layer at training
%             % time and output the result and a memory value.
%             %
%             % Inputs:
%             %         layer  - Layer to forward propagate through
%             %         X      - Input data
%             % Outputs:
%             %         Z      - Output of layer forward function
%             %         memory - Memory value for backward propagation
% 
%             % Layer forward function for training goes here.
%         end

        function [dLdX, dLdAlpha] = backward(layer, X, Z, dLdZ, memory)
            % Backward propagate the derivative of the loss function through 
            % the layer.
            %
            % Inputs:
            %         layer             - Layer to backward propagate through
            %         X                 - Input data
            %         Z                 - Output of layer forward function            
            %         dLdZ              - Gradient propagated from the deeper layer
            %         memory            - Memory value from forward function
            % Outputs:
            %         dLdX              - Derivative of the loss with respect to the
            %                             input data
            %         dLdAlpha          - Derivatives of the loss with respect to each
            %                             learnable parameter
            
            dLdX = layer.Alpha.*dLdZ.*Z;
            %dLdX = sum(dLdX,[1 2 4]);
            
            dLdAlpha = dLdZ.*X.*Z;
            dLdAlpha = sum(dLdAlpha,[1 2 4]);
        end
    end
end