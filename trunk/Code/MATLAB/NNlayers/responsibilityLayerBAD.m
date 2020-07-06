classdef responsibilityLayerBAD < nnet.layer.Layer
    
    properties
        % (Optional) Layer properties.
        
        % Layer properties go here.
        err;
        K; %number of output variables
    end
    
    properties (Learnable)
        % (Optional) Layer learnable parameters.
        piHat;
        % Layer learnable parameters go here.
    end
    
    methods
        function layer = responsibilityLayerBAD(numClasses, error)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.
            
            % Layer constructor function goes here.
            layer.K = numClasses;
            layer.piHat = (1./layer.K)*ones(layer.K,1);
            %this should be able to be set at construction (e.g. to the
            %proportions of class labels in the training set!)
            layer.err = error;
        end
        
        function [Z1] = predict(layer, X)
            % Forward input data through the layer at prediction time and
            % output the result.
            %
            % Inputs:
            %         layer       - Layer to forward propagate through
            %         X1, ..., Xn - Input data
            % Outputs:
            %         Z1, ..., Zm - Outputs of layer forward function
            
            % Layer forward function for prediction goes here.
%             fprintf("in predict")
            F = exp(squeeze(X));
            denoms = 1./(layer.piHat'*F);
            Z1=denoms.*F.*layer.piHat;
        end
        
        function [Z, memory] = forward(layer, X)
            % (Optional) Forward input data through the layer at training
            % time and output the result and a memory value.
            %
            % Inputs:
            %         layer       - Layer to forward propagate through
            %         X1, ..., Xn - Input data
            % Outputs:
            %         Z1, ..., Zm - Outputs of layer forward function
            %         memory      - Memory value for backward propagation
            
            % Layer forward function for training goes here.
%             fprintf("in forward")
            try
                F = exp(squeeze(X));%check that squeeze her gives the right 'size'
                layer.piHat = stablepoint(F,layer.piHat,layer.err,'diff',false);
                %put a copy of stablepoint into this class?
                %do i need to calculate stablepoint each time?
                
%               sprintf('size of X (input) is %d by %d',size(X))
                
                denoms = 1./(layer.piHat'*F);
                Z=denoms.*F.*layer.piHat;
                memory{1} = F;
                memory{2} = layer.piHat;
            catch e
                sprintf(e.message)
                rethrow(e)
            end
        end
        
        function [dLdX,dLDpiHat] = ...
                backward(layer, X, Z, dLdZ, memory)
            % Backward propagate the derivative of the loss function through
            % the layer.
            %
            % Inputs:
            %         layer             - Layer to backward propagate through
            %         X1, ..., Xn       - Input data
            %         Z1, ..., Zm       - Outputs of layer forward function
            %         dLdZ1, ..., dLdZm - Gradients propagated from the next layers
            %         memory            - Memory value from forward function
            % Outputs:
            %         dLdX1, ..., dLdXn - Derivatives of the loss with respect to the
            %                             inputs
            %         dLdW1, ..., dLdWk - Derivatives of the loss with respect to each
            %                             learnable parameter
            
            %TODO code something to calculate dRdF, dYdPi, and dYdF, below
            %Y=Z,F=X.
           % fprintf("in backward")
           
           %WE MUST BE SURE THAT X IS NOT NEGATIVE! 2019/07/13
            try
                F=memory{1};
                %pi_old = layer.piHat;
                layer.piHat = memory{2};%this should be stored in a passthrough variable
                [Hl,dl]=lDifferentials(F,layer.piHat);
                dRdPi = Hl.*layer.piHat+diag(dl);
                %sprintf("dRdPi is: "+[repmat(' %d ', 1, layer.K) '\n'],dRdPi)
                %sz=size(X);%check that this has 4 entries, first two should be ones
                
                [K2,~]=size(squeeze(X));%what if K = 1?
                %sprintf("Size of X is :%d by %d",K2,N)
                assert(K2 == layer.K,"Something has gone wrong with the layer output size");
                
                %Not a helpful error message above
                
%                 sprintf('size of X (input) is %d by %d',K2,N)
                
                h=((eye(K2)-dRdPi)^(-1))'*derivYpVecAdj(F,layer.piHat,dLdZ);
                dLdF=derivYFvecAdj(F,layer.piHat,dLdZ)+derivRFvecAdj(F,layer.piHat,h);
                dLdX(1,1,:,:) = dLdZ.*dLdF;
                dLDpiHat = derivPiHatvec(F,layer.piHat,dLdZ);%this works,
                %dLDpiHat = derivPiHatvecAdj(F,layer.piHat,dLdZ);%this doesn't work
            catch e
                sprintf(e.message)
                rethrow(e)
            end
        end
    end
end

%%
% Commented code below may need fixing, it is another way to do backward
%------------------------------------------------------------------------
%             dZdPi = derivYPi(X,layer.piHat,memory);
%
%             dZdX = derivYF(X,layer.piHat,memory);
%
%             dRdPi = squeeze(1/N*sum(dZdPi,2));
%             %This is now also correct! 2019/06/03
%             dRdX = squeeze(1/N*sum(dZdX,4));
%
%             dPidX = zeros(K2,N,K2);
%             for i=1:K2 %this might not be correct, but it should give an output that is not an error
%                 dPidX(:,:,i) = (eye(K2)-dRdPi)^(-1)*squeeze(dRdX(:,:,i));
%             end
%
%             DZDX=zeros(K2,N,K2);
%             for i=1:K2 %this is also probably wrong
%                 DZDX(:,:,i)=squeeze(dZdPi(:,:,i))*squeeze(dPidX(:,:,i))+squeeze(dZdX(:,:,i,:));
%             end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%