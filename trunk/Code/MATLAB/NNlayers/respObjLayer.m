classdef respObjLayer < nnet.cnn.layer.SoftmaxLayer
    
    properties
        % (Optional) Layer properties.
        
        % Layer properties go here.
        % piHat;
        err;
        K; %number of output variables
        respOp;
        tol;
        numIter;
    end
    
%     properties (Learnable)
%         (Optional) Layer learnable parameters.
%         pHat; %try a hard coded pihat! This would mean not changing pihat!
%         Layer learnable parameters go here.
%     end
    
    methods
        function layer = respObjLayer(numClasses, error, varargin)
            % Create a Responsibility Layer.
            % This function must have the same name as the class.
            p = inputParser();
            C = numClasses;
            pHat = (1./C)*ones(C,1);
            defaultTol = tolCheckerEuc(error);
            addOptional(p,'p_0',pHat)
            addOptional(p,'steps',1)
            addOptional(p,'Tolerance',defaultTol)
            
            parse(p,varargin{:})
            layer.tol = p.Results.Tolerance;
            layer.err = error;            
            layer.respOp = responsibilityOperator(numClasses,...
                layer.tol, 'ratios', p.Results.p_0);
            layer.K = numClasses;
            layer.numIter = p.Results.steps;         
            
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
            tmp=squeeze(X);
            %check that squeeze gives the right 'size'
            sz = size(tmp);
            assert(ismatrix(tmp),"X must be able to become a matrix")
            assert(sz(1) == layer.K, "X did not have the right number of classes")
            
            F = exp(tmp);
            piHat = layer.pHat; %layer.fixedResponsibility(F);
            %maybe add some plots here?
            P = piHat'*F;
            B = 1./(P);
            V = piHat*B;
            Z1(1,1,:,:) = V.*F;
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
                tmp=squeeze(X);
                %check that squeeze gives the right 'size'
                sz = size(tmp);
                assert(ismatrix(tmp),"X must be able to become a matrix")
                assert(sz(1) == layer.K, "X did not have the right number of classes")
                
                F = exp(tmp - mean(tmp,'all'))+eps(class(X));
                if any(F(:)<0), throw(MLException('%f',F));end
                %validateattributes(F,{'numeric'},{'>',0})%this should only trigger if tmp is cplx
                piHat = layer.fixedResponsibility(F);
%                 validateattributes(piHat,{'numeric'},{'>=',0})
                %MATLAB passes by value
                %do we need to calculate stable point each time?
                
                %sprintf('size of X (input) is %d by %d',size(X))
                P = piHat'*F;
                if any(F(:)<0), throw(MLException('%f',P));end
                %validateattributes(P,{'numeric'},{'>',0})
                B = 1./(P+abs(max(eps(P),[],'all','omitnan')));
                V = piHat*B;
                
                %note 1/10/2019: Z was wrong in previous iterations of the layer
                Z(1,1,:,:) = V.*F;
                assert(all(all(squeeze(Z(1,1,:,:)) == V.*F)), ...
                       'Error in memory passed Z=%s...',...
                       num2str(Z));


                memory.F = F;
                memory.P = P;
                memory.B = B;
                memory.V = V;
                memory.piHat = piHat;
            catch e
                sprintf(e.message)
                rethrow(e)
            end
        end
        
        function [dLdX,dLdpHat] = backward(layer, X, Z, dLdZ, memory)
            % Backward propagate the derivative of the loss function through
            % the layer.
            %
            % Inputs:
            %         layer             - Layer to backward propagate through
            %         X                 - Input data
            %         Z                 - Outputs of layer forward function
            %         dLdZ              - Gradients propagated from the next layers
            %         memory            - Memory value from forward function
            % Outputs:
            %         dLdX1, ..., dLdXn - Derivatives of the loss with respect to the
            %                             inputs
            %         dLdW1, ..., dLdWk - Derivatives of the loss with respect to each
            %                             learnable parameter
            
            %Y=Z,F=X.
            %fprintf("in backward")
            
            try
                F = memory.F;
                P = memory.P;
                B = memory.B;
                V = memory.V;
                piHat = memory.piHat;
                
                %TODO: make memory verification its own function,. include
                %tolerance checking.  Maybe have conditions numbers
                %included for tests involving piHat?
                assert(all(all(squeeze(Z(1,1,:,:)) == V.*F)),...
                    'Error with the memory passed Z = %s',num2str(Z))
                testPi = layer.fixedResponsibility(F);
                assert(all(testPi == piHat),...
                    'Error in memory passed piHat = %s',num2str(piHat))
                assert(all(P == piHat'*F),...
                    'Error in memory passed P = %s',num2str(P))
                assert(all(abs(B - 1./(P+max(eps(P),[],'all','omitnan')))...
                    <10^4*eps('single')),...
                    "Error with the memory passed B = %s",num2str(B))
                
                %sprintf("dRdPi is: "+[repmat(' %d ', 1, layer.K) '\n'],dRdPi)
                szX=size(X);%check that this has 4 entries, first two should be ones
                %%% print statement for szX
                
                tmp=squeeze(X);%this gives a problem if X is not a 4-D array
                %check that squeeze gives the right 'size'
                sz = size(tmp);
                assert(ismatrix(tmp),"X must be able to become a matrix")
                assert(sz(1) == layer.K, "X did not have the right number of classes")
                % assert(all(szX == [1,1,sz]),"Input size mismatch in resLayer")
                
                %sprintf('size of X (input) is %d by %d',K2,N)
                
                %TODO: Figure out a way to work with the 4D array, but
                %still have matrix operations.  MATLAB is made for stuff
                %like this, right?
                dLdZ = squeeze(dLdZ);
                
                dFhadamard = V.*dLdZ;
                dV = F.*dLdZ;
                dpiHatdot = dV*B';
                dB = piHat'*dV;
                dP = -dB.*B.^2;
                dFdot = piHat*dP;
                dpiHatT = F*dP';
                dpiHat = dpiHatT + dpiHatdot;
                dFpo = layer.dpiHatAdj(F, piHat, dpiHat);
                dF = dFhadamard + dFdot + dFpo;
                %%%printstatement for size(dF)
                dLdX = reshape(F.*dF,szX);
                %%%printstatement for size(dLdX)
                
                dLdpHat = derivPiHatvec(F,piHat,dLdZ);%dpiHat;
                %stupid dPiHat
                %dLdpHat = -(layer.pHat-piHat)./(layer.getLearnRateFactor('pHat'));
                %cast(zeros(size(layer.pHat)),'like',layer.pHat);
                
                
%                 g = sprintf('%d, ',dLdpHat);
%                 g(end)='';
%                 h = sprintf('%d, ',layer.pHat);
%                 h(end)='';
%                 sprintf(strcat('phat is: ',h))
%                 sprintf(strcat('Dphat is: ',g))
%                 
            catch e
                sprintf(e.message)
                rethrow(e)
            end
        end
        
        function [piHat] = fixedResponsibility(layer,F)
            p = layer.pHat;%(1./layer.K)*ones(layer.K,1);
            
            %TODO: Better tolerance checking!
            stop = max(2*10^(-layer.err),sum(eps(p)));
            i=0;
            m=min([60,layer.err]);
            maxIter = 10^(m);%to help pervent infinite loops
            
            new = responsibilityLayer.responsibilityMap(F,p);
            while (sum(abs(p-new))> stop*(1+norm(p)+max(eps(p))))% && i<maxIter)
                i=i+1;
                if i == maxIter
                    sprintf('Number of iterations has exceeded %d',maxIter)
                end
                p = new;
                new = responsibilityLayer.responsibilityMap(F,p);
            end
            
            tol = abs(max(eps(new)));
            new(new<=tol) = new(new<=tol) + tol;%prevent underflow?
            piHat=new./sum(new);%place it back in the right space.
        end
        
        function [dFpiHat] = dpiHatAdj(layer, F, piHat, dpiHat)
            [Hl,dl]=lDifferentials(F,piHat);%%
            dRdPi = Hl.*piHat+diag(dl);
            % MR: Replace inverse with pseudo-inverse
            %V = (eye(layer.K) - dRdPi)^-1;
            %dFpiHat = derivRFvecAdj(F,piHat,(V)'*dpiHat);%%
            V = eye(layer.K) - dRdPi;
            dFpiHat = derivRFvecAdj(F,piHat,V'\dpiHat);
        end
    end
    
    methods(Access=private,Static)
        function [newP] = responsibilityMap(F,oldP)
            if any(F(:)<0), throw(MLException('%f',F));end
            %validateattributes(F,{'numeric'},{'>',0})
            %validateattributes(oldP,{'numeric'},{'>=',0})
            assert(ismatrix(F),"F must be a K by N matrix")
            err = eps(class(oldP));
            %msg =strcat("oldP must sum to 1: ",sprintf('%d, ',oldP));
            %assert(abs(sum(oldP)-1)<=(2^10)*err,msg)
            [k,N] = size(F);
            assert(k == length(oldP),"F must have as many columns as oldP")
            
            D = F'*oldP;            
            denoms = 1./(D+err);
            newP = 1/N.*oldP.*(F*denoms);
        end
    end
end

