classdef fixedRespLoss < nnet.layer.ClassificationLayer
    
    properties
        % (Optional) Layer properties
        pi_0;
        K;
        err;
    end
    methods
        function layer = fixedRespLoss(numClasses, err, varargin)
            % (Optional) Create a fixedRespLoss Layer
            p = inputParser;
            C = numClasses;
            validScalarPosNum =@(x) isnumeric(x) && isscalar(x) && (x > 0);
            validScalarPosInt =@(x) (int32(x)==x) && validScalarPosNum(x);
            validNonNegVec = @(x)(isnumeric(x)&&isvector(x)&&all(x>=0));
            validRatio=@(x)(abs(sum(x)-1)<10^-err)...
                &&(size(x,1)==C)&&validNonNegVec(x);
            addRequired(p,'numClasses',validScalarPosInt);
            addRequired(p,'err',validScalarPosNum);          
            
            defaultPi_0 = 1/C.*(ones(C,1));
           
            addOptional(p,'ratios',defaultPi_0,validRatio);
            
            parse(p,numClasses,err,varargin{:});
            
            layer.Name = 'Responsibility Loss Layer';
            layer.K = C;
            layer.err = p.Results.err;
            layer.pi_0 = p.Results.ratios;
        end
                    
        function loss = forwardLoss(layer, Y, T)
            % Return the loss between the predictions Y and the
            % training targets T
            %
            % Inputs:
            % layer  Output layer
            % Y  Predictions made by network
            % T  Training targets
            %
            % Output:
            % loss  Loss between Y and T
            assert(all(Y(:)>=0,'all'),"Y must be positive")
            errTol = eps(class(Y));
            
            N = size(T,4);
            F = squeeze(Y);
            T = squeeze(T);
            piHat = layer.pi_0;
            P = piHat'*F;
            B = 1./(P+errTol);
            V = piHat*B;
            Z = V.*F;
            loss = -sum(sum(T.*log(Z)))/N;
        end
        
        %% must implement backward when using full dynamic responsibility
        function dLdY = backwardLoss(layer, Y, T)
            % Backward propagate the derivative of the loss function
            %
            % Inputs:
            % layer  Output layer
            % Y  Predictions made by network
            % T  Training targets
            %
            % Output:
            % dLdY - Derivative of the loss with respect to the predictions Y
            
            assert(all(Y(:)>=0,'all'),"Y must be positive")
            errTol = eps(class(Y));
            szY = size(Y);
            N = size(T,4);
            F = squeeze(Y);
            T = squeeze(T);
            piHat = layer.pi_0;
            P = piHat'*F;
            B = 1./(P+errTol);
            V = piHat*B;
            Z = V.*F;
            
            dLdZ = -1/N*(T./(Z+errTol));
            
            dFhadamard = V.*dLdZ;
            dV = F.*dLdZ;
            %dpiHatdot = dV*B';
            dB = piHat'*dV;
            dP = -dB.*B.^2;
            dFdot = piHat*dP;
            %dpiHatT = F*dP';
            %dpiHat = dpiHatT + dpiHatdot;
            %below might be a source of trouble!
            %dFpo = responsibilityLoss.dpiHatAdj(F, piHat, dpiHat);
            dF = dFhadamard + dFdot;% + dFpo;
            dLdY = reshape(dF,szY);
        end
        
        function [pHat] = fixedResponsibility(layer,F)
            stop = 2*10^(-layer.err);
            p = layer.pi_0;
            %i=0;
            
            new = responsibilityLoss.responsibilityMap(F,p);
            while (sum(abs(p-new))> stop*(1+norm(p)+max(eps(p)))) %%check stopping criterion?
                %i=i+1;
                p = new;
                new = responsibilityLoss.responsibilityMap(F,p);
            end
            tol = abs(max(eps(new)));
            new(new<=tol) = new(new<=tol) + tol;%prevent overflow?
            pHat=new./sum(new);%place it back in the right space.
        end
    end
    methods(Access=private,Static)
                     
        function [newP] = responsibilityMap(F,oldP)
            validateattributes(F,{'numeric'},{'>',0})
            validateattributes(oldP,{'numeric'},{'>=',0})
            assert(ismatrix(F),"F must be a K by N matrix")
            err = eps(class(oldP));
%             msg =strcat("oldP must sum to 1: ",sprintf('%d, ',oldP));
%             assert(abs(sum(oldP)-1)<=(2^10)*err,msg)
            [k,N] = size(F);
            assert(k == length(oldP),"F must have as many columns as oldP")
            
            D = F'*oldP;            
            denoms = 1./(D+err);
            newP = 1/N.*oldP.*(F*denoms);
        end        
        
        function [dFpiHat] = dpiHatAdj(F, piHat, dpiHat)
            K=size(F,1);
            [Hl,dl]=lDifferentials(F,piHat);%%
            dRdPi = Hl.*piHat+diag(dl);
            V = (eye(K) - dRdPi);
            dFpiHat = derivRFvecAdj(F,piHat,(V)'\dpiHat);%%
        end
        %% TODO: write a few more functions to keep dpiHatAdj contained
    end
end