classdef responsibilityLoss < nnet.layer.ClassificationLayer
    
    properties
        % (Optional) Layer properties
        piOper;%try a hard coded pihat! This would mean not changing pihat!
        K;
        err;
        numIter;
    end
    methods
        function layer = responsibilityLoss(numClasses, err, varargin)
            % (Optional) Create a myClassificationLayer
            p = inputParser;
            validScalarPosNum =@(x) isnumeric(x) && isscalar(x) && (x > 0);
            validScalarPosInt =@(x) (int32(x)==x) && validScalarPosNum(x);
            addRequired(p,'numClasses',validScalarPosInt);
            addRequired(p,'err',validScalarPosNum);
            
            C=numClasses;
            defaultPi_0 = 1/C.*(ones(C,1));
            validNonNegVec = @(x)(isnumeric(x)&&isvector(x)&&all(x>=0));
            validRatio=@(x)((sum(x)-1)<1e-4) &&...
                (size(x,1)==C) &&...
                validNonNegVec(x);
            addOptional(p,'ratios',defaultPi_0,validRatio);
            addOptional(p,'iterations',1,validScalarPosInt)
            
            parse(p,numClasses,err,varargin{:});
            
            layer.Name = 'Responsibility Loss Layer';
            layer.K = C;
            layer.err = p.Results.err;
            tol = tolCheckerHilb(10^(-err));
            layer.piOper = responsibilityOperator(C,tol,...
                'ratios',p.Results.ratios);
            layer.numIter = p.Results.iterations;
        end
        
        function loss = forwardLoss(layer, Y, T)
            % Return the loss between the predictions Y and the
            % training targets T
            %
            % Inputs:
            % layer - Output layer
            % Y – Predictions made by network
            % T – Training targets
            %
            % Output:
            % loss - Loss between Y and T
            U=Y.extractdata();
            assert(all(U(:)>=0,'all'),"Y must be positive")
            
            %             sprintf('Class of Y is %s.', class(Y))
            %             sprintf('Class of U is %s.', class(U))
            errTol = eps(class(U));
            
            %to resolve issue below.  Hopefully this doesn't run in layer
            %training.
            if ~isa(layer.piOper.pi_0,class(U))
                C= layer.K;
                layer.piOper.setPi_0(cast(ones(C,1)./C,'like',U));
            end
            
            N = size(T,4);
            F = squeeze(Y);
            T = squeeze(T);
            piHat = layer.iteratedResponsibility(F);
            %note: this is carrying over pi hats from previous runs, causes errors!
            layer.piOper.setPi_0(piHat);
            P = piHat'*F;
            B = 1./(P+errTol);
            V = piHat*B;
            Z = V.*F;
            loss = -sum(sum(T.*log(Z)))/N;
            V = loss.extractdata();
            if ~isa(V,class(U))
                disp(class(V))
                disp(class(U))
            end
        end
        
        %         %% must implement backward because of  PO!, tried not using it
        %         with only finite iterations 15 Nov 2019. WORKED!
        %         function dLdY = backwardLoss(layer, Y, T)
        %             % Backward propagate the derivative of the loss function
        %             %
        %             % Inputs:
        %             % layer - Output layer
        %             % Y – Predictions made by network
        %             % T – Training targets
        %             %
        %             % Output:
        %             % dLdY - Derivative of the loss with respect to the predictions Y
        %
        %             assert(all(Y(:)>=0,'all'),"Y must be positive")
        %             errTol = eps(class(Y));
        %             szY = size(Y);
        %             N = size(T,4);
        %             F = squeeze(Y);
        %             T = squeeze(T);
        %             piHat = layer.fixedResponsibility(F);
        %             P = piHat'*F;
        %             B = 1./(P+errTol);
        %             V = piHat*B;
        %             Z = V.*F;
        %
        %             dLdZ = -1/N*(T./(Z+errTol));
        %
        %             dFhadamard = V.*dLdZ;
        %             dV = F.*dLdZ;
        %             dpiHatdot = dV*B';
        %             dB = piHat'*dV;
        %             dP = -dB.*B.^2;
        %             dFdot = piHat*dP;
        %             dpiHatT = F*dP';
        %             dpiHat = dpiHatT + dpiHatdot;
        %             %below might be a source of trouble!
        %             dFpo = responsibilityLoss.dpiHatAdj(F, piHat, dpiHat);
        %             dF = dFhadamard + dFdot + dFpo;
        %             dLdY = reshape(dF,szY);
        %         end
        
        function [pHat] = iteratedResponsibility(layer,F)
            p = layer.piOper.pi_0;
            new_p = responsibilityLoss.responsibilityMap(F,p);
            for ii = 1:layer.numIter
                p = new_p;
                new_p = responsibilityLoss.responsibilityMap(F,p);
            end
            switch class(new_p) 
                case 'dlarray'
                    u = new_p.extractdata();
                    tol = max(eps(u));
                otherwise
                    tol = max(eps(new_p));
            end
            
            
            new_p(new_p<=tol) = new_p(new_p<=tol) + tol;%prevent overflow?
            pHat=new_p./sum(new_p);%place it back in the right space.
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
            % validateattributes(F,{'numeric'},{'>',0})
            % validateattributes(oldP,{'numeric'},{'>=',0})
            % assert(ismatrix(F),"F must be a K by N matrix")
%             sprintf('Class of oldP is %s.', class(oldP))
            switch class(oldP) 
                case 'dlarray'
                    u = oldP.extractdata();
                    err = eps(class(u));
                otherwise
                    err = eps(class(oldP));
            end
            %msg =strcat("oldP must sum to 1: ",sprintf('%d, ',oldP));
            %assert(abs(sum(oldP)-1)<=(2^10)*err,msg)
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
            V = (eye(K) - dRdPi)^-1;
            dFpiHat = derivRFvecAdj(F,piHat,(V)'*dpiHat);%%
        end
        %% TODO: write a few more functions to keep dpiHatAdj contained
    end
end