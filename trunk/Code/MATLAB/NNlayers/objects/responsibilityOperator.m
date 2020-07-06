classdef responsibilityOperator < handle
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        dim;
        pi_0;
        tolChecker;
    end
    
    methods
        function obj = responsibilityOperator(dimension, tolerance, varargin)
            %RESPONSIBIILTYOPERATOR Construct an instance of this class
            %   Detailed explanation goes here
            p = inputParser;
            addRequired(p,'dimension',...
                @responsibilityOperator.validScalarPosInt);
            addRequired(p,'tolerance',@responsibilityOperator.isTol);
            C=dimension;
            ratioCheck = @(x) responsibilityOperator.validRatio(x,C);
            defaultPi_0 = (ones(C,1))./C;
            addOptional(p,'ratios',defaultPi_0,ratioCheck);
            parse(p,dimension,tolerance,varargin{:});
            
            obj.dim = p.Results.dimension;
            obj.pi_0 = p.Results.ratios;
            obj.tolChecker = p.Results.tolerance;
        end
        
        function obj = setDim(obj,D)
            %SETDIM checks that D is a positive integer then sets the
            %dimension of OBJ to D.
            if responsibilityOperator.validScalarPosInt(D)
                obj.dim = D;
            else
                error('Dimension must be a positive integer')
            end
        end
        
        function obj = setPi_0(obj,V)
            %SETPI_0 Checks that V is a valid mixing ratio, and sets
            %OBJ.PI_0 to V if it is.
            if responsibilityOperator.validRatio(V,obj.dim)
                obj.pi_0 = V;
            else
                error('Mixing ratios must be positive reals summing to one')
            end
        end
        
        function obj = setTolerance(obj,tol)
            if responsibilityOperator.isTol(tol)
                obj.tolChecker = tol;
            else
                error('%s %s. %s %s.',...
                    'The input to setTolerance must be of type',...
                    class(matlab.unittest.constraints.Tolerance),...
                    'The variable tol is of type', class(tol));
            end
        end
        
        function resp = iteratedResp(obj, F, p, n)
            %ITERATEDRESP Takes parameters F, and starting point P. It
            %returns RESP which is the first n points in the iteration of
            %the responsibility map.
            resp = zeros(obj.dim,n);
            resp(:,1) = p;
            for i=2:n+1
                resp(:,i) = obj.responsibilityMap(F,resp(:,i-1));
            end
        end
        
        function resp = fixedResp(obj,F,p)
            %FIXEDRESP is theoreticallty equivalent to
            %obj.iteratedResp(F,p,Inf). It finds the fixed point of
            %iterating responsibility. Stopping is determined by
            %obj.tolChecker
            tol = obj.tolChecker;
            old_p = p;
            new_p = obj.responsibilityMap(F,p);
            resp = [old_p,new_p];
            while ~tol.satisfiedBy(new_p,old_p)
                old_p = new_p;
                new_p = obj.responsibilityMap(F,old_p);
                resp = [resp,new_p]; %#ok<AGROW>
            end
        end
        
        %the the following functions be static? Probably so!
        function [dpResp,obj] = derivRp(obj,F,resp,dp_n)
            [K,n] = size(resp); %The number of iterations done is n
            assert(obj.dim == K,'Resp must have same rows as operator dim')
            assert(sum(dp_n)<1e-4,'dp_n must sum to zero')
            %dpResp = zeros(K,n);
            dpResp(:,n) = dp_n;
            for i=n-1:-1:1
                dpResp(:,i) = obj.dRdPiAdj(F,resp(:,i),dpResp(:,i+1));
            end
        end
        
        function [dFResp, dpResp] = derivRFIter(obj,F,resp,dp_n)
           dpResp = obj.derivRp(F,resp,dp_n);
           n = size(resp,2); %The number of iterations done is n            
           dFResp = obj.derivRFvecAdj(F,resp(:,1),dpResp(:,2));
           for i = 2:n-1
               dFResp = dFResp + obj.derivRFvecAdj(F,resp(:,1),dpResp(:,2));
           end
        end
    end
    
    methods(Access = 'private', Static)
        function tf = validScalarPosNum(x)
            tf = isnumeric(x) && isscalar(x) && (x > 0);
        end
        
        function tf = validScalarPosInt(x)
            tf = (int32(x)==x) && responsibilityOperator.validScalarPosNum(x);
        end
        
        function tf = validNonNegVec(x)
            tf = (isnumeric(x)&&isvector(x)&&all(x>=0));
        end
        
        function tf = validRatio(x,dim)
            tf = (abs(sum(x)-1)<10^-4)...
                &&(size(x,1)==dim)&&responsibilityOperator.validNonNegVec(x);
        end
        
        function tf = isTol(x)
            import matlab.unittest.constraints.Tolerance
            tf = isa(x,'Tolerance');
        end
        
        function [newP] = responsibilityMap(F,oldP)
            %             replace with assert
            %             validateattributes(F,{'numeric'},{'>',0})
            %             validateattributes(oldP,{'numeric'},{'>=',0})
            assert(ismatrix(F),"F must be a K by N matrix")
            err = eps(class(oldP));
            %             msg =strcat("oldP must sum to 1: ",sprintf('%d, ',oldP));
            %             assert(abs(sum(oldP)-1)<=(2^14)*err,msg)
            [k,N] = size(F);
            assert(k == length(oldP),"F must have as many columns as oldP")
            
            D = F'*oldP;
            denoms = 1./(D+err);
            newP = 1/N.*oldP.*(F*denoms);
        end
        
        function [dFpiHat] = dpiHatAdj(F, piHat, dpiHat)
            K=size(F,1);
            [Hl,dl]=responsibilityOperator.lDiff(F,piHat);%%
            dRdPi = Hl.*piHat+diag(dl);
            V = (eye(K) - dRdPi);
            dFpiHat = responsibilityOperator.derivRFvecAdj(F,piHat,(V)'\dpiHat);%%
        end
        
        function [dPiAdj] = dRdPiAdj(F, p, dpi)
            [Hl,dl]=responsibilityOperator.lDiff(F,p);%%
            dRdPi = Hl.*p+diag(dl);
            dPiAdj = dRdPi'*dpi;
        end
        
        function [DFRhadj] = derivRFvecAdj(F,p,h)
            [K,N] = size(F);
            H = 1/N.*(h*ones(1,N));
            
            Pbar=1./(p'*F);
            
            DFRhadj = p.*H.*Pbar-(p*ones(1,K))*(p.*F.*(Pbar.^2).*(H));
        end
        
        function [Hl,dl] = lDiff(F,p)
            %LDifferentials calculates the gradient and Hessian of the averaged
            %log likelihood of a joint distribution of N samples from a
            %mixture of K different distributions.
            %   F is a K by N matrix, the evaluations of each point in the various
            %   mixture pdf's
            %   P is a K by 1 vector of the probability components. sum(P) = 1.
            N=size(F,2);
            
            %This has the effect of multiplying each row by the same entry of p, and then summing the columns.
            denoms=p'*F;%F is K by N, P is K by 1.
            err = eps(class(F));
            G=F./(denoms+err);%add err in case an entry of denoms is small.
            dl=1/N.*sum(G,2);
            
            %This comes directly from lemma 4.3 in dissertation
            Hl=-1/N.*(G*G');
        end
    end
end

