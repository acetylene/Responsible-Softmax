classdef tolCheckerHilb < matlab.unittest.constraints.Tolerance
    %TOLCHECKERHILB Uses Hilbert's projective distance to check if two 
    %vectors in the probability simplex in R^N are within a set tolerance.
    %   Given two real vectors u, v such that  all entries are positive and 
    %   sum(u)=sum(v)=1, the Hilbert distance between them is defined to be
    %   log(M/m). Here M=max(u./v) and m=min(u./v).  For further
    %   information, see the paper by B. Lemmens and R. Nussbaum at
    %   https://arxiv.org/abs/1304.7921
    
    properties
        value;
        diagON;
    end
    
    methods
        function obj = tolCheckerHilb(val, varargin)
            %VAL gets assigned to the value of the tolerance checker if it
            %is a positive numeric scalar.
            %   TODO Add possible distance function?
            p = inputParser;
            diagDefault = false;
            addRequired(p,'val',@tolCheckerHilb.checknum);
            addOptional(p,'diagON',diagDefault);
            parse(p,val,varargin{:});
            obj.value = p.Results.val;
            obj.diagON = p.Results.diagON;
        end
        
        function tf = supports(~,V)
            tf = tolCheckerHilb.inSimplex(V);
        end
        
        function tf = satisfiedBy(tol, actual, expected)
            if ~tol.supports(actual) || ~tol.supports(expected)
                tf = false;
                return
            end
            if length(actual) ~= length(expected)
                tf = false;
                return
            end
            dist = tolCheckerHilb.hilbertDistance(actual,expected);
            tf = (dist<=tol.value);
            if ~tf && tol.diagON
                sprintf(tol.getDiagnosticFor(actual,expected))
            end
        end
        
        function diag = getDiagnosticFor(tolerance, actual, expected)
            import matlab.unittest.diagnostics.StringDiagnostic
            
            if length(actual) ~= length(expected)
                str = 'Compared vectors must have the same dimension.';
            else
                str = sprintf('%s%d.\n%s%d.', ...
                    'The vectors have a Hilbert distance of ', ...
                    tolerance.hilbertDistance(actual, expected), ...
                    'The allowable distance is ', ...
                    tolerance.value);
            end
            diag = StringDiagnostic(str);
        end
        
    end
    
    methods(Access = private, Static)
        
        function tf = checknum(num)
            tf = false;
            if ~isscalar(num)
                error('Input is not scalar');
            elseif ~isnumeric(num)
                error('Input is not numeric');
            elseif (num <= 0)
                error('Input must be > 0 and <14');
%             elseif (num~=int32(num))
%                 error('Input must be a whole number');
            else
                tf = true;
            end
        end
                
        function dist = hilbertDistance(U,V)
            if any(U == 0)||any(V == 0)
                dist = Inf;
            else
                div=U./V;
                M=max(div);
                m=min(div);
                dist=log(M./m);
            end
        end
        
        function tf = inSimplex(V)
            tf = (isnumeric(V)) && (isvector(V)) && (abs(sum(V)-1)<=1e-4);
        end
    end 
end