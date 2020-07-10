classdef tolCheckerEuc < matlab.unittest.constraints.Tolerance
    %TOLCHECKEREUC Uses standard Euclidean distance to check if two 
    %vectors in the probability simplex in R^N are within a set tolerance.
    %   
    
    properties
        value;
        diagON;
    end
    
    methods
        function obj = tolCheckerEuc(val, varargin)
            %VAL gets assigned to the value of the tolerance checker if it
            %is a positive numeric scalar.
            %   TODO Add customization of distance function
            p = inputParser;
            diagDefault = false;
            addRequired(p,'val',@tolCheckerEuc.checknum);
            addOptional(p,'diagON',diagDefault);
            parse(p,val,varargin{:});
            obj.value = p.Results.val;
            obj.diagON = p.Results.diagON;
        end
        
        function tf = supports(~,V)
            tf = tolCheckerEuc.inSimplex(V);
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
            dist = tolCheckerEuc.distance(actual,expected);
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
                    'The vectors have a Euclidean distance of ', ...
                    distance(actual, expected), ...
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
                
        function d = distance(U,V)
            if any(isnan(U))||any(isnan(V))
                error('Cannot compare two vectors containing NaN')
            else
                d = sqrt(norm(U-V));
            end
        end
        
        function tf = inSimplex(V)
            tf = (isnumeric(V)) && (isvector(V)) && (abs(sum(V)-1)<=1e-4);
        end
    end 
end