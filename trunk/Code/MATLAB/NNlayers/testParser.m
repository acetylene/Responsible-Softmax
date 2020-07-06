testParse(51,53:55,'d',52)

function pars = testParse(a,b,varargin)
    sprintf('Length of varargin is %d',length(varargin))
    sprintf('varargin{%d} is %d\n',1:length(varargin),varargin{:})

    p = inputParser();
    addRequired(p,'a',@isnumeric)
    addRequired(p,'b',@isnumeric)
    addOptional(p,'d',@isnumeric)
    %sprintf(cell2table(varargin{:}))
    
    parse(p,a,b,varargin{:})
    
    sprintf('The first entry of varargin is %s',varargin{1})
    sprintf('The first arg passed (a) is %s', a)
    sprintf('The 2nd entry of varargin is %s',varargin{2})
    sprintf('The 2nd arg passed (b) is %s', b)
    
    pars = true;
    
end
%notes: a and b are NOT a part of varargin.  you get trouble when trying to
%access parts of varargin that don't exist (indexing). Som issue with type
%checking... 'd' isn't accepted correctly. does it have to do with the way
%that varargin{:} works?
%it works like above... i'm not sure what happened with the script vs. the
%function file!
