function make_title_frame(txt, varargin)
    p=inputParser;
    p.addRequired('txt',@(x)ischar(x)||iscell(x));
    p.addOptional('Figure',[]);
    p.addOptional('FontSize',40);
    p.addOptional('FontName','Helvetica');
    p.parse(txt,varargin{:});
    fig = p.Results.Figure;
    set(fig, 'menubar','none') ;
    ah = gca;
    if iscell(txt) 
        tmp ='';
        for j=1:numel(txt)
            tmp = [tmp,'\n\n',txt{j}];
        end
    end
    th = text(1,1,txt,'FontSize',p.Results.FontSize,...
              'FontName',p.Results.FontName);
    set(ah,'visible','off','xlim',[0 2],'ylim',[0 2],'Position',[0 0 1 1]) ;
    set(th,'HorizontalAlignment','center',...
           'VerticalAlignment','middle');
end