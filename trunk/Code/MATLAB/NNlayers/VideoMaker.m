classdef VideoMaker < handle
% VIDEOMAKER - class which makes creating videos easy
%   Example: Create 1-frame video with title 'testvideo.avi':
%
%       vidMaker = VideoMaker('VideoTitle', 'dummy_title');
%       plot(1:10);
%       vidMaker.capture_frame;
%       vidMaker.close;
%
    properties
        vidObj;
        isEnabled;
        pauseLength;
        position;
    end
    
    properties(Dependent)
        FrameRate;
    end

    methods 
        function this = VideoMaker(varargin)
        % VIDEOMAKER - constructor
        %     THIS = VIDEOMAKER(...) accepts a number of options:
        %    'VideoTitle'     - the root of the video file to be created
        %    'VideoPosition'  - position on the screen (during video
        %                       making)
        %    'FrameRate'      - number of frames to play per secon
        %    'Pause'          - if disabled, length of pause where
        %                       normally a frame would be captured
        %    'IsEnabled'      - sets the property 'isEnabled' which
        %                       if true causes frame to be captured
            p = inputParser;
            defaultVideoTitle='testvideo';
            p.addParameter('VideoTitle',defaultVideoTitle,...
                           @(x)ischar(x)||isstring(x));
            p.addParameter('VideoPosition',[0,0,1920,1080],@isnumeric);
            p.addParameter('FrameRate',1,@isscalar);
            p.addParameter('Pause', 0, @isscalar);
            p.addParameter('IsEnabled', true, @islogical);
            p.parse(varargin{:});

            this.position = p.Results.VideoPosition;
            set(gcf, 'Position',this.position);
            this.vidObj = VideoWriter(p.Results.VideoTitle);
            this.vidObj.FrameRate = p.Results.FrameRate;
            this.pauseLength = p.Results.Pause;
            open(this.vidObj);
            this.isEnabled = p.Results.IsEnabled;
        end

        function FrameRate = get.FrameRate(this)
            FrameRate=this.vidObj.FrameRate;
        end

        function close(this)
            if isvalid(this)
                close(this.vidObj);
                delete(this);
            end
        end

        function capture_frame(this, num_reps)
        % CAPTURE_FRAME - capture the content of the current figure
        % capture_frame(this, rep)
            narginchk(1,2);
            if nargin < 2
                num_reps = 1;
            end
            if ~isvalid(this)
                error('This instance of Videomaker is no longer valid.');
            end
            if ~isvalid(this.vidObj)
                error('This instance of VideoWriter is no longer valid.');
            end
            if this.isEnabled 
                try
                    set(gcf, 'Position',this.position);
                    drawnow;
                    currFrame = getframe(gcf);
                    for rep=1:num_reps
                        writeVideo(this.vidObj,currFrame);
                    end
                catch me
                    warning('Video closing due to errors');
                    close(this);
                    throw(me);
                end
            elseif this.pauseLength > 0
                pause(num_reps * this.pauseLength);    
            end
        end

    end
end
