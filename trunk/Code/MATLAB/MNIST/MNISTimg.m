classdef MNISTimg
    %MNISTIMG is a container for MNIST dataset images
    %   IMAGE and LABEL are what their name suggest
    
    properties
        image
        label
    end
    
    methods
        function obj = MNISTimg(newIMG,newLbl)
            %UNTITLED3 Construct an instance of this class
            %   Detailed explanation goes here
            obj.image = newIMG;
            obj.label = newLbl;
        end
        
        function outval = isequal(obj1,obj2)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            outval = obj1.label==obj2.label;
        end
        
        function outval = equals(obj1,value)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            outval = obj1.label==value;
        end
        
        function outval=show(obj)
            outval=imshow(obj.image);
        end
  
    end
end

