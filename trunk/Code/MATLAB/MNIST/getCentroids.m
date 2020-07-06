function [Cent,WeightCent,BBCent] = getCentroids(image)
%GETCENTROIDS returns the centroid, the weighted centroid and the center of
%the bounding box for IMAGE using MATLAB function regionprops.
%   Detailed explanation goes here
I=image;
BW = imbinarize(I);

props = regionprops(BW,I,{'Centroid','WeightedCentroid','BoundingBox'});

bb=props.BoundingBox;
Cent = props.Centroid;
WeightCent = props.WeightedCentroid;
BBCent = [bb(1)+bb(3)/2,bb(2)+bb(4)/2];

end

