function [newImg,numcomponents] = mnistunrotate(img)
%MNISTUNROTATE will help deskew handwritten mnist letters. 
%   Detailed explanation goes here
tmp = reshape(img,numel(img),1);
threshold=mean(tmp)+1.5*var(tmp);

binaryImage = img > threshold;
% Fill in the black letters so we just have a white block.
binaryImage = imfill(binaryImage, 'holes');

% Label each blob with 8-connectivity, so we can make measurements of it
[labeledImage, numberOfBlobs] = bwlabel(binaryImage, 8);
numcomponents=numberOfBlobs;

% Get all the blob properties.
blobMeasurements = regionprops(labeledImage, 'Orientation','MajorAxisLength','Area');

%axisLengths=[blobMeasurements.MajorAxisLength];
allOrientations = [blobMeasurements.Orientation];
areas = [blobMeasurements.Area];

% Extract just one orientation
% We want final image to be horizontally aligned
[~,mainIdx] = max(areas);%max(axisLengths);
angle = allOrientations(mainIdx);

%fprintf('angle is: %d \n mainIdx is: %d',angle,mainIdx);

rotangle=90-angle;

% Rotate the image.
tmp = imrotate(img, rotangle,'bilinear','crop');
newImg = tmp';

end

