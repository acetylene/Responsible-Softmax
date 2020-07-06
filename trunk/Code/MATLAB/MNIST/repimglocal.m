function [repdImg] = repimglocal(img)
%REPIMGLOCAL is a repeat of some of the code for SINGLEIMAGECOV to allow
%for calling other types of covariance geneteration.
%   see singleimagecov.m for more info.
multiImg=im2col(img,[3 3],'sliding');
multiImg=reshape(multiImg,9,18,18);
repImg=repmat(img,[1, 1, 9]);

multiIdx=[3:20;2:19;1:18];
imgIdx=0;

for i=1:3
    for j=1:3
        imgIdx=imgIdx+1;
        repImg(multiIdx(j,:),multiIdx(i,:),imgIdx)=squeeze(multiImg(imgIdx,:,:));
    end
end

repdImg = repImg;
end

