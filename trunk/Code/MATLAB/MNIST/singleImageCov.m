function [covariance,sample] = singleImageCov(img,delta)
%SINGLEIMAGECOV Returns a sparse covariance matrix based on a 3 by 3 box
% around each pixel from the image IMG. If the input image is N by M then 
% COV is N*M by N*M.
%   may need to update this to be more flexible
muFun = @(block_struct) mean2(block_struct.data)*(ones(size(block_struct.data)));
sigFun = @(block_struct) cov(circSample(block_struct.data));


end

%%old code, based on a bad idea
% multiImg=im2col(img,[3 3],'sliding');
% multiImg=reshape(multiImg,9,18,18);
% repImg=repmat(img,[1, 1, 9]);
% 
% multiIdx=[3:20;2:19;1:18];
% imgIdx=0;
% 
% for i=1:3
%     for j=1:3
%         imgIdx=imgIdx+1;
%         repImg(multiIdx(j,:),multiIdx(i,:),imgIdx)=squeeze(multiImg(imgIdx,:,:));
%     end
% end
% 
% sample = reshape(repImg,[9,400]);
% tmpcov = cov(sample);
% covariance = sparse(tmpcov.*(abs(tmpcov)>delta));
