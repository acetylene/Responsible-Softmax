function [newImg] = svdsmoothing(img,bound)
% SVDSMOOTHING useds svd to throw away small singularvalues and smooth an
% image. We suppose the image is square.
s=size(img,1);

[U,S,V]=svd(img);

Z=S(S>bound);
rem=s-length(Z);

newImg=U*diag([Z;zeros(rem,1)])*V';


end

