function [newImage,MU,COV] = mnistDeskew(img)%,threshold)
%MNISTDESKEW uses image moments to deskew handwrittten letters
%   Detailed explanation goes here
% x=reshape(img,1,numel(img));
% cutoff = mean(x)+threshold*var(x);
bwImg=imbinarize(img);   %img>cutoff;

[MU,COV]=imageMoments(bwImg);
th=0.5*atan(2*COV(1,2)/(COV(1,1)-COV(2,2)));
alpha = -COV(1,2)/COV(1,1);
beta = -COV(1,2)/COV(2,2);



if  abs(beta) >= .01 &&abs(alpha)>=.01 %
    if abs(alpha)<=abs(beta)
        skMat=[1,0;alpha,1];
    else
        skMat=[1,beta;0,1];
    end
    
    imcenter = size(img)/2;
    offset = MU - imcenter*skMat';
    skewtransform = affine2d([skMat,[0;0];offset,1]);
    
    tmp = imwarp(img,skewtransform);
    
    tmp = imresize(tmp,[20, 20]);
    
    newImage = (tmp-min(min(tmp)))/(max(max(tmp))-min(min(tmp)));
else
    if th>.1%Rotational deskew
        r=[cos(th),sin(th),0;-sin(th),cos(th),0;0,0,1];
        rotTrans=affine2d(r^-1);
        tmp =imwarp(img,rotTrans);
        
        tmp = tmp(sum(tmp,2)>0,sum(tmp,1)>0);
        V=tmp;
        
        a=20/size(V,2);
        b=20/size(V,1);
        stretch=affine2d(diag([a,b,1]));
        V = imwarp(V,stretch);
        
        tmp=V;
        newImage = (tmp-min(min(tmp)))/(max(max(tmp))-min(min(tmp)));
    else
        newImage = (img-min(min(img)))/(max(max(img))-min(min(img)));
    end
    
    
end
end
