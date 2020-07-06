function [mu_vec,covariance] = imageMoments(image)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
[X,Y]=meshgrid(1:size(image,1),1:size(image,2));

total=sum(sum(image));
mu_X=sum(sum(X.*image))/total;
mu_Y=sum(sum(Y.*image))/total;
mu_vec = [mu_X,mu_Y];

mu_20=sum(sum((X-mu_X).^2.*image))/total;
mu_02=sum(sum((Y-mu_Y).^2.*image))/total;
mu_11=sum(sum((X-mu_X).*(Y-mu_Y).*image))/total;

covariance = [mu_20,mu_11;mu_11,mu_02];
end

