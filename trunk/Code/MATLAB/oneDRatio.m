function [ tHat ] = oneDRatio( F, t )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
p=[t;1];
sums=sum(F.*p,1);

tHat=1/size(F,2)*sum(F(1,:)*t./sums)/sum(F(2,:)/sums);

end

