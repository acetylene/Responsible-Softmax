function [X,Y] = noisyFunction(func,numsamples,a,b)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
X=rand(1,numsamples).*(abs(b-a))+a;
Y = func(X)+randn(1,numsamples);
scatter(X,Y)
end

