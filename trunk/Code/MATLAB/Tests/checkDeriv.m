function [pass] = checkDeriv(f,Df,x,h)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

%% check DF is linear
%% check that norm(f(x+h)-f(x)-Df(x,h))/norm(h)is 'reasonable' in size 
% O(norm(h)) or so.  Perhaps for several multiples of h, e.g. h, 2h, .5h,
% etc.
%% Find another way to do numerical diff and check that way?
% Maybe decompose h as a sum of basis vctors (if possible) and check
% numerical partial derivatives?
%% Switch roles of x and h, if possible.



pass=True;
end

