function [Y] = parameter_map(F,p)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
denoms = 1./(p'*F);
Y=p.*F.*denoms;

end

