function [dPidFHadj] = derivPiHatvecAdj(F,piHat,h)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
[K,~] = size(F);
[Hl,dl]=lDifferentials(F,piHat);
dRdPi = Hl.*piHat+diag(dl);
V = eye(K)-dRdPi;
dPidFHadj = derivRFvecAdj(F,piHat,V'\h);
end

