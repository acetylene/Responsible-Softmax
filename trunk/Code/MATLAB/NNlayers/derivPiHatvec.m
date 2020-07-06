function [dPidFH] = derivPiHatvec(F,piHat,H)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
[K,~] = size(F);
[Hl,dl]=lDifferentials(F,piHat);
dRdPi = Hl.*piHat+diag(dl);
V = eye(K)-dRdPi;
%  dPidFH = V^(-1)*derivRFvec(F,piHat,H);
% MR: Use left division
dPidFH = V\derivRFvec(F,piHat,H);

end

