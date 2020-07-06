function [ iterated ] = iteratedRatio( F,Start,nIter,tol )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
iterated=zeros(1,nIter);
I=1;
iterated(I)=oneDRatio(F,Start);
while(I==1 || (I<nIter && abs(iterated(I-1)-iterated(I))>=tol))
    iterated(I+1)=oneDRatio(F,iterated(I));
    I=I+1;
end


end

