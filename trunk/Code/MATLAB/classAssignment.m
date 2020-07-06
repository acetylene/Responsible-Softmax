function [assignments,costs] = classAssignment(nodeMat,probVec)
%CLASSASSIGNMENT Takes in a K by N matrix NODEMAT, and a K by 1 vector
%PROBVEC and returns
%   Detailed explanation goes here
[K,N]=size(nodeMat);
assert((K>0)&&(N>0),"You must use a nodeMat of size K by N");
assert(length(probVec)==K,"Probvec must have same first dimension as nodeMat");

costMat=log(nodeMat.*probVec);

cumulatCostMat=zeros(K,N);
cumulatCostMat(:,1)=costMat(:,1);

for i=2:N
    cumulatCostMat(:,i)=costMat(:,i)+max(costMat(:,i-1));
end

[costs,assignments] = sort(cumulatCostMat,'descend');

end

