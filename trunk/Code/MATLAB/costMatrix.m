function [costMat] = costMatrix(nodeMat,probVec)
%COSTMATRIX Takes in a K by N matrix NODEMAT, and a K by 1 vector PROBVEC
%and returns a cost matrix COSTMAT, a K by N by K+1 array. 
%   The first K entries of costMat at the vector at (i,j) are the costs from 
%the j-1 row. The last entry is the cost of the individual node.
%For k < K+1, the (i,j,k) entry is the max entry of the vector 
%v=(k,j-1,1:K).+(k,j-1,K+1).  This algorithm might work better with
%structs.
[K,N]=size(nodeMat);
assert((K>0)&&(N>0),"You must use a nodeMat of size K by N");
assert(length(probVec)==K,"Probvec must have same first dimension as nodeMat");

costMat=zeros(K,N+1,K+1);
costMat(:,1:N,K+1)=log(nodeMat.*probVec);

costVec=zeros(K,1);
for j=2:N+1
    for k=1:K
        tmp=squeeze(costMat(k,j-1,1:K));
        cost=squeeze(costMat(k,j-1,K+1));
        tmp=tmp+cost;
        costVec(k)=max(tmp);
    end
    costMat(:,j,1:K)=repmat(costVec',K,1);
    costVec=zeros(K,1);
end
end

