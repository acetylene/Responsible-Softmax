function [sigma] = blockCov(block)
%BLOCKCOV is designed to return a covariance matrix with just the elemnts
%of the block having any variance. Used in conjunction with blockproc.
%   Uses sparse matrix construction to make it work
data=block.data;
sz=size(data);
m=sz(1);
n=sz(2);
sz=block.imageSize;
M=sz(1);
N=sz(2);
sz=block.location;
r=sz(1);
c=sz(2);

indices=zeros(1,n*m);
pos=1;
for i=0:m-1
    for j=0:n-1
      indices(pos) = (r-1+i)*N+c+j;
      pos=pos+1;
    end
end

dataSig = cov(circSample(data));
assert(numel(dataSig)==n^2*m^2);

i=repmat(indices,1,n*m);
j=reshape(reshape(i,n*m,n*m)',1,n^2*m^2);
v=reshape(dataSig,1,numel(dataSig));

sigma = full(sparse(i,j,v,M*N,M*N));

end

