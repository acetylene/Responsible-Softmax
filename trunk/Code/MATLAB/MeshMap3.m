function [ ImgU, U, Colors, orbits] = MeshMap3( F, resolution, err, method, orbitsON)
%MESHMAP3 applies the K-means iteration function with parameters F 
%        to the 2 simplex in 3 space
%   Detailed explanation goes here
v=zeros(size(F));
k=1;
for i=0:resolution:1
for j=0:resolution:1-i
v(:,k)=[i;j;abs(1-i-j)];
k=k+1;
end
end

l=size(v,2);
h=waitbar(0/l,'Calculating images');

U=zeros(size(v));
for k=l:-1:1
[U(:,k),orbits{k}]=stablepoint(F,v(:,k),err, method, orbitsON);
waitbar((l-k)/l,h)
end
delete(h);

rndu=round(U,5);   
ImgU=unique(rndu','rows')';

Colors=zeros(1,l);
h=waitbar(0/l,'Calculating colors');
for k=1:l
    [~,loc]=ismember(rndu(:,k)',ImgU','rows');
    Colors(k)=loc;
    waitbar(k/l,h)    
end
delete(h);
end

