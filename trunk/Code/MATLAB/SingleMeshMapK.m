function [ ImgU, Colors ] = SingleMeshMapK(F, Mesh, err, method)

% change to a specified number of iterations, and have a keyword for doign
% the full iteration? what about orbits?  Probably best to have two
% functions, one with orbits, and one with limited iteration.  but I can
% call both from the same function!

%MESHMAPK applies the K-means iteration function with parameters F to the
%K-dimensional set of points defined by MESH
%   Detailed explanation goes here
assert(size(F,1)==size(Mesh,1),'F and Mesh must have the same first dimension')
v=Mesh;
l=size(v,2);
formatMsg='Calculating images: %d of %d';
h=waitbar(0/l,sprintf(formatMsg,0,l));

U=zeros(size(v));

for k=1:l
    %this is to work like MESHMAPK, but one do one iteration of the
    %SIMPLEX_MAP on every point in MESH
    [U(:,k)]=simplex_map(F,v(:,k),method);
    waitbar(k/l,h,sprintf(formatMsg,k,l))
end
delete(h);

rndu=round(U,err);
ImgU=unique(rndu','rows')';

Colors=zeros(1,l);

h=waitbar(0/l,'Calculating colors');
for k=1:l
    [~,Colors(k)]=ismember(rndu(:,k)',ImgU','rows');
    waitbar(k/l,h)    
end
delete(h);


end
