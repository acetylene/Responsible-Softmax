function [ ImgU, Colors , iterates ] = MeshMapK(F, Mesh, err , orbitsOn, method)
%MESHMAPK applies the K-means iteration function with parameters F to the K-dimensional set of points defined by MESH
%   Detailed explanation goes here

%TODO add capability of using Newton method.
assert(size(F,1)==size(Mesh,1),'F and Mesh must have the same first dimension')
v=Mesh;
l=size(v,2);
formatMsg='Calculating images: %d of %d';
h=waitbar(0/l,sprintf(formatMsg,0,l));

U=zeros(size(v));
if (orbitsOn)
    iterates=zeros(size(v,2),size(v,1),10^(err/2));%placeholder until stablepoint returns the iterates
else
    iterates=[];
end

for k=1:l
    %It might save considerable time to not store the orbits, or perhaps to
    %store them and not skip the points in MESH that are already covered?
    %It may be okay to say things close to something in an orbit go to the
    %same place.  No proof yet, but it appears to be the case.
    [U(:,k),tempiterates]=stablepoint(F,v(:,k),err,method);%todo can I prevent storing the orbits at the next level down?
    if(orbitsOn)
        iterates(k,:,1:size(tempiterates,2))=tempiterates;
    end
    waitbar(k/l,h,sprintf(formatMsg,k,l))
end
delete(h);

rndu=round(U,5);
ImgU=unique(rndu','rows')';

Colors=zeros(1,l);

h=waitbar(0/l,'Calculating colors');
for k=1:l
    [~,Colors(k)]=ismember(rndu(:,k)',ImgU','rows');
    waitbar(k/l,h)    
end
delete(h);


end
