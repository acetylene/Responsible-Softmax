function [newT] = iterT(T)
sz=size(T);
N=max(sz);
K=min(sz);
if all(sz==[K,N])
    T=T';
end
if all(sum(T,2)~=1)
    T=T./sum(T,2);
end

p=(1/N.*sum(T))';
d=T*p;

newT=(T.*p')./d;
end

