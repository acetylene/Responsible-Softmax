function [circvec] = circSample(vec)
%CIRCSAMPLE Takes the vector VEC and returns all the cyclic permuatations
%of VEC as CIRCVEC. If VEC is a matrix, it will reshape it to be a vector
%first

lin=reshape(vec, 1, numel(vec));
circvec=zeros(numel(vec));
for i=1:numel(vec)
    circvec(i,:)=circshift(lin,i-1);
end

end

