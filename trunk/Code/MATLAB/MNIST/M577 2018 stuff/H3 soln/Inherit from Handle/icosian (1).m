function [A,V,W] = icosian
%Creates the adjacency matrix of the icosian
% [A,V,W] = ICOSIAN returns
%   A - adjacency matrix
%   V - vertex names
%   W - vertices of a Hamiltonian circuit
V = 'abcdefghijklmnopqrst';
W = 'abchsnoklmrgqfpjtide';
N = length(V);

a = sparse([]);
i = 0;
for v = V
    i=i+1;
    a(v)=i;
end
a=full(a);

edges = ['ab';'bc';'cd';'de';'ea';
         'fq';'qg';'gr';'rh';'hs';
         'si';'it';'tj';'jp';'pf';
         'kl';'lm';'mn';'no';'ok';
         'af';'bg';'ch';'di';'ej';
         'kp';'lq';'mr';'sn';'to'];


A=zeros(N,N);
for i = 1: size(edges,1)
    e = edges(i,:);
    e1 = e(1);
    e2 = e(2);
    j1 = a(e1);
    j2 = a(e2);   
    A(j1,j2) = 1;
end
A=A+A';    
imagesc(A);
V=V';