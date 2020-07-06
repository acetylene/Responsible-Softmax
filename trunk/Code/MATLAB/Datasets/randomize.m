function B=randomize(A,rowcol)
% Usage: B=randomize(A,rowcol)
% randomize row orders or column orders of A matrix
% rowcol: if =0 or omitted, row order (default)
%         if = 1, column order
% copyright (C) 1996-2001 by Yu Hen Hu
% Last modified: 6/16/2017  Use randperm.m

rand('state',sum(100*clock))
if nargin == 1,
   rowcol=0;
end
if rowcol==0, 
   [m,~]=size(A);
   B = A(randperm(m),:); 
elseif rowcol==1,
   Ap=A';
   [m,~]=size(Ap);
   B=Ap(randperm(m),:)';
end
