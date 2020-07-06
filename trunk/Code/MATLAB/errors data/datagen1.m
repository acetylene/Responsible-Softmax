function X=datagen1(Nvec,mean_var)
% Usage: X=datagen1(Nvec,mean_var)
% generate 2D mixture of Gaussian data
% Nvec: (1xclass) # data in each of the c gaussian distr.
% mean_var: (3 x class): mean (1st 2 rows) and variance of each class.
% copyright (c) 1996 by Yu Hen Hu
% created:  9/3/96

[m,c]=size(mean_var);
if m ~= 3 || c ~=length(Nvec),
   error(' dimension not match, break ')
end
X=[];
for i=1:c,   
   randn('seed',sum(100*clock));
   tmp=sqrt(mean_var(3,i))*randn(Nvec(i),2); % scaled by variance
   mean=mean_var(1:2,i);  % mean is a 2 by 1 vector
   X=[X;tmp+ones(Nvec(i),2)*diag(mean)];
end
