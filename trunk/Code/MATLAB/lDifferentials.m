function [Hl,dl] = lDifferentials(F,p)
%LDifferentials calculateds the gradient and hessian of the averaged
%log likelihood of a joint distribution of N samples from a
%mixture of K different distributions.
%   F is a K by N matrix, the evaluations of each point in the various
%   mixture pdf's
%   P is a K by 1 vector of the probability components. sum(P) = 1. 
N=size(F,2);

%This has the effect of multiplying each row by the same entry of p, and then summing the columns.
denoms=p'*F;%F is K by N, P is K by 1.

G=F./denoms;

dl=1/N.*G*ones(N,1);%

%This comes directly from lemma 4.3 in dissertation
Hl=-1/N.*(G*G');%changed, 03/07/19                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      

end

%TODO Revisit this!