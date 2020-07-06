%%%%This script is the beginning of a unit test for the functions
%%%%derivYpi.m and derivYF.m.  Much of the proceudure was inferred from
%%%%calculations in the maxima file ../../Maxima/Finding_dYdF.wxmx
%% staging
rng(3361522108);
K=randi([3,20]);
N=randi([10*K,20*K]);
sprintf('K is %d, N is %d',K,N)
%Below are the G and H from the maxima file
%G=[0.1996785, 0.3358599; 0.1960031, 0.4571297; 0.4883747, 0.0208052]'
%H=[9.571669482429456e-04,8.002804688888001e-04,4.217612826262750e-04; ...
%   4.853756487228412e-04,1.418863386272153e-04,9.157355251890671e-04]
G=rand(K,N);
pHat=stablepoint(G,ones(K,1)./K,12,'diff',true);
%format long
%G
%pHat
Y=parameter_map(G,pHat);
H=rand(K,N)/(100);
dY=parameter_map(G+H,pHat);
sprintf('Norm of deltaY minus Y is %d', norm(dY-Y))

denoms = 1./(pHat'*G);
delY=derivYF(G,pHat,denoms);

%% Test DR via hessians vs vectorized code
delYpi=derivYPi(G,pHat,denoms);

dRsum=1/N*squeeze(sum(delYpi,2));
[Hl,dl]=lDifferentials(G,pHat);
dRHess=Hl.*pHat+diag(dl);
sprintf('Norm of DR via hessian minus DR via sums is %d',norm(dRHess-dRsum))

%% Test that DY and DYvec are within tolerance
delYHvec=derivYFvec(G,pHat,H);

delYH=zeros(K,N);
tmp=zeros(K,N);
for n=1:N
for k=1:K
tmp(k,:) = squeeze(delY(k,:,:,n))*H(:,n);
delYH(k,n)=tmp(k,n);
end
end
norm(delYH-delYHvec)

%% Test that DY is a reasonable frechet derivative
sprintf('Norm of deltaY minus Y  minus DYH is %d',norm(dY-Y-delYH))
sprintf('Norm squared of H is %d',trace(H'*H)^2)

%tests to run: 
% - Are the various derivatives and adjoints in the right place?
% - Do the various methods agree?
% - What happend when adj out is input for fn? vice versa?
% - Is DpR a matrix, or can it be represented as one?
% - Check for when (I-DpR) is non inverstible (DpR has 1 as an e.v.) TODO:
% MATH?

% DRFvec vs. DRFvec Adj. FAIL adj is an order of magnitude larger in dot
% product

% DYFvec vs. DYFvec Adj. FAIL adj is an order of magnitude larger in dot
% product


