%% Remark:
% Verifying the Correctness of a Gradient Implementation). The
% definition of the partial derivatives as the limit of the corresponding dif-
% ference quotient (see (5.39)) can be exploited when numerically checking
% the correctness of gradients in computer programs: When we compute Gradient checking
% gradients and implement them, we can use finite differences to numer-
% ically test our computation and implementation: We choose the value h
% to be small (e.g., h = 10^−4 ) and compare the finite-difference approxima-
% tion from (5.39) with our (analytic) implementation of the gradient. If the
% error is small, our gradient implementation is probably correct. “Small”
% could mean that
% sqrt((Sum_i (dh_i −df_i )^2)/(Sum_i (dh_i +df_i )^2)) < 10^−6 , where dh_i is the finite-difference
% approximation and df_i is the analytic gradient of f with respect to the i th
% variable x_i . 
% See: https://mml-book.com chapter 5.2 for more info

%% TODO: find repeated patterns.  List where each derivative and each adjoint maps to.
%%Test DfR and DfRadj
rng(71194673);
for i=2:20
    K=i;
    N=30*K;%need to find a good bound for this!
    [F,pHat,H,h,U,u,tol]=setup(K,N);
    if ~testAdj(F,pHat,H,h,U,u,tol,K)
        break
    end
end

function [F,pHat,H,h,U,u,tol]=setup(K,N)
    F=rand(K,N);
    p=ones(K,1)./K;
    H=rand(K,N)./1000;
    U=rand(K,N);
    h=rand(K,1);
    h=h./sum(h);
    u=rand(K,1);
    u=u./sum(u);
    pHat = stablepoint(F,p,12,"diff",false);
    %More robust if I had a good bound for what this should do!
    %the bound likely depends on K and N.
    t=100; 
    %max(norm(H)*norm(U),norm(h)*norm(u));%t>0
    if t<1
        tol=eps/t^2;
    else
        tol=eps*t^2;
    end
end

%% Test DF for R and Y
function [success] = testAdj(F,pHat,H,h,U,u,tol,K)
    y=derivRFvec(F,pHat,H);
    a=dot(y,u);
    X=derivRFvecAdj(F,pHat,u);
    b=trace(H'*X);
    failmsg = sprintf("The difference is too big. a-b is %d and tol is %d",...
        abs(a-b),tol);
    assert(abs(a-b)<tol,failmsg)

%[a;b]
%Good here!!


    A=derivYFvec(F,pHat,H);
    a=trace(A'*U);
    X=derivYFvecAdj(F,pHat,U);
    b=trace(H'*X);
    failmsg = sprintf("The difference is too big. a-b is %d and tol is %d",...
        abs(a-b),tol);
    assert(abs(a-b)<tol,failmsg)
    %[a;b]

    %% Test Dpi for R and Y
    A=derivYpVec(F,pHat,h);
    a=trace(A'*U);
    x=derivYpVecAdj(F,pHat,U);
    b=h'*x;
    failmsg = sprintf("The difference is too big. a-b is %d and tol is %d",...
        abs(a-b),tol);
    assert(abs(a-b)<tol,failmsg)

    A=derivRpVec(F,pHat,h);
    a=trace(A'*u);
    x=derivRpVecAdj(F,pHat,u);
    b=h'*x;
    failmsg = sprintf("The difference is too big. a-b is %d and tol is %d",...
        abs(a-b),tol);
    assert(abs(a-b)<tol,failmsg);

    %% Test DFPiHat

    [Hl,dl]=lDifferentials(F,pHat);
    dRdPi = Hl.*pHat+diag(dl);
    V=eye(K)-dRdPi;
    DFR = derivRFvec(F,pHat,H);
    x=V^-1*DFR;
    a=x'*u;
    y=derivRFvecAdj(F,pHat,(V^-1)'*u);
    b=trace(H'*y);
    failmsg = sprintf("The difference is too big. a-b is %d and tol is %d",...
        abs(a-b),tol);
    assert(abs(a-b)<tol,failmsg);
    
    %If we get here, then all the asserts passed.
    success = true;
end
