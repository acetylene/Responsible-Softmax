function [w,e,o]=backprop(x,t,varargin)
%BACKPROP performs back-propagation training of a neuron
%  [W,E,OBJ]=BACKPROP(X,T,L,ETA,ALPHA) accepts an array X, 
%   which is an N-by-I matrix, with N observations and I variables.
%  The output consists of:
%  W   -  the weight vector of I weights;
%  E   -  the error vector T-Y, where Y is the output vector;
%  O   -  the value of the objective function.
%  
%  This command understands the following options:
%
%  'eta' or 'LearningRate'  - is the learning rate;
%  'alpha' or 'WeightDecay' - is the weight decay;
%  'L' or 'NumberOfEpochs'  - the number of epochs.
%
narginchk(2,inf);
options=varargin;
num_options=length(options);
if mod(num_options,2)~=0
    error('Number of options must be even.');
end;
num_options=num_options/2;
% Parse command line options
for opt=1:num_options
    key=options{2*opt-1};
    val=options{2*opt};
    assert(isa(key,'char'));
    switch key,
      case {'NumberOfEpochs','L'},
        assert(isa(val,'double')),
        L=val;
      case {'LearningRate','eta'},
        assert(isa(val,'double')),
        eta=val;
      case {'WeightDecay','alpha'}
        assert(isa(val,'double')),
        alpha=val;
      case 'Display'
        assert(isa(val,'char'));
        switch val
          case {'on', 'off'},
            Display=val;
          otherwise 
            error(['Option ''Display'' value must be',...
                   '''on'' or ''off''']);
        end
    end
end

% Set defaults
if ~exist('L','var'); L=1;end;
if ~exist('eta','var'); eta=1;end;
if ~exist('alpha','var'); alpha=1;end;
if ~exist('Display','var'); Display='off';end;

% Algorithm 39.5 implementation
[N,I]=size(x);
a=zeros(N,1);
y=zeros(N,1);
% Start with random weights
w=2*rand(I,1)-1;
if strcmp(Display,'on') 
    disp('Initial Weights:');
    disp(w');
end
o=zeros(L,1);
for l=1:L
    a=x*w;                     % compute all activations
    y=sigmoid(a);              % compute outputs
    e=t-y;                     % compute errors
    g=-x'*e;                   % compute the gradient
    w=w-eta*(g+alpha*w);       % make step using learning rate eta
    ol=objective(t,y);
    if strcmp(Display,'on') 
        fprintf('Epoch: %3d, Objective: %12.6g\n',l,o);
        disp('Weights:');disp(w');
    end
    o(l)=ol;
end

function f=sigmoid(v)
%SIGMOID is the logistic function.
f=1./( 1+exp(-v));

function o=objective(t,y)
%OBJECTIVE is the function being *minimized*.
o=-sum(t.*log(y)+(1-t).*log(1.0-y));