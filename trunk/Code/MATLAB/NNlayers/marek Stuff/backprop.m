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
%  'LearningRate'  - is the learning rate;
%  'WeightDecay' - is the weight decay;
%  'NumberOfEpochs'  - the number of epochs.
%
narginchk(2,inf);
p=inputParser;
p.addRequired('x');
p.addRequired('t');
p.addOptional('NumberOfEpochs',1,@(x)isa(x,'double'));
p.addOptional('LearningRate',1,@(x)isa(x,'double'));
p.addOptional('WeightDecay',1,@(x)isa(x,'double'));
p.addOptional('Display','off',@(x)any(validatestring(x,{'on', 'off'})));
p.parse(x,t,varargin{:});    

% Set defaults
L=p.Results.NumberOfEpochs;
eta=p.Results.LearningRate;
alpha=p.Results.WeightDecay;
Display=p.Results.Display;

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