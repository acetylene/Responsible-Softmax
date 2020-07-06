function [LogCost,grad] = logisticCost(Theta,Data,labels)
%LOGISTICCOST Returns the cross entropy (or Log loss) used to minimize for
%logistic regression.
%   THETA are the weights that we adjust to get the values of the sigmoid
%   function. (The predicted value as the variable Y below)
%   DATA are the observed values
%   LABELS is the appropriate label for each data point.

N=size(Data,1);
X=[ones(N,1),Data];
% disp(['size of X; ', num2str(size(X))])

y=sigmf(X*Theta,[1,0])+eps;%might adding some small, non-normal noise help here? (use max(min(y,1-eps),eps)?)
% disp(['size of y; ', num2str(size(y))])

%negative here is to use minimization instead of maximizing
LogCost=-1/N*sum( labels.*log(y)+(1-labels).*log(1-y)); %should I use some ridge regression here?

grad = zeros(size(Theta,1),1);

for i=1:size(grad)
    grad(i)=1/N*sum((y-labels).*X(:,i));
end

%     function Y=sigmoid(X)
%        Y=1./(1+exp(-X)); 
%     end

end

