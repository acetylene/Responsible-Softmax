function [L, grad] = softmLoss(W,X,T,alpha)
        Y = sftmax(W * X);                % Compute activations/activity
        L = -sum(sum(T .* log( eps + Y) , 2));%sum along cols first
        L = L + alpha * sum(sum(abs(W)));             % Add regularization term (lasso)
        
        E = T - Y;		                		% Errors
        grad = - E*X' + alpha * W;                % Gradient
end