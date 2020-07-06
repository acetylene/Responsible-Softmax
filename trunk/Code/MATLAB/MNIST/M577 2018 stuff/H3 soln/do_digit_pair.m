function [X,T,Y,W,V]=do_digit_pair(first_digit, second_digit, num_epochs, bypass_pca)
[X,T,H,W]=prepare_training_data(first_digit,second_digit);
T=T(:,1);                               % For two classes, only one
                                        % column needed
if bypass_pca
    X1 = [ones([size(X,1),1]),X];
    [Y,~,W] = train_perceptron(X1,T,num_epochs);
    V=[];
else
    [U,S,V] = svd(X);                       % Note: X=U*S*V'
                                            %T = X * V;
    M = U * S;                              % This is the score
    k = 40;
    Mk = M(:,1:k);
    Xk = Mk * V(:,1:k)';

    L = diag(S);
    ExplainedVariance = 1 - sum(L((k+1):end).^2) ./ sum(L.^2)

    % Reduced data in image space
    Digit = reshape(Xk,[size(X,1),W,H]);

    % Show reduced images
    if false
        for j=1:size(Digit,1)
            subplot(2,1,1), imagesc(squeeze(Digit(j,:,:))'),
            subplot(2,1,2), bar(Mk(j,:)),
            drawnow;
            pause(.2);
        end
    end

    clf;
    scatter3(M(T==0,1),M(T==0,2),M(T==0,3),5,'Red','o');
    hold on;
    scatter3(M(T==1,1),M(T==1,2),M(T==1,3),5,'Blue','+'); 
    view(-60,60);
    hold off,
    drawnow;

    Mk = [ones([size(Mk,1),1]),Mk];         % Add a column of ones
    V = V(:,1:k);
    [Y,~,W] = train_perceptron(Mk,T,num_epochs);
end

