first_digit = 3;
second_digit = 8;
[X,T,H,W]=prepare_training_data(first_digit,second_digit);
[U,S,V] = svd(X);                       % Note: X=U*S*V'

n_factors = 16;
grid_size = ceil(sqrt(n_factors));

for j=1:n_factors 
    subplot(grid_size,grid_size,j),imagesc(reshape(V(:,j),H,W)');
end
