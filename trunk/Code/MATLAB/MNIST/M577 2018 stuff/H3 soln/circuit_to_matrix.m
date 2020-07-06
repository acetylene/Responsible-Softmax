function X = circuit_to_matrix(V, W)
% Find the matrix representing the circuit W
    n = length(W);
    X = zeros(n, n);
    % Compute positions of cities in the circuit
    for i=1:n
        j=find(W(i)==V);
        X(j,i)=1;
    end
end