function H = matrix_to_subgraph(X)
% Find the matrix representing the circuit W
    assert( size(X, 1) == size(X, 2) );
    n = size(X, 1);
    H = graph(zeros(n,n));
    % Compute positions of cities in the circuit
    j00 = find(X(:,1));
    j0 = j00;
    for i=2:n
        j1=find(X(:,i));
        H = addedge(H,j0,j1,1);
        j0=j1;
    end
    H = addedge(H,j0,j00,1);
end