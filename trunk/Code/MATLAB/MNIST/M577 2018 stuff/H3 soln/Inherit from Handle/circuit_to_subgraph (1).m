function H = circuit_to_subgraph(V, W)
% Find the matrix representing the circuit W
    n = length(W);
    H = graph(zeros(n,n));
    % Compute positions of cities in the circuit
    j00=find(W(1)==V);
    j0 = j00;
    for i=2:n
        j1=find(W(i)==V);
        H=addedge(H,j0,j1,1);
        j0=j1;
    end
    H=addedge(H,j0,j00,1);
end