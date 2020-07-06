[A, V, W] = icosian

% Visualize Icosian graph and its Hamiltonian circuit
G = graph(A);
H = circuit_to_subgraph(V, W);
h = plot(G,'Layout', 'force3');
highlight(h, H, ...
          'EdgeColor', 'r', ...
          'LineWidth', 2);
drawnow;

D = double(~A);                      % Distance matrix for Hamiltonian
                                        % circuit
rng(666,'twister');

% Create tsp_solver with default cost matrix
obj = tsp_solver(D, ...
                 'beta', .015, ...
                 'betaIncrement', 0.04, ...
                 'numRuns', 100000, ...
                 'tau', 1,...
                 'visualize', false,...
                 'energyThreshold', 3e-2,...
                 'energyAccepted', eps);

% Compute the Hamiltonian circuit as matrix
X = circuit_to_matrix(V, W);
assert( energy(obj, X) == 0);

assert( validate(obj) );

%e = addlistener(obj, 'best', 'PostSet', @myCallback);


% Run simulation
figure;
obj = sim(obj)


% Print best cost
disp(obj.best);

figure;
K = matrix_to_subgraph(obj.best.x);
h = plot(G,'Layout', 'force3');
highlight(h, K, ...
          'EdgeColor', 'b', ...
          'LineWidth', 4);
title('Found Hamiltonian Circuit');