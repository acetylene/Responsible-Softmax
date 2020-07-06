digit0 = 0;
digit1 = 1;
digit2 = 2;
digit3 = 3;
[X,T] = prepare_training_data(0:9);
% Matlab expects samples in columns and T to be a row vector

X=X'; T=T';                             

% Change pixel value to logarithmic odds off being black
%X = log ( (eps + X) ./ (1 + eps - X) );

% Straight from PATTERNNET help page
net = patternnet(512);                   % Patternnet with hidden layer
net = train(net,X,T);
Y = net(X);
perf = perform(net,T,Y)

% Simulate network, i.e. create outputs
Y=sim(net,X);

% Plot confusion
plotconfusion(T,Y);