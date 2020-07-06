% Adjust for your installation, to point to data provided with H2
file = fullfile('H2 soln','digit_data.mat');
load(file);

X = double(reshape(I, [size(I,1),size(I,2) * size(I,3)]));
% After solution to H2 is posted, you will be able to do:
% [X,T] = prepare_training_data(digit0,digit1,digit2,digit3,digit4);


% Pick only digits 0,1,2,3
IDX = (T<=3);
X = X(IDX,:); 
T = T(IDX);

% Shuffle the samples
N = size(X,1);
P = randperm(N);
X = X(P,:); T = T(P);
X = double(X) ./ 255;


% Matlab expects samples in columns and T to be a row vector
X = X'; T = T';

% Display the first sample as image, as a sanity check
imagesc(reshape(X(:,1),[28,28])');

% Straight from PATTERNNET help page
net = patternnet(10);                    % Equivalent to 'perceptron'
net = train(net,X,T);
Y = net(X);
perf = perform(net,T,Y);

% Simulate network, i.e. create outputs
Y = sim(net,X);

% Plot confusion
%plotconfusion(T,Y);