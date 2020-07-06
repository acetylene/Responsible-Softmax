function [nets] = train_nets(layers,data,targets,options,seed)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
S = length(layers);
for ii = S:-1:1
    rng(seed);
    nets{ii} = trainNetwork(data, targets, layers{ii}, options);
end

end

