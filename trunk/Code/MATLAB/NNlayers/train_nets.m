function [nets] = train_nets(layers,data,targets,options,seed)
%TRAIN_NETS trains the neural networks represented by LAYERS using DATA for
%training. OPTIONS should contain validation data if desired. SEED is used
%for initiating neural net weights in the same manner for each training
%run.
S = length(layers);
for ii = S:-1:1
    rng(seed);
    nets{ii} = trainNetwork(data, targets, layers{ii}, options);
end

end

