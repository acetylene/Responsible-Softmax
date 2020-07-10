function [acc,confMat,pcts] = ...
    test_nets(nets, test_data, test_targets)
%TEST_NETS uses TEST_DATA to evaluate the performance of neural networks in
%   NETS via confusion matrices.  TEST_TARGETS are the target
%   classifications of TEST_DATA.
S = length(nets);
targets = full(ind2vec(test_targets'));

for i = S:-1:1
    outputs = zeros(size(targets));
    [~,classHat{i}]= max(nets{i}.predict(test_data),[],2);
    outputs = full(ind2vec(classHat{i}'));
    if ~all(size(outputs) == size(targets))
       o = size(outputs);
       t = size(targets);
       if (o(1) < t(1))%it should be the case that o(1)<=t(1)
           pad = zeros(t(1)-o(1),o(2));
           outputs = [outputs;pad];
       end
       if(o(2)~=t(2))
           error('incompatible target and output data size');
       end
    end
    [c(i),cm(:,:,i),~,per(:,:,i)] = confusion(targets,outputs);
end
 acc = c;
 confMat = cm;
 pcts = per;
end

