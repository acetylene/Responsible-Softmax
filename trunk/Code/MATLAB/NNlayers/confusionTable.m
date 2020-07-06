function [confTable] = ...
    confusionTable(targetData, targetClass, nets, sz)
%CONFUSIONTABLE Displays a set of confusion matrices for the give nets.
%   Detailed explanation goes here
 Ytest=targetData;
 C=targetClass;
 a = sz(1);
 b = sz(2);
 confTable = figure;
for i = 1:length(nets)
    [~,classHat] = max(nets{i}.predict(Ytest),[],2);
    
    subplot(a,b,i)
    confusionchart(C,classHat)
    title(['Net',num2str(i),' Confusion'])
end
 
end

