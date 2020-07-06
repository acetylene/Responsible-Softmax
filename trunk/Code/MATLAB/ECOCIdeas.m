%% Length 15 cyclic code
codgen15 = cyclpoly(15,10,'all');
codgen15
[parmat15,genmat15] = cyclgen(15,codegen15(1,:));
[parmat15,genmat15] = cyclgen(15,codgen15(1,:));
genmat 15
genmat15
[parmat15,genmat15] = cyclgen(15,codgen15(3,:));
genmat15
sum(genmat15)
sum(genmat15,2)
genmat15(:,1:5)
%% Length 24 cyclic code
[parmat,genmat] = cyclgen(24,genpoly);
genmat(:,1:14)
sum(genmat(:,1:14))
sum(genmat(:,1:14),2)
genmat(:,1:14)
codgen
%% Indices with better than 50% confidence on binary logit with fminunc minimization
[percents',C]
badPct=(percents < 0.5)';
goodPct=~badPct;
[percents(goodPct)',C(goodPct,:)]
sum(goodPct)
unique(C(goodPct,2))