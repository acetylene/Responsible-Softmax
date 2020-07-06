function [div] = kullbackLeibler(P, Q)
tmp = P./(Q+eps);
div = sum(P.*log2(tmp));
end

