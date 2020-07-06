function [div] = kullbackLeiblerNat(P, Q)
tmp = P./(Q+eps);
div = sum(P.*log(tmp));
end

