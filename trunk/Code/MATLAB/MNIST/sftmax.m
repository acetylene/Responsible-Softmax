function [y] = sftmax(x)
        y = exp(x)./sum(exp(x));
end