function [v, n] = push(v, x)
   n = length(v) + length(x);
   if (size(v,1) == 1) || (isempty(v) && size(x,1)==1)
        v = [v(:);x(:)]';
   else
        v = [v(:);x(:)];
   end
end