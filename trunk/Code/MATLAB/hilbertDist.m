function [dist] = hilbertDist(x,y)
%HILBERTDIST calculates the Hilbertdistance between the vectors X and Y
%  
k = max(x./y);
l = min(x./y);

dist = log(k./l);

end

