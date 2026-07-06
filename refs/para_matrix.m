function P = para_matrix(p1, p2, p3, p4, ncols)
%PARA_MATRIX Create a grid matrix following the MMM experiment convention.
%
% The first four columns contain all combinations of p1, p2, p3, p4. The
% remaining columns are initialized to zero for metrics and runtime.

if nargin < 5
    ncols = 17;
end
nrows = numel(p1) * numel(p2) * numel(p3) * numel(p4);
P = zeros(nrows, ncols);
idx = 0;
for i = 1:numel(p1)
    for j = 1:numel(p2)
        for k = 1:numel(p3)
            for l = 1:numel(p4)
                idx = idx + 1;
                P(idx, 1:4) = [p1(i), p2(j), p3(k), p4(l)];
            end
        end
    end
end
end
