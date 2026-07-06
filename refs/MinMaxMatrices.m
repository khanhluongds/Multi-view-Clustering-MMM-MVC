%input a cell of different matrices
function [AMinMax] = MinMaxMatrices(A, m)
t= 1;
for v = 1:m
    for s = 1:m
        if s~=v
            Atemp{t,1} = max(A{v,1}, A{s,1});
            t = t+1;
        end
    end
end
for t = 1:length(Atemp)
    AMinMax(:,:,t) = Atemp{t,1};
end
AMinMax = min(AMinMax,[],3);