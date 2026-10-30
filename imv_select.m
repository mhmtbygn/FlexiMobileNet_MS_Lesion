function sel = imv_select(P, y, minTop)
%IMV_SELECT Iterative majority voting and greedy selection (Eqs. 24-28).
%   P      n x m candidate predictions on the selection rows
%   y      n x 1 true labels of the same rows
%   The m outcomes are ranked by accuracy, the top q outcomes are combined by
%   majority vote for q = minTop..m, and the candidate with the maximum
%   accuracy among the m + (m - minTop + 1) outcomes is kept:
%   b = argmax(acc). Ties keep the first candidate.
y = y(:);
m = size(P, 2);
acc = mean(P == y, 1);
[~, order] = sort(acc, 'descend');
qs = minTop:m;
V = zeros(size(P, 1), numel(qs));
for j = 1:numel(qs)
    V(:, j) = mode(P(:, order(1:qs(j))), 2);
end
allAcc = [acc, mean(V == y, 1)];
[best, b] = max(allAcc);
if b <= m
    sel.type = 'single'; sel.members = b; sel.oof = P(:, b);
else
    sel.type = 'imv'; sel.members = order(1:qs(b - m)); sel.oof = V(:, b - m);
end
sel.selectionAccuracy = best;
end
