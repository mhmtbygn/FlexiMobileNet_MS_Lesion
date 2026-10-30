function P = knn_predict(Xtr, ytr, Xte, grid, which)
%KNN_PREDICT Predictions of the selected tkNN settings (columns of P).
if nargin < 5 || isempty(which), which = 1:numel(grid); end
P = zeros(size(Xte, 1), numel(which));
for j = 1:numel(which)
    g = grid(which(j));
    mdl = fitcknn(Xtr, ytr, 'NumNeighbors', g.k, 'Distance', g.distance, ...
                  'DistanceWeight', g.weight, 'Standardize', false);
    P(:, j) = predict(mdl, Xte);
end
end
