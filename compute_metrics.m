function M = compute_metrics(y, p, nC)
%COMPUTE_METRICS Confusion matrix and the metrics reported in the manuscript.
%   Rows of M.cm are true classes and columns predicted classes. For binary
%   tasks class 1 (active lesion) is the positive class.
y = y(:); p = p(:);
cm = accumarray([y p], 1, [nC nC]);
tp = diag(cm);
rec  = tp ./ max(sum(cm, 2), 1);
prec = tp ./ max(sum(cm, 1)', 1);
f1   = 2 * tp ./ max(sum(cm, 2) + sum(cm, 1)', 1);
n  = sum(cm(:));
po = sum(tp) / n;
pe = sum(sum(cm, 1)' .* sum(cm, 2)) / n^2;
M.cm = cm;
M.accuracy = po;
M.balancedAccuracy = mean(rec);
M.macroPrecision = mean(prec);
M.macroF1 = mean(f1);
M.harmonicF1 = 2 * M.macroPrecision * M.balancedAccuracy / max(M.macroPrecision + M.balancedAccuracy, eps);
M.kappa = (po - pe) / max(1 - pe, eps);
M.recall = rec'; M.precision = prec'; M.f1 = f1';
if nC == 2
    M.sensitivity = rec(1);
    M.specificity = rec(2);
end
end
