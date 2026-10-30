function export_run(res, T, data, tag, cfg)
%EXPORT_RUN Write the records of one nested run as CSV files.
%   <tag>_predictions.csv  image_id, patient_code, outer_fold, true_class, predicted_class,
%                          and the scale-wise predictions
%   <tag>_folds.csv        outer_fold, image_id, patient_code, role ('test' or 'inner_k')
%   <tag>_selection.csv    outer_fold, scale (0 = cross-scale fusion), n_selected_features,
%                          selection_type, members (tkNN settings or ranked outcomes),
%                          selection_accuracy (inner folds)
if ~isfolder(cfg.outDir), mkdir(cfg.outDir); end
ids = string(data.imageId(T.idx));
pat = data.patient(T.idx);
P = table(ids, pat, res.outer, T.y(:), res.pred, ...
    'VariableNames', {'image_id', 'patient_code', 'outer_fold', 'true_class', 'predicted_class'});
for s = 1:size(res.scalePred, 2)
    P.(sprintf('scale%d_prediction', s)) = res.scalePred(:, s);
end
writetable(P, fullfile(cfg.outDir, [tag '_predictions.csv']));

nOuter = size(res.innerAssign, 2);
n = numel(ids);
role = strings(n * nOuter, 1);
of = zeros(n * nOuter, 1);
for o = 1:nOuter
    r = (o - 1) * n + (1:n);
    of(r) = o;
    a = res.innerAssign(:, o);
    role(r) = "inner_" + string(a);
    role(r(a == 0)) = "test";
end
Fd = table(of, repmat(ids, nOuter, 1), repmat(pat, nOuter, 1), role, ...
    'VariableNames', {'outer_fold', 'image_id', 'patient_code', 'role'});
writetable(Fd, fullfile(cfg.outDir, [tag '_folds.csv']));

L = cell2table(res.log, 'VariableNames', {'outer_fold', 'scale', 'n_selected_features', ...
    'selection_type', 'members', 'selection_accuracy'});
writetable(L, fullfile(cfg.outDir, [tag '_selection.csv']));
end
