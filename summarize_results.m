function [S, P] = summarize_results(runs, data, cfg)
%SUMMARIZE_RESULTS Metrics with image-level and patient-cluster intervals, and paired tests.
%   runs: struct array with fields tag, protocol, task, model, T, res.
%   S (summary_metrics.csv): one row per run and metric. Accuracy carries an
%     image-level Wilson interval and a patient-cluster bootstrap interval.
%   P (paired_tests.csv): FlexiMobileNet vs. global MobileNetV2 (same folds)
%     and every model vs. the all-chronic reference classifier: exact
%     McNemar, clustered McNemar and patient-cluster bootstrap intervals of
%     the differences.
S = table(); P = table();
for r = 1:numel(runs)
    R = runs(r); T = R.T;
    y = T.y(:); p = R.res.pred; nC = numel(T.names);
    pat = data.patient(T.idx);
    M = compute_metrics(y, p, nC);
    wil = wilson_ci(sum(p == y), numel(y));
    hasPat = ~all(strlength(pat) == 0);
    if hasPat, CB = cluster_bootstrap(y, p, [], pat, nC, cfg); end
    names = {'accuracy', 'balancedAccuracy', 'macroF1', 'macroPrecision', 'harmonicF1', 'kappa'};
    if nC == 2, names = [names, {'sensitivity', 'specificity'}]; end
    for j = 1:numel(names)
        v = M.(names{j});
        lo = NaN; hi = NaN; lo2 = NaN; hi2 = NaN;
        if strcmp(names{j}, 'accuracy'), lo = wil(1); hi = wil(2); end
        if hasPat && isfield(CB, names{j}), lo2 = CB.(names{j})(1); hi2 = CB.(names{j})(2); end
        S = [S; table(string(R.protocol), string(R.task), string(R.model), string(names{j}), v, ...
            lo, hi, lo2, hi2, 'VariableNames', {'protocol', 'task', 'model', 'metric', 'value', ...
            'wilson_low', 'wilson_high', 'cluster_boot_low', 'cluster_boot_high'})]; %#ok<AGROW>
    end
    for c = 1:nC
        S = [S; table(string(R.protocol), string(R.task), string(R.model), ...
            "recall_" + string(T.names{c}), M.recall(c), NaN, NaN, NaN, NaN, 'VariableNames', ...
            {'protocol', 'task', 'model', 'metric', 'value', 'wilson_low', 'wilson_high', ...
            'cluster_boot_low', 'cluster_boot_high'})]; %#ok<AGROW>
    end
    writematrix(M.cm, fullfile(cfg.outDir, [R.tag '_confusion.csv']));

    % paired comparisons
    comps = {};
    if ~isempty(T.baseline), comps(end+1, :) = {'all_chronic', T.baseline}; end
    if strcmp(R.model, 'flexi')
        k = find(strcmp({runs.protocol}, R.protocol) & strcmp({runs.task}, R.task) & ...
                 strcmp({runs.model}, 'mnv2_global'), 1);
        if ~isempty(k), comps(end+1, :) = {'mnv2_global', runs(k).res.pred}; end
    end
    for c = 1:size(comps, 1)
        pr = paired_tests(y, p, comps{c, 2}, pat);
        dlo = NaN; dhi = NaN;
        if hasPat
            CBd = cluster_bootstrap(y, p, comps{c, 2}, pat, nC, cfg);
            dlo = CBd.diff_accuracy(1); dhi = CBd.diff_accuracy(2);
        end
        Mb = compute_metrics(y, comps{c, 2}, nC);
        P = [P; table(string(R.protocol), string(R.task), string(R.model), string(comps{c, 1}), ...
            M.accuracy - Mb.accuracy, pr.b, pr.c, pr.pExact, pr.chi2Clustered, pr.pClustered, dlo, dhi, ...
            'VariableNames', {'protocol', 'task', 'model', 'comparator', 'accuracy_difference', ...
            'b_only_model_correct', 'c_only_comparator_correct', 'p_exact_mcnemar', ...
            'chi2_clustered_mcnemar', 'p_clustered_mcnemar', 'cluster_boot_diff_low', ...
            'cluster_boot_diff_high'})]; %#ok<AGROW>
    end
end
writetable(S, fullfile(cfg.outDir, 'summary_metrics.csv'));
writetable(P, fullfile(cfg.outDir, 'paired_tests.csv'));
end
