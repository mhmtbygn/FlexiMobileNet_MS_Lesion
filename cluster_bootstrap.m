function B = cluster_bootstrap(y, pA, pB, patient, nC, cfg)
%CLUSTER_BOOTSTRAP Patient-cluster bootstrap percentile intervals.
%   Patients are resampled with replacement and all images of a drawn patient
%   enter the replicate, so within-patient correlation is preserved. If pB is
%   given, the paired differences A - B are bootstrapped on the same
%   replicates. Returns 95% percentile intervals for accuracy, balanced
%   accuracy, macro-F1 (and sensitivity/specificity for binary tasks).
rng(cfg.seed, 'twister');
[~, ~, pid] = unique(patient);
nP = max(pid);
rowsOf = accumarray(pid, (1:numel(pid))', [], @(r) {r});
names = {'accuracy', 'balancedAccuracy', 'macroF1'};
if nC == 2, names = [names, {'sensitivity', 'specificity'}]; end
vals = nan(cfg.nBoot, numel(names), 1 + ~isempty(pB));
for b = 1:cfg.nBoot
    draw = randi(nP, nP, 1);
    r = vertcat(rowsOf{draw});
    MA = compute_metrics(y(r), pA(r), nC);
    for j = 1:numel(names), vals(b, j, 1) = MA.(names{j}); end
    if ~isempty(pB)
        MB = compute_metrics(y(r), pB(r), nC);
        for j = 1:numel(names), vals(b, j, 2) = MA.(names{j}) - MB.(names{j}); end
    end
end
for j = 1:numel(names)
    B.(names{j}) = prctile(vals(:, j, 1), [2.5 97.5]);
    if ~isempty(pB)
        B.(['diff_' names{j}]) = prctile(vals(:, j, 2), [2.5 97.5]);
    end
end
B.nPatients = nP;
end
