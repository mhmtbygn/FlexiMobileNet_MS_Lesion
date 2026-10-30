%TEST_SMOKE Fast check of the nested pipeline on synthetic features.
%   Does not need MobileNetV2 or image data. Takes about one minute.
rng(1, 'twister');
cfg = flexi_config();
cfg.outDir = fullfile(tempdir, 'flexi_smoke');
cfg.innerFolds = 3; cfg.kValues = 1:3; cfg.distances = {'cityblock', 'euclidean'};
cfg.nBoot = 200;
nPat = 8; perPat = 12;
patient = strings(nPat * perPat, 1); label = zeros(nPat * perPat, 1);
for k = 1:nPat
    r = (k - 1) * perPat + (1:perPat);
    patient(r) = sprintf('P%02d', k);
    label(r) = repmat((1:4)', perPat / 4, 1);
end
F = cell(1, 3);
for s = 1:3
    F{s} = single(randn(numel(label), 40) + 0.8 * full(sparse(1:numel(label), label, 1, numel(label), 40)));
end
data.imageId = cellstr("img" + string(1:numel(label))');
data.label = label; data.patient = patient;
runs = struct('tag', {}, 'protocol', {}, 'task', {}, 'model', {}, 'T', {}, 'res', {});
plan = {'loso', 'fourclass', 'flexi'; 'loso', 'fourclass', 'mnv2_global'; 'loso', 'case2', 'flexi'; ...
        'image10', 'fourclass', 'flexi'};
cfg.outerImageFolds = 4;
for r = 1:size(plan, 1)
    [protocol, task, model] = plan{r, :};
    T = define_task(label, task);
    if strcmp(model, 'flexi'), Fs = cellfun(@(X) X(T.idx, :), F, 'UniformOutput', false);
    else, Fs = {F{1}(T.idx, 1:10)}; end
    res = run_nested(Fs, T.y, patient(T.idx), protocol, cfg);
    assert(numel(res.pred) == numel(T.y) && all(res.pred > 0), 'prediction vector incomplete');
    if strcmp(protocol, 'loso')
        for o = 1:max(res.outer)        % no test patient may appear in its own training rows
            testPat = unique(patient(T.idx(res.outer == o)));
            assert(numel(testPat) == 1 && all(res.innerAssign(res.outer == o, o) == 0));
        end
    end
    tag = sprintf('%s_%s_%s', protocol, task, model);
    export_run(res, T, data, tag, cfg);
    runs(end+1) = struct('tag', tag, 'protocol', protocol, 'task', task, 'model', model, ...
                         'T', T, 'res', res); %#ok<SAGROW>
    fprintf('%-28s accuracy %.2f%%\n', tag, 100 * mean(res.pred == T.y));
end
[S, P] = summarize_results(runs, data, cfg);
disp(P(:, {'task', 'model', 'comparator', 'p_exact_mcnemar', 'p_clustered_mcnemar'}));
fprintf('Smoke test passed. Outputs in %s\n', cfg.outDir);
