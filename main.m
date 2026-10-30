%MAIN FlexiMobileNet: nested evaluation of MS lesion activity/sequence classification.
%   1. Place the images in cfg.dataRoot (one folder per class, see flexi_config.m).
%   2. Place data/image_patient_map.csv (columns image_id, patient_code).
%   3. Run this script. Records and statistics are written to results/.
clear; clc;
cfg = flexi_config();
rng(cfg.seed, 'twister');
if ~isfolder(cfg.outDir), mkdir(cfg.outDir); end

data = load_dataset(cfg);
F = extract_features(data, cfg);            % cached in features/

% protocol, task, model
plan = { 'loso',    'fourclass',  'flexi'
         'loso',    'fourclass',  'mnv2_global'
         'loso',    'case1',      'flexi'
         'loso',    'case2',      'flexi'
         'loso',    'flair_only', 'flexi'
         'loso',    't1_only',    'flexi'
         'image10', 'fourclass',  'flexi' };

runs = struct('tag', {}, 'protocol', {}, 'task', {}, 'model', {}, 'T', {}, 'res', {});
for r = 1:size(plan, 1)
    [protocol, task, model] = plan{r, :};
    T = define_task(data.label, task);
    if strcmp(model, 'flexi')
        Fs = cellfun(@(X) X(T.idx, :), F, 'UniformOutput', false);
    else
        Fs = {F{1}(T.idx, 1:1280)};          % global MobileNetV2 GAP features only
    end
    tag = sprintf('%s_%s_%s', protocol, task, model);
    fprintf('\n=== %s ===\n', tag);
    tic;
    res = run_nested(Fs, T.y, data.patient(T.idx), protocol, cfg);
    fprintf('%s: accuracy %.2f%% (%.1f min)\n', tag, 100 * mean(res.pred == T.y), toc / 60);
    export_run(res, T, data, tag, cfg);
    save(fullfile(cfg.outDir, [tag '.mat']), 'res', 'T', 'cfg');
    runs(end+1) = struct('tag', tag, 'protocol', protocol, 'task', task, 'model', model, ...
                         'T', T, 'res', res); %#ok<SAGROW>
end
summarize_results(runs, data, cfg);
fprintf('\nDone. See %s\n', cfg.outDir);
