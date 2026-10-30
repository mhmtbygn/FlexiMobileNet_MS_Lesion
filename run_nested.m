function res = run_nested(Fs, y, patient, protocol, cfg)
%RUN_NESTED Nested cross-validation of the complete FlexiMobileNet pipeline.
%   Fs        1 x S cell of feature matrices (one per patch scale; S = 1 for
%             the global MobileNetV2 baseline)
%   y         n x 1 class labels
%   patient   n x 1 string patient codes
%   protocol  'loso'    outer leave-one-patient-out, inner patient-grouped folds
%             'image10' outer stratified 10-fold, inner stratified folds (image level)
%
%   In every outer fold, normalisation, CWNCA, the tkNN setting, the IMV
%   combination and the cross-scale fusion are selected on inner folds of the
%   outer training rows only. The selected pipeline is then refitted on all
%   outer training rows and applied once to the held-out rows.

y = y(:);
n = numel(y);
grid = knn_grid(cfg);
switch protocol
    case 'loso'
        if all(strlength(patient) == 0)
            error('run_nested:nopatient', 'LOSO needs patient codes (image_patient_map.csv).');
        end
        [~, ~, outer] = unique(patient);
        nOuter = max(outer);
    case 'image10'
        rng(cfg.seed, 'twister');
        cvp = cvpartition(y, 'KFold', cfg.outerImageFolds);
        outer = zeros(n, 1);
        for k = 1:cvp.NumTestSets
            outer(test(cvp, k)) = k;
        end
        nOuter = cvp.NumTestSets;
    otherwise
        error('run_nested:protocol', 'Unknown protocol %s.', protocol);
end

foldOut = cell(nOuter, 1);
parfor (o = 1:nOuter, cfg.numWorkers)
    foldOut{o} = one_outer_fold(o, Fs, y, patient, outer, protocol, grid, cfg);
end

res.protocol    = protocol;
res.outer       = outer;
res.pred        = zeros(n, 1);
res.scalePred   = zeros(n, numel(Fs));
res.innerAssign = zeros(n, nOuter);      % inner fold of each training row; 0 = outer test
res.log         = cell(0, 6);
for o = 1:nOuter
    f = foldOut{o};
    res.pred(f.te)         = f.pred;
    res.scalePred(f.te, :) = f.scalePred;
    res.innerAssign(f.tr, o) = f.inner;
    res.log = [res.log; f.log];
end
end
