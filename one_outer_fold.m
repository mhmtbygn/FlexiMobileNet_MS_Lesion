function f = one_outer_fold(o, Fs, y, patient, outer, protocol, grid, cfg)
%ONE_OUTER_FOLD Inner selection and outer refit for outer fold o.
rng(cfg.seed + o, 'twister');
te  = find(outer == o);
tr  = find(outer ~= o);
ytr = y(tr);
inner = inner_folds(ytr, patient(tr), protocol, cfg);
nS = numel(Fs);
m  = numel(grid);

% ---------------------------------------------------------------- inner selection
sels = cell(1, nS);
oofScale = zeros(numel(tr), nS);
for s = 1:nS
    Poof = zeros(numel(tr), m);
    for k = 1:max(inner)
        itr = inner ~= k;
        iva = inner == k;
        mdl = cwnca_fit(Fs{s}(tr(itr), :), ytr(itr), cfg);       % refitted in every inner fold
        Poof(iva, :) = knn_predict(cwnca_apply(mdl, Fs{s}(tr(itr), :)), ytr(itr), ...
                                   cwnca_apply(mdl, Fs{s}(tr(iva), :)), grid);
    end
    sels{s} = imv_select(Poof, ytr, cfg.imvMinTop);              % 120 + 118 candidates
    oofScale(:, s) = sels{s}.oof;
end
if nS >= cfg.imvMinTop
    fus = imv_select(oofScale, ytr, cfg.imvMinTop);               % 5 scale + 3 voted outcomes
else
    fus = imv_select(oofScale, ytr, nS + 1);                      % single outcomes only
end

% ---------------------------------------------------------------- outer refit and test
scalePred = zeros(numel(te), nS);
logRows = cell(nS + 1, 6);
for s = 1:nS
    mdl = cwnca_fit(Fs{s}(tr, :), ytr, cfg);
    Pte = knn_predict(cwnca_apply(mdl, Fs{s}(tr, :)), ytr, ...
                      cwnca_apply(mdl, Fs{s}(te, :)), grid, sels{s}.members);
    if strcmp(sels{s}.type, 'single')
        scalePred(:, s) = Pte(:, 1);
    else
        scalePred(:, s) = mode(Pte, 2);
    end
    logRows(s, :) = {o, s, numel(mdl.idx), sels{s}.type, mat2str(sels{s}.members), sels{s}.selectionAccuracy};
end
logRows(nS + 1, :) = {o, 0, NaN, fus.type, mat2str(fus.members), fus.selectionAccuracy};

f.te = te;
f.tr = tr;
f.inner = inner;
f.scalePred = scalePred;
f.pred = apply_selection(fus, scalePred);
f.log = logRows;
end

function inner = inner_folds(ytr, ptr, protocol, cfg)
%INNER_FOLDS Patient-grouped (LOSO) or stratified (image-level) inner folds.
K = cfg.innerFolds;
switch protocol
    case 'loso'
        [~, ~, pid] = unique(ptr);
        np = max(pid);
        K = min(K, np);
        fp = zeros(np, 1);
        fp(randperm(np)) = mod(0:np-1, K) + 1;   % random, balanced patient-to-fold assignment
        inner = fp(pid);
    otherwise
        c = cvpartition(ytr, 'KFold', K);
        inner = zeros(numel(ytr), 1);
        for k = 1:K
            inner(test(c, k)) = k;
        end
end
end
