function model = cwnca_fit(X, y, cfg)
%CWNCA_FIT Min-max normalisation and cumulative-weighted NCA selection (Eqs. 19-22).
%   All parameters are learned from the rows passed in (training rows only).
%   The features are ranked by their NCA weights, and the smallest set whose
%   cumulative weight reaches cfg.cumThreshold is kept.
mn = min(X, [], 1);
rg = max(X, [], 1) - mn;
rg(rg == 0) = 1;
Xn  = double((X - mn) ./ rg);
nca = fscnca(Xn, y, 'Standardize', false);
w   = nca.FeatureWeights;
[ws, idx] = sort(w, 'descend');
cw  = cumsum(ws) / max(sum(ws), eps);
nSel = find(cw >= cfg.cumThreshold, 1, 'first');
if isempty(nSel), nSel = numel(idx); end
model.mn  = mn;
model.rg  = rg;
model.idx = idx(1:nSel);
end
