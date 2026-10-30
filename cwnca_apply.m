function Xs = cwnca_apply(model, X)
%CWNCA_APPLY Apply a fitted normalisation and feature selection to new rows.
Xs = double((X - model.mn) ./ model.rg);
Xs = Xs(:, model.idx);
end
