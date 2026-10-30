function S = paired_tests(y, pA, pB, patient)
%PAIRED_TESTS Exact McNemar test and clustered McNemar test of overall correctness.
%   cA/cB mark the images classified correctly by models A and B.
%   Exact test: binomial test on the discordant images (independent images).
%   Clustered test (Durkalski et al., Stat Med 2003;22:2417-2428): with
%   d_k = b_k - c_k the discordance difference of patient k,
%   X2 = (sum d_k)^2 / sum d_k^2 ~ chi-square(1), which allows correlated
%   images within a patient.
cA = pA(:) == y(:);
cB = pB(:) == y(:);
b = sum(cA & ~cB);
c = sum(~cA & cB);
S.b = b; S.c = c;
if b + c == 0
    S.pExact = 1;
else
    S.pExact = min(1, 2 * binocdf(min(b, c), b + c, 0.5));
end
[~, ~, pid] = unique(patient);
bk = accumarray(pid, double(cA & ~cB));
ck = accumarray(pid, double(~cA & cB));
d = bk - ck;
if sum(d.^2) == 0
    S.chi2Clustered = 0; S.pClustered = 1;
else
    S.chi2Clustered = sum(d)^2 / sum(d.^2);
    S.pClustered = 1 - chi2cdf(S.chi2Clustered, 1);
end
end
