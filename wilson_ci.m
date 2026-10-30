function ci = wilson_ci(k, n, z)
%WILSON_CI Wilson score interval for k successes out of n (independent units).
if nargin < 3, z = 1.959963984540054; end
p = k / n;
d = 1 + z^2 / n;
c = (p + z^2 / (2 * n)) / d;
h = z * sqrt(p * (1 - p) / n + z^2 / (4 * n^2)) / d;
ci = [c - h, c + h];
end
