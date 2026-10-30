function F = extract_features(data, cfg)
%EXTRACT_FEATURES Global and patch-level MobileNetV2 GAP features (Eqs. 15-18).
%   F{s} is an N x 1280*(1 + P_s) single matrix for patch size cfg.patchSizes(s):
%   the global feature of the whole 240 x 240 ROI followed by the features of
%   its P_s non-overlapping patches in row-major order. Features are cached.
%   Requires the Deep Learning Toolbox Model for MobileNet-v2 Network.

if ~isfolder(cfg.featDir), mkdir(cfg.featDir); end
cacheFile = fullfile(cfg.featDir, 'mobilenetv2_features.mat');
if isfile(cacheFile)
    S = load(cacheFile, 'F', 'imageId', 'patchSizes');
    if isequal(S.imageId, data.imageId) && isequal(S.patchSizes, cfg.patchSizes)
        F = S.F;
        return
    end
end

net  = mobilenetv2;
inSz = net.Layers(1).InputSize(1:2);
nImg = numel(data.file);
nS   = numel(cfg.patchSizes);
F    = cell(1, nS);
for s = 1:nS
    np   = (cfg.inputSize / cfg.patchSizes(s))^2;
    F{s} = zeros(nImg, 1280 * (1 + np), 'single');
end
for i = 1:nImg
    J = preprocess_image(imread(data.file{i}), cfg);
    g = gap_features(net, {J}, inSz, cfg.gapLayer);                 % 1 x 1280
    for s = 1:nS
        p = cfg.patchSizes(s);
        blocks = {};
        for r = 1:p:cfg.inputSize
            for c = 1:p:cfg.inputSize
                blocks{end+1} = J(r:r+p-1, c:c+p-1);                  %#ok<AGROW>
            end
        end
        pf = gap_features(net, blocks, inSz, cfg.gapLayer);          % P_s x 1280
        F{s}(i, :) = [g, reshape(pf.', 1, [])];
    end
    if mod(i, 50) == 0, fprintf('features: %d/%d images\n', i, nImg); end
end
imageId = data.imageId; patchSizes = cfg.patchSizes; %#ok<NASGU>
save(cacheFile, 'F', 'imageId', 'patchSizes', '-v7.3');
end

function f = gap_features(net, imgs, inSz, layer)
X = zeros([inSz 3 numel(imgs)], 'single');
for k = 1:numel(imgs)
    X(:, :, :, k) = repmat(single(imresize(imgs{k}, inSz)), [1 1 3]);
end
f = activations(net, X, layer, 'OutputAs', 'rows');
end
