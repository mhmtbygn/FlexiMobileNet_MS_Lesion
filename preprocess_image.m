function J = preprocess_image(I, cfg)
%PREPROCESS_IMAGE Threshold-based brain-area cropping and resizing (Eqs. 12-14).
%   Rows and columns that contain at least one pixel brighter than
%   cfg.threshold (8-bit grey level) define the bounding box. This is a
%   bounding-box approximation, not skull stripping.
if size(I, 3) == 3
    I = rgb2gray(I);
end
G = im2uint8(I);
mask = G > cfg.threshold;
rows = find(any(mask, 2));
cols = find(any(mask, 1));
if ~isempty(rows) && ~isempty(cols)
    G = G(rows(1):rows(end), cols(1):cols(end));
end
J = imresize(G, [cfg.inputSize cfg.inputSize]);
end
