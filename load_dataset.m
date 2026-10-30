function data = load_dataset(cfg)
%LOAD_DATASET List the images of the four class folders and attach patient codes.
%   data.file      full path of each image
%   data.imageId   '<class folder>/<file name>' (key of image_patient_map.csv)
%   data.label     1 active T1, 2 active FLAIR, 3 chronic T1, 4 chronic FLAIR
%   data.patient   pseudonymous patient code (string)

exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'};
file = {}; imageId = {}; label = [];
for c = 1:numel(cfg.classFolders)
    d = dir(fullfile(cfg.dataRoot, cfg.classFolders{c}));
    d = d(~[d.isdir]);
    for i = 1:numel(d)
        [~, ~, e] = fileparts(d(i).name);
        if ~any(strcmpi(e, exts)), continue; end
        file{end+1, 1}    = fullfile(d(i).folder, d(i).name);          %#ok<AGROW>
        imageId{end+1, 1} = [cfg.classFolders{c} '/' d(i).name];      %#ok<AGROW>
        label(end+1, 1)   = c;                                         %#ok<AGROW>
    end
end
if isempty(file)
    error('load_dataset:empty', 'No images found under %s.', cfg.dataRoot);
end
data.file = file; data.imageId = imageId; data.label = label;

if isfile(cfg.patientMap)
    T = readtable(cfg.patientMap, 'TextType', 'string', 'Delimiter', ',');
    [tf, loc] = ismember(string(imageId), T.image_id);
    if ~all(tf)
        error('load_dataset:map', '%d images have no patient code in %s.', sum(~tf), cfg.patientMap);
    end
    data.patient = string(T.patient_code(loc));
else
    warning('load_dataset:nomap', ['No image-patient map found (%s). ' ...
        'Patient-disjoint protocols cannot be run.'], cfg.patientMap);
    data.patient = strings(numel(file), 1);
end
fprintf('%d images, %d patients, class counts: %s\n', numel(file), ...
    numel(unique(data.patient)), mat2str(accumarray(label, 1)'));
end
