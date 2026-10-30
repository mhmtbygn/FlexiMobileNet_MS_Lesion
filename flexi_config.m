function cfg = flexi_config()
%FLEXI_CONFIG Settings of the FlexiMobileNet pipeline and of the validation protocols.
%   Edit the data paths and class folder names before running main.m.

% ---------------------------------------------------------------- data
cfg.dataRoot     = fullfile(pwd, 'data', 'MS_Lesion');          % one sub-folder per class
cfg.classFolders = {'Active_T1', 'Active_FLAIR', 'Chronic_T1', 'Chronic_FLAIR'};
cfg.classNames   = {'Active T1', 'Active FLAIR', 'Chronic T1', 'Chronic FLAIR'};
cfg.patientMap   = fullfile(pwd, 'data', 'image_patient_map.csv');  % image_id,patient_code
cfg.featDir      = fullfile(pwd, 'features');
cfg.outDir       = fullfile(pwd, 'results');

% ---------------------------------------------------------------- preprocessing and features
cfg.threshold  = 20;                    % 8-bit grey level for brain-area cropping
cfg.inputSize  = 240;                   % cropped ROI is resized to 240 x 240
cfg.patchSizes = [120 80 60 48 40];     % 4, 9, 16, 25 and 36 non-overlapping patches
cfg.gapLayer   = 'global_average_pooling2d_1';   % MobileNetV2 GAP layer (1280-D)

% ---------------------------------------------------------------- CWNCA
cfg.cumThreshold = 0.99;                % cumulative NCA weight retained

% ---------------------------------------------------------------- tkNN and IMV
cfg.distances = {'cityblock', 'euclidean', 'cosine', 'spearman'};
cfg.kValues   = 1:10;
cfg.weights   = {'equal', 'inverse', 'squaredinverse'};   % 4 x 10 x 3 = 120 settings
cfg.imvMinTop = 3;                      % IMV combines the top 3 ... top m outcomes

% ---------------------------------------------------------------- validation
cfg.innerFolds      = 5;                % patient-grouped (LOSO) or stratified (image level)
cfg.outerImageFolds = 10;               % outer folds of the nested image-level protocol
cfg.seed            = 2026;             % rng(seed + outer fold) for every inner split
cfg.numWorkers      = 0;                % >0 runs the outer folds in parallel (parfor)

% ---------------------------------------------------------------- statistics
cfg.nBoot = 10000;                      % bootstrap resamples (patients or images)
end
