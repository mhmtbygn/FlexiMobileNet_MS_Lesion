# FlexiMobileNet: nested evaluation of active and chronic MS lesion classification

MATLAB implementation of the FlexiMobileNet pipeline and of the validation protocols described in

> Tatli S, Macin G, Tasci I, Tasci B, Yildirim K, Gun MV, Baygin M, Algul FE, Dogan S, Tuncer T.
> *Classification of Active and Chronic Multiple Sclerosis Lesions Using Multi-Scale Transfer Learning.*
> Biomedicines (under review, manuscript biomedicines-4651816).

## Pipeline

| Step | Function | Manuscript |
|---|---|---|
| Threshold-based brain-area cropping (grey level > 20), resize to 240 × 240 | `preprocess_image.m` | Eqs. (12)–(14) |
| MobileNetV2 GAP features of the whole ROI and of 4, 9, 16, 25 and 36 non-overlapping patches (120, 80, 60, 48 and 40 px) | `extract_features.m` | Eqs. (15)–(18) |
| Min–max normalisation and CWNCA (NCA weights, cumulative threshold 0.99) | `cwnca_fit.m`, `cwnca_apply.m` | Eqs. (19)–(22) |
| tkNN: 4 distances × k = 1…10 × 3 weightings = 120 settings | `knn_grid.m`, `knn_predict.m` | Eq. (23) |
| IMV over the top 3…top 120 outcomes (118 voted) and greedy selection, `b = argmax(acc)` | `imv_select.m` | Eqs. (24)–(28) |
| Cross-scale IMV (5 scale outcomes + 3 voted) and greedy selection | `imv_select.m`, `apply_selection.m` | Eqs. (30)–(34) |

## Validation protocols (`run_nested.m`, `one_outer_fold.m`)

* **Nested leave-one-patient-out (`loso`).** Each outer fold holds out all images of one patient
  (30 outer folds). The remaining patients are assigned at random to `cfg.innerFolds = 5`
  patient-grouped inner folds (`rng(cfg.seed + outer fold)`, `cfg.seed = 2026`). In every inner
  fold the normalisation and CWNCA are refitted on the inner-training images, and the 120 tkNN
  settings predict the inner-validation images. The tkNN setting or IMV combination of each scale
  and the cross-scale fusion are chosen by their out-of-fold inner accuracy. The chosen pipeline is
  then refitted on all images of the outer-training patients and applied once to the held-out
  patient.
* **Nested image level (`image10`).** Outer stratified 10-fold split of the images (seed 2026) with
  the same inner selection on stratified image-level inner folds.

`main.m` runs the four-class task, the global MobileNetV2 baseline (same LOSO folds), Case 1
(sequence control), Case 2 (active vs. chronic), the FLAIR-only and T1-only active-vs-chronic tasks
under nested LOSO, and the four-class task under the nested image-level protocol.

## Statistics (`summarize_results.m`)

* Image-level Wilson intervals for accuracy (`wilson_ci.m`).
* **Patient-cluster bootstrap** percentile intervals (`cluster_bootstrap.m`, 10,000 resamples of
  patients with replacement) for accuracy, balanced accuracy, macro-F1, sensitivity and
  specificity, and for paired differences between two models.
* Exact McNemar test and the **clustered McNemar test** of Durkalski et al. (Stat Med 2003;22:2417–2428)
  for FlexiMobileNet vs. global MobileNetV2 and for every model vs. the all-chronic classifier
  (`paired_tests.m`).

## Requirements

* MATLAB R2023b or later
* Deep Learning Toolbox and *Deep Learning Toolbox Model for MobileNet-v2 Network*
* Statistics and Machine Learning Toolbox (`fscnca`, `fitcknn`, `cvpartition`)
* Image Processing Toolbox
* Optional: Parallel Computing Toolbox (`cfg.numWorkers > 0` runs the outer folds in parallel)

## Data

1. Download the anonymized images (Kaggle: `buraktaci/ms-lesion-mri-dataset`) and place them as
   `data/MS_Lesion/<class folder>/<image>`. Edit `cfg.classFolders` in `flexi_config.m` to match the
   folder names; their order defines the labels 1 active T1, 2 active FLAIR, 3 chronic T1, 4 chronic FLAIR.
2. Provide `data/image_patient_map.csv` with the columns `image_id` (`<class folder>/<file name>`)
   and `patient_code` (pseudonymous). Patient-disjoint protocols need this file. The authors' map
   is not included in this release.

## Running

```matlab
test_smoke      % about one minute, synthetic features, checks the nested logic and the outputs
main            % full analysis; CWNCA on 47,360-D vectors is the main cost (hours on a CPU)
```

## Outputs (`results/`)

| File | Content |
|---|---|
| `<protocol>_<task>_<model>_predictions.csv` | image_id, patient_code, outer_fold, true and predicted class, scale-wise predictions |
| `<protocol>_<task>_<model>_folds.csv` | role of every image in every outer fold (`test` or `inner_k`) |
| `<protocol>_<task>_<model>_selection.csv` | selected feature count, tkNN setting or IMV members and inner accuracy per scale and outer fold; scale 0 = cross-scale fusion |
| `<protocol>_<task>_<model>_confusion.csv` | confusion matrix (rows true, columns predicted) |
| `summary_metrics.csv` | metrics with Wilson and patient-cluster bootstrap intervals |
| `paired_tests.csv` | exact and clustered McNemar tests and cluster-bootstrap intervals of the accuracy differences |

Run records of the analyses reported in the manuscript are not part of this release; running
`main.m` on the data writes them.

## License

MIT (see `LICENSE`).
