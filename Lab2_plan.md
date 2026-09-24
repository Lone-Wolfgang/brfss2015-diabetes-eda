Reference Wolfgang's Lab1:
/Users/jwkle/diabetes/Lab1/WolfgangKlein.ipynb

Notebook: /Users/jwkle/diabetes/Lab2/WolfgangKlein.ipynb

# Lab 2

Status: logistic regression is built. First kernel screen is done. The wider kernel screen is written in the notebook (§4) but not run yet. Wolfgang runs it.

## Intro

Briefly introduce the problem. In Lab 1, we introduced a dataset that associates health, demographics, and socioeconomics with a positive diabetes diagnosis. Simply drop a link, and reference Lab 1 for a more detailed review.

The goal in this analysis is to develop a pair of predictive models.

## Models

In this lab, we will be training a pair of model families: Logistic Regression and Support Vector Machines. Do a simple introduction. Fitting a logistic regression model is trivial for modern machinery. If the assumptions of logistic regression fit, then the resulting parameters are useful for interpretation. If the classes are linearly separable, then it should perform just as well as any other model, and there are opportunities to represent non-linearities through interactions and polynomial terms.

SVMs are non-parametric models. If the priority is prediction over interpretation, then the pathway to the final model may be shorter. An SVM projects training points into a vector space and learns boundaries that define the classes. One of the primary hyperparameters is the kernel, and there are three popular options: linear, polynomial, and Radial Basis Function (RBF). A linear kernel trains quickly, and if the data is linearly separable, then it should be a top performer. Polynomial draws more complex, curved boundaries. It trains a bit slower, but wall-clock time is still reasonable. A big advantage is that it can represent interactions and polynomial terms without explicit feature engineering. RBF is capable of representing complex non-linearities, but training time is substantial. If some parts of the data are not linearly separable, then RBF may yield a big boost in performance.

In Lab 1, the EDA supported that a logistic model is a fair choice, and a preliminary logistic model returned an AUC of 0.814. The expectation is that logistic regression will perform comparably to the SVM models. RBF takes a considerable amount of time to train. Before fitting on the full dataset, experiments will be run on a subset. If there is a significant performance boost, then we will run more expensive trials.

## Metrics

The purpose of the model is to use surface features to identify those at risk for diabetes. Positive cases will be referred to a doctor for screening. Diabetes is a devastating disease, and the prognosis improves substantially with early detection. On the other hand, the process for definitive detection is expensive, time-consuming, and uncomfortable for the patient.

The traditional metrics, recall and precision, will be used to compare models. In this case, high recall represents more lives saved, and precision represents better use of time and money. Both are important, but recall is the most important metric. Thus, models will be compared on the basis of F2, which is a harmonic mean of precision and recall that gives more weight to recall.

AUC is used only inside the SVM hyperparameter search (see below), because it does not depend on a threshold. F2 is still the metric that ranks the final models.

## Feature Considerations

In Wolfgang's Lab 1, Section 11, it was explained that a cholesterol check is associated with higher prevalence, which is an unnatural association. Withhold `CholCheck` from both the train and the test set.

Lab 1 discovered a ragged tail for outlying high BMIs. Winsorize BMI at 70.

Allow quadratic terms for Age and Income. Age and Income are centered before squaring.

All models use the same 22 features. The SVMs also standardize the features (`StandardScaler` inside a `Pipeline`, so it is fit on training folds only). Logistic regression stays unscaled so its coefficients stay in natural units.

## Splits

Do a 70/15/15 split, stratified on the target (seed 0). 70%, `train`, is used for fitting. For the SVM, it is also used for tuning hyperparameters through 5-fold internal CV. The first 15%, `validation`, is used for threshold tuning. The final 15%, `test`, is used as an unbiased evaluation at the selected threshold.

## Logistic Regression (done)

Fit on `train` with statsmodels, and print the parameter table with coefficients, odds ratios, 95% confidence intervals, and p-values.

## SVM

### Workflow

The notebook and the full search are separate.

- **Notebook** (`Lab2/WolfgangKlein.ipynb`): a record of how the training budget was set. It times fits, fits a cost curve, and picks the training size. It does **not** run the full search. It loads the search's saved results to draw the parallel coordinates plot and fill in the results table.
- **Script** (`scripts/svm_search.py`): runs the full grid search and final refits overnight. It uses the same features, split, and seed as the notebook. It writes to `Lab2/cache/`:
  - `svm_cv_results.csv`: one row per configuration, with mean/std CV AUC and mean fit time
  - `svm_best.json`: best configuration per kernel, training size, wall-clock totals
  - `svm_linear.joblib`, `svm_poly.joblib`: refit final models
  - a timestamped log, so a run that dies overnight can be diagnosed
- **Checkpointing (decided).** The script loops over the grid by hand instead of using `GridSearchCV`. It appends each finished configuration to the log as soon as it completes, and on restart it skips configurations already in the log. A crash overnight loses at most the fits in progress.

### Kernel screen

**Question:** is any kernel worth more than logistic regression on this data? This decides which kernels go into the overnight search.

**Setup (same for every trial).** `class_weight="balanced"` is fixed for every model here and in the overnight search (decided). Fit on a stratified 50,000-row sample of `train` (seed 0). Score AUC with `decision_function` on the other 127,576 `train` rows. `validation` and `test` are not touched. Features are standardized in a `Pipeline`.

**First screen (done, during planning).** sklearn defaults gave poly 0.735, RBF 0.708, and sigmoid 0.585, against 0.8165 for logistic regression. Most of that gap was class imbalance, not the kernels:

| model | AUC | fit s | score s |
|---|---:|---:|---:|
| RBF `gamma=0.01`, balanced | 0.8173 | 31 | 101 |
| poly d=2 `coef0=1`, balanced | 0.8170 | 33 | 34 |
| logistic, balanced | 0.8167 | 0.03 | 0.01 |
| linear SVM, balanced | 0.8163 | 0.05 | 0.01 |
| sigmoid `gamma=0.001`, balanced | 0.8155 | 27 | 42 |
| RBF default `gamma`, balanced | 0.8009 | 34 | 100 |
| poly d=3 `coef0=1`, balanced | 0.7956 | 45 | 33 |
| sigmoid default `gamma`, balanced | 0.6804 | 37 | 41 |

All SVMs used `C=1`. Only a few `gamma` values were tried, so this screen is too narrow to rule a kernel out.

**Second screen: slightly wider (next step).** Budget: **about 30 minutes** wall-clock.

| model | grid | trials |
|---|---|---:|
| logistic | baseline, no grid | 1 |
| linear (`LinearSVC`) | `C` ∈ {0.01, 0.1, 1, 10} | 4 |
| poly, degree 2 | `C` ∈ {0.1, 1, 10} x `coef0` ∈ {0.5, 1, 2}, `gamma="scale"` | 9 |
| poly, degree 3 | `C` ∈ {0.1, 1} x `coef0` ∈ {1, 2}, `gamma="scale"` | 4 |
| RBF | `C` ∈ {0.1, 1, 10} x `gamma` ∈ {0.003, 0.01, 0.03} | 9 |
| sigmoid | `C` ∈ {0.1, 1, 10} x `gamma` ∈ {0.0003, 0.001, 0.003}, `coef0=0` | 9 |

36 trials in all, 31 of them `SVC`.

Notes on the grid:

- For the poly kernel, `gamma` and `coef0` only matter through their ratio, and the overall scale is absorbed by `C`. So `gamma` stays fixed and only `coef0` varies.
- Degree 3 did worse than degree 2 in the first screen. It gets a small check (4 trials), not a full grid.
- The RBF and sigmoid `gamma` values bracket the best value from the first screen (0.01 and 0.001).

**Time estimate.** The first screen ran 8 trials in parallel in about 2.5 minutes, mostly scoring time. Running about 12 trials at a time on 14 cores, 31 `SVC` trials is about 3 waves. `C=10` fits are the slowest: at 40k rows they took about 7x longer than `C=1`. Estimate: 20-30 minutes.

**How it runs.**

- Runs in the notebook (§4 of `Lab2/WolfgangKlein.ipynb`), not a script. It uses `joblib.Parallel(n_jobs=12)`. Progress prints one line per finished trial: elapsed minutes, trial, AUC, fit and score seconds.
- Slowest first: trials are ordered `C=10` first, then by kernel cost (RBF, poly, sigmoid, linear), so the long jobs don't leave a tail at the end.
- Each trial is saved as a JSON line in `Lab2/cache/kernel_screen.jsonl` when it finishes. The line holds model, kernel, `C`, `gamma`, `coef0`, `degree`, AUC, fit s, score s, and support vector count. A rerun skips trials already in the file.
- The script prints the elapsed wall-clock time. If a wave is running well past the estimate, stop it. Finished trials are kept.

**Output.** The notebook reads `kernel_screen.jsonl` and shows the same table as above, sorted by AUC, with the hyperparameters as columns. Below it, a short table with the best trial per kernel and its AUC gap to logistic regression.

**Decision rule (decided).** A non-linear kernel goes into the overnight search only if its best trial beats logistic regression by **at least 0.005 AUC**. The linear SVM is cheap, so it is always kept. Otherwise it is recorded as screened out, with this table as the reason. The best region of any surviving kernel sets its overnight grid.

**Caveat.** Hyperparameters are picked on the same rows they are scored on, so the best AUCs are slightly optimistic. That is acceptable for a screen. The final models are still judged on the untouched `validation` and `test` sets.

### Training budget

**Goal:** the full search plus the final refits finish in one overnight run. **Budget: 8 hours** wall-clock on this machine (14 cores).

**Timing cell.** Time single fits of the poly SVM at increasing training sizes (e.g. 5k, 10k, 20k, 40k rows, stratified from `train`). Time two configurations:

- the cheapest (`degree=2, C=0.1`)
- the most expensive (`degree=3, C=10`)

Record wall-clock time with `time.perf_counter()` for each `fit`, and also for `decision_function` on the full validation set. Scoring cost grows with the number of support vectors, so it is not free. Show the timings as a table and a log-log plot of fit time against rows.

**Cost curve.** Fit `time = a * n^b` to each configuration's timings (a straight line on the log-log plot). Extrapolate to:

- one fit on the full `train` set (177,576 rows)
- the full search at size `n`: 12 poly configurations x 5 folds, each on `0.8 n` rows. Estimate each configuration from the cheap/expensive curves, conservatively using the expensive curve for all `C=10` fits. The linear search is negligible (a `LinearSVC` fit on all of `train` takes about 0.1 s).

**Decision.** Pick the largest `n` where the estimated search + refit fits in 8 hours with about a 2x safety margin, since `n_jobs=-1` runs folds in parallel but big fits can compete for memory. Print the chosen `n` and the projected wall-clock time. After the overnight run, the notebook compares the projection with the real total from `svm_best.json`.

**Preliminary probe** (single fits, `degree=3`, `coef0=1`, scaled features, run once during planning):

| rows | `C=1` | `C=10` |
|---:|---:|---:|
| 10,000 | 0.9 s | 3.2 s |
| 20,000 | 3.8 s | 24.7 s |
| 40,000 | 24.0 s | 175.9 s |

Doubling the rows costs about 6-7x, so fit time grows roughly as `n^2.7`. A rough extrapolation to all 177,576 training rows: ~20 min for `C=1`, ~2.5+ hours for `C=10`, per fit. A full 5-fold search on all of `train` will not fit overnight. A single final refit on all of `train` probably will. The timing cell makes these numbers official.

Both kernels are searched on the same subset of size `n`, so their CV AUCs are directly comparable.

### Hyperparameter search space

A light grid search: a few values for each of the most important hyperparameters. 5-fold stratified CV on the subset, `scoring="roc_auc"`, `n_jobs=-1`.

**Linear kernel.** Use `LinearSVC` (much faster than `SVC(kernel="linear")`, same kind of boundary).

| hyperparameter | values | why |
|---|---|---|
| `C` | 0.01, 0.1, 1, 10 | regularization strength; the only important knob for a linear SVM |

4 configurations x 5 folds = 20 fits.

**Polynomial kernel.** Use `SVC(kernel="poly")`.

| hyperparameter | values | why |
|---|---|---|
| `C` | 0.1, 1, 10 | regularization strength |
| `degree` | 2, 3 | 2 = pairwise interactions and squares; 3 adds three-way terms |
| `coef0` | 0, 1 | 0 keeps only the highest-order terms; 1 also keeps the lower-order terms |
| `gamma` | `"scale"` (fixed) | overlaps with `C` for the poly kernel, so it is held fixed to keep the grid small |

12 configurations x 5 folds = 60 fits.

Total: 16 configurations, 80 fits.

### Parallel coordinates plot

One plot showing all 16 configurations. Each line is one configuration. Axes, left to right:

1. `kernel` (linear, poly)
2. `C` (log scale)
3. `degree` (linear is placed at 1)
4. `coef0` (linear is placed at 0)
5. mean CV AUC

All lines are drawn in the neutral grey (`PALETTE["neg"]`). The best configuration by mean CV AUC is drawn thick in `PALETTE["pos"]` and labelled with its AUC. Also print the full grid as a table sorted by CV AUC.

### Final SVM models

Keep the best configuration **of each kernel**, so that both appear in the results table.

- Linear: refit the best `C` on the full `train` set (it is cheap).
- Poly: refit the best configuration on the full `train` set if the budget allows it, otherwise on the largest size the budget allows. The timing cell decides.

SVMs do not output probabilities. Threshold tuning uses `decision_function` scores. The x-axis of the threshold plot is therefore the SVM score, not a 0-1 probability. No Platt scaling for now.

### Deferred

- RBF and sigmoid in the overnight search: only if they pass the kernel screen's decision rule.
- Tuning `gamma` for the poly kernel (it only matters through its ratio with `coef0`).

## Threshold tuning

For each model, produce an ROC curve on the left and a threshold tuning plot on the right. On the X, place the threshold. On the Y, plot Precision, Recall, and F2. Mark the optimal spot as determined by F2. All on `validation`.

## Results

Produce a table that shows all models, with these columns: Rank, Model, Validation F2, Test F2. Rank by Validation F2. Expected rows: Logistic regression, SVM (linear), SVM (poly).

Also, produce a confusion matrix for the best model on the test set.

## Open questions

- Is 8 hours the right overnight budget?

