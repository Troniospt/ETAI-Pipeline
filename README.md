# Baseline Predictive Pipeline -- ETAI
20260490 António Favila Vieira

WEEK 2:
Conclusion:

The logistic regression  its training and test accuracies are almost identical. Probably no overfitting. Its  accuracy its 67.8% (ok), so the model is good but still could be better.

The second model seems to overfit. It has 82.9% training accuracy but only 63.1% test accuracy—a 19.8 difference. It is not dealing weel with unseen data.

WEEK 3:
Before data cleaning, the Logistic Regression model suffered from underfitting, demonstrating high stability but low predictive power with a well-aligned training accuracy of 67.9% and test accuracy of 67.8% (a minimal gap of 0.001). However, after cleaning, its training accuracy increased substantially to 83.1%, while its test performance degraded to 61.9%, causing the generalization gap to expand dramatically to 0.212. This shift indicates that while the cleaning process introduced feature transformations or encodings that allowed the linear model to better fit the training set, it ultimately induced severe overfitting to training-specific noise rather than improving its ability to evaluate unseen data

In contrast, the Decision Tree model exhibited persistent overfitting across both datasets, maintaining nearly identical training and testing metrics regardless of data cleaning. Prior to cleaning, the tree scored 82.9% on training data and 63.1% on test data (a 0.198 gap), which slightly deteriorated post-cleaning to an 83.1% train accuracy and a 62.1% test accuracy (a 0.210 gap). Because unconstrained decision trees naturally memorize patterns and noise, cleaning the raw features had a negligible effect on reducing variance, demonstrating that structural regularization—such as limiting tree depth—is necessary to improve generalization rather than relying solely on data preprocessing.

Week 4:
Both holdout and cross-validation have logistic regression as the best model, with validation accuracy of 67.1% and 67.2%. It have small train\validation gap that indicates small overfitting. Random forest and decision tree have higher training scores but lower validation results with larger gaps, that problably indicates overfitting. The dummy model provided only a 54.9% baseline and failed to identify any positive cases.

On the locked test set, logistic regression achieved 66% accuracy, confirming that its performance generalizes reasonably well. But, it detected only 48% of positive cases, and its false-positive rate was higher for African-American individuals than for Caucasian individuals (28% vs 14%). So logistic regression is the strongest model tested, but its recall and limitations should be considered.

Holdout and cross-validation produced similar rankings, both selecting logistic regression. However, cross-validation is more reliable because it evaluates each model across several data splits, while holdout depends on a single split and may produce less stable results.

WEEK 5:
Hyperparameter tuning greatly improved the Decision Tree. With its default settings, the model obtained 61.0% validation accuracy and showed a large train-validation gap of 0.085, indicating overfitting. After Optuna tuning, its nested cross-validation accuracy increased to 67.2% with a much smaller gap of 0.013. The selected configuration used a maximum depth of 6, a minimum of 101 samples per leaf, and the Gini criterion. This confirms that limiting the size of the tree helps it generalize instead of memorizing the development data.

The Random Forest also benefited from tuning, increasing from 65.0% to 68.1% nested cross-validation accuracy while reducing its gap from 0.082 to 0.009. Its best configuration used a maximum depth of 22, 42 samples per leaf, and half of the available features at each split. The best-trial tuning scores were slightly higher than the honest nested scores, but the optimism was only 0.002 for the Decision Tree and 0.001 for the Random Forest. Therefore, the nested cross-validation results are the numbers that should be reported, since they evaluate the complete tuning procedure on folds that were not used to choose the hyperparameters.

WEEK 6:
The ensemble comparison showed that tuning was most useful for the two forest models. Random Forest improved from 65.0% to 68.1%, while Oblique Random Forest improved from 64.7% to 67.9% and reduced its large default overfitting gap from 0.114 to 0.013. AdaBoost already performed well with its default shallow trees, obtaining 67.6%, and tuning produced a smaller improvement to 68.2%. Its final configuration used trees of depth 4, 220 estimators, and a learning rate of 0.025.

AdaBoost achieved the highest nested cross-validation mean, but its advantage over Random Forest was only 0.001, which is much smaller than the fold-to-fold standard deviations of 0.017 and 0.015. The tuned ensembles should therefore be considered practically tied rather than treating AdaBoost as a clear winner. I would report AdaBoost's honest nested score of 68.2% ± 1.7% as the numerical best result, while noting that Random Forest is an equally reasonable choice and that all models appear to be approaching the performance limit of the available features.


This is the **starting point** for your semester project: a small but *complete* predictive pipeline -- every piece a real project needs (entry point, config, data loading, preprocessing, model, evaluation), just kept as simple as possible for now.

The task: predict two-year recidivism using ProPublica's COMPAS
dataset -- the data behind a real 2016 investigation into a risk-
assessment algorithm actually used by US courts to help inform bail and sentencing decisions. See `data/README.md` for the full problem description and a complete data dictionary before you start.

It has some **deliberately weak spots**. Part of your work this
semester is finding them and making them better -- see the pipeline progress table below, which tracks what changes and why as the weeks
go on.

## Project structure

```
.
├── main.py                # entry point: run the whole pipeline
├── config.yaml             # all tunable settings live here
├── requirements.txt
├── src/
│   ├── data.py             # loading
│   ├── preprocessing.py    # cleaning + train/test split
│   ├── model.py             # model construction
│   ├── evaluate.py         # accuracy metrics + fairness check
│   └── results.py          # saves each run's report to disk
├── results/                # created automatically -- one file per run (not tracked in git)
└── data/
    ├── compas_two_year_recidivism.csv
    └── README.md            # problem description + full data dictionary
```

## Pipeline progress


| Week | Practical class focus | Added to the pipeline |
|------|------------------------|------------------------|
| 2 | Introduction & baseline pipeline | Initial version: project structure, a single naive train/test split (no cross-validation), minimal preprocessing (drop rows with missing values, one-hot encode categoricals), logistic regression baseline, a first (deliberately simple) fairness check comparing our model's and COMPAS's own false-positive rate by race, train-vs-test accuracy reporting (to start spotting overfitting), and each run's full report saved automatically to `results/` |
| 3 | Data quality and cleaning | Config-driven validity rules, category canonicalisation, placeholder handling, duplicate removal, and missingness indicators. |
| 4 | Preprocessing and cross-validation | Leakage-safe preprocessing inside a scikit-learn pipeline, a locked test set, stratified CV, out-of-fold reports, and comparison of baseline models. |
| 5 | Hyperparameter tuning | Optuna search spaces and nested cross-validation so tuned models are assessed on outer folds that were not used to choose their parameters. |
| 6 | Ensembles | Random forest, an oblique random-projection forest, and AdaBoost, with model-specific nested-CV search spaces. |

## Model evaluation

### Hyperparameter tuning

The pipeline uses Optuna to sample each model's configured search space. Every candidate is evaluated by five inner folds of the complete preprocessing/model pipeline, while five outer folds estimate the performance of the tuning procedure without reusing the data that chose the hyperparameters. The default configuration uses 30 trials; the best-trial score selects parameters, but the nested-CV score is the number reported for model comparison because it corrects the winner's-curse optimism.

| Model | Default: CV mean ± std | Tuned: nested CV mean ± std | Tuning score (best trial) | Optimism | Chosen hyperparameters |
|---|---:|---:|---:|---:|---|
| Decision tree | 0.610 ± 0.018 | 0.672 ± 0.017 | 0.679 | +0.002 | `max_depth=6`, `min_samples_leaf=101`, `criterion=gini` |
| Logistic regression | 0.672 ± 0.013 | — | — | — | — |
| Random forest | 0.650 ± 0.019 | 0.681 ± 0.015 | 0.683 | +0.001 | `max_depth=22`, `min_samples_leaf=42`, `max_features=0.5` |

Tuning substantially improves the decision tree and random forest, by about 0.062 and 0.031 respectively—both larger than their default fold-to-fold standard deviations. Logistic regression was not tuned in the supplied run. Tuning changes the tree from a clearly overfitting model into one competitive with logistic regression, while the tuned random forest is slightly ahead; the nested-CV result is the appropriate performance estimate to report.

### Ensembles

The default and nested-CV values reproduce the supplied class notebook and pre-run folds. The tuning scores and final hyperparameters come from this repository's completed 30-trial `python main.py` runs on 17 October 2026.

| Model | Default: CV mean ± std (gap) | Tuned: nested CV mean ± std (gap) | Tuning score (best trial) | Chosen hyperparameters |
|---|---:|---:|---:|---|
| Logistic regression | 0.672 ± 0.013 (+0.003) | — | — | — |
| Decision tree | 0.610 ± 0.018 (+0.085) | 0.672 ± 0.017 (+0.013) | 0.679 | `max_depth=6`, `min_samples_leaf=101`, `criterion=gini` |
| Random forest | 0.650 ± 0.019 (+0.082) | 0.681 ± 0.015 (+0.009) | 0.683 | `max_depth=22`, `min_samples_leaf=42`, `max_features=0.5` |
| Oblique random forest | 0.647 ± 0.011 (+0.114) | 0.679 ± 0.013 (+0.013) | 0.683 | `max_depth=6`, `min_samples_leaf=3`, `n_components=5` |
| AdaBoost | 0.676 ± 0.012 (+0.006) | 0.682 ± 0.017 (+0.004) | 0.684 | `max_depth=4`, `n_estimators=220`, `learning_rate=0.025` |

AdaBoost has the highest nested-CV mean, but its lead over random forest is only 0.001, far below either model's fold-to-fold standard deviation. The three tuned ensembles are therefore effectively tied on these folds; selecting AdaBoost as an unequivocal winner would overstate the evidence. Tuning chiefly helps the default forests, whose deep trees overfit, while default AdaBoost with shallow stumps is already well controlled. The number to report for model comparison is the honest nested-CV score—not the best tuning-trial score—with AdaBoost at 0.682 ± 0.017 as the narrow numerical leader.

## Environment setup

You only need to do this once per machine.

### macOS / Linux
```bash
python3 -m venv venv                 # creates an isolated Python environment in a folder called "venv"
source venv/bin/activate             # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```

### Windows -- PowerShell
```powershell
python -m venv venv                  # creates an isolated Python environment in a folder called "venv"
venv\Scripts\activate                # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```
If PowerShell blocks the activation script, run this once first:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Windows -- cmd.exe
Same three steps as above, just with cmd's own activation command:
```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

Once the environment is active you'll see `(venv)` at the start of your prompt. To leave it later, run `deactivate` (same command on every OS).

### Every time after the first

Creating the environment and installing packages only needs to happen once, ever. Every other time you sit down to work -- a new terminal window, the next practical class, tomorrow -- you don't repeat any of the steps above. From the project's root folder, you just need to:

**macOS / Linux**
```bash
source venv/bin/activate
python main.py
```

**Windows**
```powershell
venv\Scripts\activate
python main.py
```

That's it -- activate, then run. If you don't see `(venv)` at the start of your prompt, the environment isn't active and `python main.py` may use the wrong Python (or fail to find a package) entirely.

## Running the pipeline

With the environment active (see above), from the project's root
folder, on any OS:
```bash
python main.py
```

This loads `config.yaml`, loads and preprocesses the data, trains the model, and prints:
- **train accuracy and test accuracy, side by side.** Comparing the two is how you catch overfitting: if the model looks much better on the data it was trained on than on data it's never seen, it has memorised rather than learned something that generalises. 
- a classification report on the test set
- a false-positive-rate-by-race comparison between our model and
  COMPAS's own score

All of this is also saved to a timestamped file in `results/` (e.g.`results/run_20260916_143012.txt`), so it doesn't just scroll past in your terminal -- open it later, or change something in `config.yaml` (like the model type) and compare the new file to the last one.
`results/` is created automatically the first time you run the
pipeline, and isn't tracked in git (see `.gitignore`) since it's
generated output, not source.

You're free to improve on this structure or restructure it entirely -- what matters is that your project stays runnable end-to-end with a single command, and that each piece (data, preprocessing, model, evaluation) stays easy to find and change independently.

## Dataset

See `data/README.md`.
