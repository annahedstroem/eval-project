# QGE - Quality gap estimator

This project allows for the replication of the experiments in the paper "Evaluate with the Inverse: \\ Efficient Approximation of Latent Explanation Quality Distribution".

Please note that this is only the code for the experiments. The link to the final implementation of QGE will be available after the double-blind peer-review process.

## Instructions to replicate the experiments

Replicating one of the results generally consists of four steps:

  1. Get the data & train the model (notebooks in `nbs/0-Model training`)
  1. Compute the target measures (`src/extract_measures.py`)
  1. Compute the results for the paper (`src/2-measure-performance.py`, `src/get_deltas.py` & `src/get_qstds.py`)
  1. Generate plots if necessary (`nbs/3-plot.ipynb`)

### Details for each specific result

 - Section 4.1.a
```
    1. Get model & data: nbs/0-Model training/0-avila-train.ipynb & nbs/0-Model training/0-avila-train.ipynb
    2. Compute measures: src/1-extract_measures.py (line 90)
    3. Compute results: src/2-measure-performance.py (line 72)
    4. Figures 2 & 3: nbs/3-plot.ipynb (uncomment line in first cell)
```
 - Section 4.1.b
```
    1. Get models & data:
        - 20newsgroups: nbs/0-Model training/0-20newsgroups-train.ipynb
        - MNIST: nbs/0-Model training/0-MNIST-train.ipynb
        - CIFAR: nbs/0-Model training/0-CIFAR-train.ipynb
        - Imagenet: Not needed
    2. Compute measures: src/1-extract_measures.py (line 94)
    3. Compute results: src/2-measure-performance.ipynb (line 76)
    4. Aggregate results:
        - Table 1: src/3-get_deltas.py (line 8)
```

 - Section 4.1.c
```
    1. Get models & data:
        - ood-mean & ood-zeros: nbs/0-Model training/0-avila-train-ood.ipynb & nbs/0-Model training/0-avila-train-ood.ipynb
        - undertrained & untrained: nbs/0-Model training/0-avila-train.ipynb & nbs/0-Model training/0-avila-train.ipynb
            - Stop training when test accuracy reaches 70% for undertrained
            - Save weights with no training for untrained
    2. Compute measures: src/1-extract_measures.py (line 105)
    3. Compute results: src/2-measure-performance.ipynb (line 91)
    4. Aggregate results:
        - Table 2: src/3-get_deltas.py (line 30) & src/3-get_stds.py
```

 - Section 4.1.d
```
    1. Get models & data: nbs/0-Model training/0-cmnist-train.ipynb
    2. Compute measures:
        - Faithfulness: src/1-extract_measures_only_quantus.py (line 127)
        - Localization: src/1-extract_measures_localization-cmnist.py (line 122)
    3. Compute results: src/2-measure-performance.ipynb (lines 106 & 115)
    4. Aggregate results:
        - Table 3: src/3-get_deltas.py (line 47) (Pixel-Flipping results taken from Table 1).
        - Table 4: src/3-get-deltas.py (line 57)
```

 - Section 4.2
```
    Use meta_evaluation.py in "inverse-estimation" repository.
```

  - Appendix
 ```
    A.1 - Figure 4: After having the results for Section 4.1.a, run nbs/plot-qge-distribution.ipynb
    A.2, A.4 - Figures 5 & 6: After having the results for Section 4.1.a, nbs/3-plot.ipynb (uncomment line in first cell)
    A.3 - Table 5:
        1. Compute measures: src/1-extract_measures.py (line 118)
        2. Compute results: src/2-measure-performance.ipynb (line 123)
        3. Aggregate results: src/3-get_deltas.py (line 66)
    A.5 - Figures 7 & 8: use meta_evaluation.py in "inverse-estimation" repository.
```