<br/><br/>
<p align="center">
<img width="250" src="https://github.com/annahedstroem/eval-project/blob/73c7b62b8d8243f8f883358a83e50204cd5183ae/qge_logo.png">
<h3 align="center"><b>A method to estimate explanation quality</b></h3>
<p align="center">PyTorch</p>

This repository contains the code and experiments for the paper **["Evaluate with the Inverse: Efficient Approximation of Latent Explanation Quality Distribution"]([Link]([https://openreview.net/pdf?id=ukLxqA8zXj](https://arxiv.org/pdf/2502.15403)))** (AAAI, 2025 alignment track!) by Eiras-Franco et al., 2025. 

Please note that this repository is under active development!

## Citation

If you find this work interesting or useful in your research, use the following Bibtex annotation to cite us:

```bibtex
@article{gqe2025,
    title={Evaluate with the Inverse: Efficient Approximation of Latent Explanation Quality Distribution}, 
    author={Carlos Eiras-Franco and Anna Hedström and Marina M. -C. Höhne},
    booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
    volume={39}
    year={2025},
}
```

## Paper highlights 📚

The Quality gap estimator helps determine if a given explanation for a machine learning model prediction for a particular input is better or worse than alternative explanations attending to any quality metric.

### Example:

For input $\mathbf{x_0}$, your model $f$ predicts output $\mathbf{y_0}$. You use an explanation method (for instance, [Integrated gradients](https://captum.ai/docs/attribution_algorithms#integrated-gradients)) to get an explanation $\mathbf{e}$, consisting of an attribution value for each variable $x_i$ in $\mathbf{x}$.

**The problem**
Now you want to evaluate whether $\mathbf{e}$ is a good explanation. You use a function $\psi(\mathbf{x_0},\mathbf{y_0},f,\mathbf{e})$ to determine some quality of the explanation (for this example, let's say it's Faithfulness using [FaithfulnessCorrelation](https://github.com/understandable-machine-intelligence-lab/Quantus/blob/main/quantus/metrics/faithfulness/faithfulness_correlation.py)). This yields a scalar value that measures the faithfulness of $\mathbf{e}$; let's say it's 4. Is your explanation more faithful than other explanations $\mathbf{e}'$? Maybe the most explanations for $f(\mathbf{x_0})=\mathbf{y_0}$ have a faithfulness of around 4. Or maybe 4 is a very exceptional faithfulness value for the explanations of $f(\mathbf{x_0})=\mathbf{y_0}$, which usually have faithfulness around 0.5. How can you determine if $\mathbf{e}$ has exceptional faithfulness when compared to alternative explanations?

**The solution**
You could sample many alternative explanations to get a sense of what the distribution of the faithfulness value is for different explanations. However, this is very costly, since running _FaithfulnessCorrelation_ is computationally expensive.

QGE allows you to obtain a reliable estimation of how exceptionally faithful (or any other quality metric you may be using) $\mathbf{e}$ by running _FaithfulnessCorrelation_ (or your desired metric) just once more.

## How to use QGE 👩‍💻🧑‍💻

This project allows for the replication of the experiments in the paper.

Please note that this is only the code for the experiments. To use QGE, please refer to its implementation in the [Quantus framework](https://github.com/understandable-machine-intelligence-lab/Quantus).

## Instructions to replicate the experiments 🧪

Replicating one of the results generally consists of four steps:

  1. Get the data & train the model (notebooks in `nbs/0-Model training`)
  1. Compute the target measures (`src/extract_measures.py`)
  1. Compute the results for the paper (`src/2-measure-performance.py`, `src/get_deltas.py` & `src/get_qstds.py`)
  1. Generate plots if necessary (`nbs/3-plot.ipynb`)

**Details for each specific result**

<details>
<summary><b>Section 4.1.a</b></summary>

1. **Get model & data:**  
   `nbs/0-Model training/0-avila-train.ipynb` & `nbs/0-Model training/0-avila-train.ipynb`  
2. **Compute measures:**  
   `src/1-extract_measures.py` (line 90)  
3. **Compute results:**  
   `src/2-measure-performance.py` (line 72)  
4. **Figures 2 & 3:**  
   `nbs/3-plot.ipynb` (uncomment line in first cell)  

</details>

<details>
<summary><b>Section 4.1.b</b></summary>

1. **Get models & data:**  
   - **20newsgroups:** `nbs/0-Model training/0-20newsgroups-train.ipynb`  
   - **MNIST:** `nbs/0-Model training/0-MNIST-train.ipynb`  
   - **CIFAR:** `nbs/0-Model training/0-CIFAR-train.ipynb`  
   - **Imagenet:** Not needed  
2. **Compute measures:**  
   `src/1-extract_measures.py` (line 94)  
3. **Compute results:**  
   `src/2-measure-performance.ipynb` (line 76)  
4. **Aggregate results:**  
   - **Table 1:** `src/3-get_deltas.py` (line 8)  

</details>

<details>
<summary><b>Section 4.1.c</b></summary>

1. **Get models & data:**  
   - **ood-mean & ood-zeros:**  
     `nbs/0-Model training/0-avila-train-ood.ipynb` & `nbs/0-Model training/0-avila-train-ood.ipynb`  
   - **undertrained & untrained:**  
     `nbs/0-Model training/0-avila-train.ipynb` & `nbs/0-Model training/0-avila-train.ipynb`  
     - Stop training when test accuracy reaches 70% for undertrained  
     - Save weights with no training for untrained  
2. **Compute measures:**  
   `src/1-extract_measures.py` (line 105)  
3. **Compute results:**  
   `src/2-measure-performance.ipynb` (line 91)  
4. **Aggregate results:**  
   - **Table 2:** `src/3-get_deltas.py` (line 30) & `src/3-get_stds.py`  

</details>

<details>
<summary><b>Section 4.1.d</b></summary>

1. **Get models & data:**  
   `nbs/0-Model training/0-cmnist-train.ipynb`  
2. **Compute measures:**  
   - **Faithfulness:** `src/1-extract_measures_only_quantus.py` (line 127)  
   - **Localization:** `src/1-extract_measures_localization-cmnist.py` (line 122)  
3. **Compute results:**  
   `src/2-measure-performance.ipynb` (lines 106 & 115)  
4. **Aggregate results:**  
   - **Table 3:** `src/3-get_deltas.py` (line 47) (Pixel-Flipping results taken from Table 1).  
   - **Table 4:** `src/3-get-deltas.py` (line 57)  

</details>

<details>
<summary><b>Section 4.2</b></summary>

Use `meta_evaluation.py` in "inverse-estimation" repository.  

</details>

<details>
<summary><b>Appendix</b></summary>
  
A.1 - Figure 4: After having the results for Section 4.1.a, run `nbs/plot-qge-distribution.ipynb`
A.2, A.4 - Figures 5 & 6: After having the results for Section 4.1.a, `nbs/3-plot.ipynb` (uncomment line in first cell)
A.3 - Table 5:
    1. Compute measures: `src/1-extract_measures.py` (line 118)
    2. Compute results: `src/2-measure-performance.ipynb` (line 123)
    3. Aggregate results: `src/3-get_deltas.py` (line 66)
A.5 - Figures 7 & 8: use `meta_evaluation.py` in "inverse-estimation" repository.

</details>

### Thank you

We hope our repository is beneficial to your work and research. If you have any feedback, questions, or ideas, please feel free to raise an issue in this repository. Alternatively, you can reach out to us directly via email for more in-depth discussions or suggestions. 

📧 Contact us: 
- Carlos Eiras-Franco: [carlos.eiras.franco@udc.es](mailto:carlos.eiras.franco@udc.es)
- Anna Hedström: [hedstroem.anna@gmail.com](mailto:hedstroem.anna@gmail.com)

Thank you for your interest and support!
