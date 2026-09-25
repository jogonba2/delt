# Experimental Results and Analysis

This subfolder contains the results and analysis of the experiments conducted for the paper, including additional experiments using **Jev**.

## Results

The complete set of experimental results is available in [`all_results.tsv`](all_results.tsv).

The file contains the following information for each experiment:

* **Modality**
* **Encoder name**
* **Dataset**
* **Training strategy**
* **Number of shots**
* **Macro F1 score**
* **Bootstrapping confidence intervals** for Macro F1
* **Training time**
* **Number of parameters**

The results include both the main experiments and additional experiments conducted using Jev.

## Analysis and Visualization
The [`inspect_results.ipynb`](inspect_results.ipynb) notebook provides an interactive overview and analysis of the experimental results and can be used to reproduce the analysis and plots of the paper.

The notebook:

* Displays and explores the complete set of experimental results.
* Compares **zero-shot** and **label-tuned** sub-1B encoders with **Jev**.
* Analyzes performance across datasets, modalities, training strategies, and numbers of shots.
* Generates the plots and figures presented in the paper.
