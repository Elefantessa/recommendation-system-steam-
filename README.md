
# Steam Game Recommender: Balancing Accuracy & Diversity

**Author:** Hala Alramli
**Project:** Artificial Intelligence Project, University of Antwerp

## Project Overview

Recommender systems in the gaming industry often face the **"Accuracy-Diversity Dilemma."** Traditional algorithms excel at predicting relevant items but frequently trap users in "filter bubbles" by recommending redundant content (e.g., suggesting multiple sequels of the same game).

This project implements a **Hybrid Recommender System** using the **Steam Dataset**. It combines:

1. **EASE (Embarrassingly Shallow Autoencoders):** For high-accuracy candidate generation based on implicit feedback.

2. **MMR (Maximal Marginal Relevance):** For post-processing re-ranking to inject diversity and novelty.

The goal is to answer the research question: *How does a diversity-aware re-ranking algorithm impact the trade-off between relevance and exploration in game recommendations?*

## Key Features & Methodology

### 1. Data Preprocessing

* **Dataset:** Steam interactions (Playtime) and Metadata.

* **Filtering:** Recursive **5-core filtering** to remove users/items with fewer than 5 interactions, ensuring model stability.

* **Implicit Feedback:** Playtime is binarized (1 = played, 0 = not played).

* **Seen Item Filtering:** Strictly excluding games the user has already played to ensure recommendations are for **new discovery**.

### 2. Feature Engineering

Extensive analysis was conducted to determine the best metadata for calculating diversity.

* **Selected Features:** `Genres` + `Tags` (Combined).

* **Excluded Features:** `Specs` (Technical specifications were found to act as noise, artificially inflating similarity between unrelated games).

### 3. Models

* **Baseline:** User-Based K-Nearest Neighbors (**UserKNN**).

* **Core Model:** **EASE** ($\lambda=300$). Selected for its closed-form solution and superior performance on sparse data.

* **Re-ranking:** **MMR** ($\lambda=0.7$). Greedily selects items that maximize a linear combination of relevance and dissimilarity to the already selected list.

## Repository Structure

### 1. Main Pipeline

* `EASE+MMR_recomendation_final.ipynb`: **(Start Here)** The final submission notebook. It loads the data, trains the optimal EASE model, builds the feature matrix (Genres+Tags), runs MMR re-ranking, and generates the final submission file (`submission.csv`).

### 2. Analysis & Tuning

* `EASE+MMR_validation+tuning.ipynb`: Performs Grid Search to find the best regularization parameter for EASE ($\lambda=300$) and analyzes the Accuracy-Diversity trade-off curve.

* `userKNN.ipynb`: Implements the baseline neighbor-based model for comparison metrics.

### 3. Feature Selection Experiments

These notebooks contain the ablation study used to determine which content features best represent diversity:

* `feature-selection-generic-tag-spec.ipynb`: Analyzing all features combined.

* `feature-selection-tag.ipynb`: Analyzing Tags only.

* `feature-selection-spec.ipynb`: Analyzing Specs only (Proved to be noise).

* `feature-selection-generic.ipynb`: Analyzing Genres only.

## Key Results

Through rigorous offline evaluation using **Strong Generalization** (disjoint user split), we found:

1. **Baseline Superiority:** EASE outperformed UserKNN by ~3% in both Accuracy (NDCG) and Diversity (ILD).

2. **The "Sweet Spot":** Applying MMR with **$\lambda=0.7$** is the Pareto-optimal solution.

   * **Diversity Gain:** +7% (ILD increased from 0.62 to 0.69).

   * **Accuracy Loss:** < 4% (NDCG dropped slightly from 0.37 to 0.35).

3. **Conclusion:** Diversity is not a zero-sum game. A small sacrifice in theoretical accuracy yields a significantly more varied and engaging recommendation list.

## Installation & Usage

### Requirements

* Python 3.8+

* Jupyter Notebook

* Libraries: `numpy`, `pandas`, `scipy`, `sklearn`, `matplotlib`, `tqdm`

### Steps to Run

1. **Data Setup:** Ensure the Steam dataset CSV files are placed in a folder named `./cleaned_datasets_students/`.

2. **Run the Final Model:** Open `EASE+MMR_recomendation_final.ipynb` and execute all cells. This will train the model and save the recommendations.

3. **Reproducibility:** A fixed random seed (42) is used throughout the notebooks to ensure results are reproducible.

## References

1. H. Steck, "Embarrassingly Shallow Autoencoders for Sparse Data," WWW '19.

2. J. Carbonell and J. Goldstein, "The Use of MMR, Diversity-Based Reranking," SIGIR '98.

```
