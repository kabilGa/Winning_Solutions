🌿 ESG Multi-Label Classification: Rank 1 Phase 2
This is a high-performance stacking ensemble. To make your GitHub README stand out, we’ll use a clean, professional structure with shields, a visual pipeline explanation, and an interactive "Quick Start" feel.

This repository contains the winning Phase 2 solution for ESG (Environmental, Social, and Governance) text classification. The pipeline utilizes a Triple-Transformer Stacking Ensemble with a LightGBM Meta-Learner to achieve a robust F1-score.
🚀 The Architecture

The solution follows a multi-stage stacking strategy:

    Base Layer: 7-Fold Cross-Validation using three diverse backbones:

        ClimateBERT: Specialized for environmental context.

        DistilBERT: Efficient, general-purpose linguistic features.

        DeBERTa-v3-Base: State-of-the-art attention mechanism.

    Meta Layer: A LightGBM classifier trained on the Out-of-Fold (OOF) probabilities.

    Post-Processing: Logical ESG consistency enforcement (ensuring non_ESG exclusivity).

🛠️ Installation & Setup

Get the environment ready in seconds:
Bash

# Install the core dependency for stratified multi-label splitting
pip install iterative-stratification

# Ensure you have the heavy hitters
pip install torch transformers datasets lightgbm scikit-learn

📊 Performance Logic
Feature	implementation
Strategy	7-Fold Multilabel Stratified CV
Loss Function	Weighted BCE (Handling Class Imbalance)
Scheduler	Cosine Annealing with Warmup
Meta-Threshold	0.48 (Optimized for F1-Max)
🏃 How to Run
1. Data Placement

Ensure your datasets are located in the following directory structure:
./datasets/kabilgannouni/godatasc/

    train.csv

    test.csv

2. Execute Training

Run the script to perform the full 21-fold training (3 models × 7 folds):
Python

python train_stacking_ensemble.py

3. Logic Recovery (Auto-Applied)

The script automatically enforces the ESG Mutual Exclusivity Rule:

    If (E or S or G) == 1, then non_ESG = 0.
    If (E and S and G) == 0, then non_ESG = 1.

📁 Repository Structure

    train_stacking_ensemble.py: The main execution engine.

    esg_rank1_phase2_final.csv: The final generated submission file.

    README.md: Documentation.

🤝 Contributing

Feel free to fork this repo, open issues, or submit PRs if you want to optimize the DeBERTa-v3 hyperparameters further!

Disclaimer: This model was optimized for a specific competition distribution. Results may vary on general ESG corpora.
