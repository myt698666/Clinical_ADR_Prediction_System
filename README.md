Clinical ADR Prediction Ecosystem

This repository contains the data and code for an advanced machine learning framework for predicting Adverse Drug Reactions (ADRs). The framework encompasses the original baseline implementation established in the paper 'An Interpretable Machine Learning Framework for Adverse Drug Reaction Prediction from Drug-Target Interactions' (Roberts-Nuttall et al., 2026), alongside an upgraded, multi-modal predictive ecosystem.

Project Overview

The original baseline provided a static approach for ADR prediction using Random Forest models and standard database descriptors. This repository has been extended to include a robust, end-to-end pipeline that addresses the original baseline limitations, including data retention issues, class imbalance, and model interpretability. The following sections outline the structure of the repository and the enhancements made to the original methodology.

Folder Breakdown

The repository is structured to maintain compatibility with the original baseline while introducing new modules for the upgraded predictive ecosystem.

baseline_2026

This folder contains the original code and data structures used in the 2026 baseline study. It serves as the reference point for all comparative benchmarking and contains the initial Random Forest implementation and static feature extraction scripts.

Clinical_Dashboard.py

This script provides an interactive clinical decision support interface built with Streamlit. It enables real-time ADR risk assessment, visualizes prediction uncertainty derived from Bayesian inference, and integrates with the NCBI PubMed API to provide peer-reviewed medical literature when model confidence is low.

Ensemble_Voting_Engine.py

This module implements a hybrid soft-voting ensemble model that combines the predictive performance of XGBoost with a custom 3-layer PyTorch Multilayer Perceptron. This engine is designed to capture complex, non-linear biological pathways that traditional tree-based models fail to represent.

Hierarchical_Granularity_Expansion.py

This script implements a Local Classifier per Parent Node architecture, allowing the system to route macro System Organ Class predictions down to specific Preferred Term clinical conditions, effectively increasing the granularity of ADR prediction.

WGAN_GP_Synthesizer.py

This module contains a Wasserstein GAN with Gradient Penalty used for synthetic tabular data generation. It addresses severe class imbalance in rare ADR categories by synthesizing high-fidelity minority feature vectors.

Knowledge_Graph_Generator.py

This script constructs a tripartite directed graph linking chemical drug compounds to biological protein targets and MedDRA clinical ADR categories. It provides topological interpretability of the pharmacological pathways identified by the model.

FAERS_Global_Data_Scaling_Engine.py

This engine provides a standardized adapter for ingesting real-world pharmacovigilance data from the US FDA FAERS and JADER databases, enabling the system to scale beyond the initial study dataset.

Quick Start and Usage

To replicate the original baseline results, navigate to the baseline_2026 directory and execute the primary Random Forest scripts. To utilize the upgraded predictive ecosystem, execute the Clinical_Dashboard.py script using Streamlit. All necessary environment dependencies are listed in the requirements.txt file.

Citation

This project extends the methodology originally presented by:
Roberts-Nuttall J, Jones AM, Castellani M, Pham D (2026). An interpretable machine learning framework for adverse drug reaction prediction. PLoS One.