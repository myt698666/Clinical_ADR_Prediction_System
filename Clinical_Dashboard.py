import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
from Bio import Entrez

warnings.filterwarnings('ignore')

# NCBI requires an email address for API usage
Entrez.email = "clinical_research@example.com"

# Page Configuration
st.set_page_config(page_title="Clinical ADR Pro", layout="wide")


class ClinicalMLP(nn.Module):
    def __init__(self, input_dim):
        super(ClinicalMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def predict_with_uncertainty_single(model, x_tensor, n_iter=100):
    # Set the whole model to evaluation mode to bypass BatchNorm batch size 1 error
    model.eval()

    # Specifically enable Dropout layers for MC-Dropout sampling
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

    preds = []
    for _ in range(n_iter):
        with torch.no_grad():
            preds.append(model(x_tensor).item())

    preds = np.array(preds)
    return preds.mean(), preds.std(), preds


@st.cache_data
def load_data():
    try:
        slim_df = pd.read_csv("STITCH_Identifiers/Slim_Fused_Feature_Matrix.csv", index_col=0)
        vip_features = slim_df.columns.tolist()

        inputs = pd.read_csv("STITCH_Identifiers/Ultimate_Fused_Feature_Matrix.csv")
        outputs = pd.read_csv("ADR_Summary/SOC_significance_matrix.csv")

        target_soc = 'Neopl'
        merged_data = inputs.merge(outputs[['Drug', target_soc]], how='left', left_on='Matched Drug',
                                   right_on='Drug').dropna()

        X = merged_data[vip_features].values
        y = merged_data[target_soc].values.astype(int)

        # Explicitly extract the drug names to fix the dropdown numerical issue
        drug_names = merged_data['Drug'].astype(str).tolist()

        return X, y, drug_names, vip_features, merged_data
    except Exception as e:
        st.error(f"Data loading failed: {e}")
        return None, None, [], [], None


@st.cache_resource
def train_global_model(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = ClinicalMLP(X.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    X_t = torch.tensor(X_scaled, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    model.train()
    for _ in range(100):
        optimizer.zero_grad()
        criterion(model(X_t), y_t).backward()
        optimizer.step()

    return model, scaler


def search_pubmed(query, max_results=3):
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
        record = Entrez.read(handle)
        handle.close()
        return record["IdList"]
    except Exception as e:
        return []


def fetch_pubmed_details(id_list):
    if not id_list:
        return []
    try:
        ids = ",".join(id_list)
        handle = Entrez.efetch(db="pubmed", id=ids, retmode="xml")
        records = Entrez.read(handle)
        handle.close()

        articles = []
        for pubmed_article in records['PubmedArticle']:
            article_data = pubmed_article['MedlineCitation']['Article']
            title = article_data.get('ArticleTitle', 'No Title Available')
            pub_year = 'Unknown Year'
            try:
                pub_year = article_data['Journal']['JournalIssue']['PubDate']['Year']
            except KeyError:
                pass
            pmid = str(pubmed_article['MedlineCitation']['PMID'])
            articles.append({'title': title, 'year': pub_year, 'pmid': pmid})
        return articles
    except Exception as e:
        return []


def main():
    st.title("Clinical ADR Pro: Toxicity Predictor")
    st.markdown(
        "A machine learning framework integrating multi-modal features and Bayesian Uncertainty Quantification.")
    st.divider()

    with st.spinner("Initializing Knowledge Graph and Multi-modal Matrix..."):
        X, y, drug_names, vip_features, merged_data = load_data()
        if X is None:
            return
        model, scaler = train_global_model(X, y)

    st.sidebar.header("Patient / Drug Input")
    st.sidebar.info("Select a drug compound to predict the risk of inducing neoplasm-related adverse reactions.")

    selected_drug = st.sidebar.selectbox("Select Drug Compound:", drug_names, key="drug_selector")

    if st.sidebar.button("Run Prediction Analysis", type="primary"):
        with st.container():
            st.subheader(f"Drug Analysis Report: {selected_drug}")

            try:
                drug_idx = drug_names.index(selected_drug)
                drug_features = X[drug_idx].reshape(1, -1)
                drug_scaled = scaler.transform(drug_features)
                x_tensor = torch.tensor(drug_scaled, dtype=torch.float32)

                # Run MC-Dropout
                mean_prob, std, all_preds = predict_with_uncertainty_single(model, x_tensor, n_iter=100)

                # Rendering Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(label="Toxicity Risk Probability (Mean)", value=f"{mean_prob * 100:.2f}%")
                with col2:
                    st.metric(label="Prediction Uncertainty (Std)", value=f"{std:.4f}")
                with col3:
                    confidence = "High Confidence" if std < 0.1 else "Low Confidence"
                    color = "normal" if std < 0.1 else "inverse"
                    st.metric(label="System Confidence", value=confidence,
                              delta="Manual Review Required" if std >= 0.1 else "Reliable", delta_color=color)

                st.markdown("### Bayesian Sampling Probability Distribution (MC-Dropout 100 iterations)")
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.hist(all_preds, bins=20, color='#3498db', edgecolor='white', alpha=0.7)
                ax.axvline(mean_prob, color='#e74c3c', linestyle='dashed', linewidth=2, label=f'Mean: {mean_prob:.3f}')
                ax.set_xlabel('Predicted Probability')
                ax.set_ylabel('Frequency')
                ax.legend()

                st.pyplot(fig)
                plt.close(fig)

                # Fetch real label if available
                try:
                    real_label = merged_data.iloc[drug_idx][merged_data.columns[-1]]
                    label_text = "Positive (1)" if real_label == 1 else "Negative (0)"
                    st.info(f"Real-world Database Record: The neoplasm label for this drug is {label_text}")
                except Exception as e:
                    st.warning("Could not retrieve real-world label for this specific format.")

                if std >= 0.1:
                    st.warning("Low Confidence Detected. Initiating Automated Literature Retrieval Protocol...")
                    with st.spinner("Searching PubMed for clinical evidence..."):
                        search_query = f"{selected_drug} AND Neoplasm AND Adverse Drug Reaction"
                        pmid_list = search_pubmed(search_query, max_results=3)
                        articles = fetch_pubmed_details(pmid_list)

                        if articles:
                            with st.expander("View Retrieved Scientific Literature", expanded=True):
                                st.markdown(
                                    "The system has retrieved the following related articles to assist manual review:")
                                for idx, article in enumerate(articles, 1):
                                    pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{article['pmid']}/"
                                    st.markdown(f"**{idx}. [{article['title']}]({pubmed_url})**")
                                    st.markdown(f"*Published: {article['year']} | PMID: {article['pmid']}*")
                        else:
                            st.info(
                                "Automated Retrieval Complete: No highly relevant articles found for this specific interaction in the recent indexing.")

            except Exception as e:
                st.error(f"Error during prediction: {e}")


if __name__ == "__main__":
    main()