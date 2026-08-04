import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')


class FAERSDataIngestionEngine:
    """
    Standardized adapter for ingesting raw pharmacovigilance data
    (e.g., FDA FAERS, JADER) into the system's internal feature space.
    """

    def __init__(self, mapping_dict_path=None):
        self.mapping_dict = pd.read_csv(mapping_dict_path) if mapping_dict_path else None
        print("FAERS Data Ingestion Engine initialized.")

    def ingest(self, raw_faers_data):
        """
        Performs data cleaning, record alignment, and standardization.
        """
        print("Processing raw clinical pharmacovigilance records...")

        # Data cleaning: Drop records missing essential drug or event identifiers
        clean_df = raw_faers_data.dropna(subset=['drugname', 'pt'])

        # Identifier alignment: Map external drug names to internal gold-standard drug IDs
        if self.mapping_dict is not None:
            aligned_df = clean_df.merge(
                self.mapping_dict,
                left_on='drugname',
                right_on='fda_name',
                how='inner'
            )
            print(f"Successfully aligned {len(aligned_df)} drug-event records.")
            return aligned_df
        return clean_df

    def format_for_model(self, aligned_df):
        """
        Transforms aligned clinical records into the system-ready (Drug, SOC) matrix format.
        """
        print("Transforming clinical records to the internal feature space...")
        # Aggregation of Preferred Terms (PT) by internal drug identifiers
        formatted_df = aligned_df.groupby('internal_drug_id')['pt'].apply(list).reset_index()
        return formatted_df


# --- Demonstration Module ---
if __name__ == "__main__":
    # Simulated clinical dataset
    mock_faers_data = pd.DataFrame({
        'drugname': ['Fluoxetine', 'Clozapine', 'Diazepam', 'RandomDrugX'],
        'pt': ['Leukemia', 'Cardiac Arrhythmia', 'Sedation', 'UnknownEffect']
    })

    # Simulated internal drug mapping dictionary
    mock_mapping = pd.DataFrame({
        'fda_name': ['Fluoxetine', 'Clozapine', 'Diazepam'],
        'internal_drug_id': ['Fluoxetine', 'Clozapine', 'Diazepam']
    })

    # Engine execution
    engine = FAERSDataIngestionEngine()
    engine.mapping_dict = mock_mapping

    processed_results = engine.ingest(mock_faers_data)
    final_input = engine.format_for_model(processed_results)

    print("\nProcessed System-Ready Clinical Data:")
    print(final_input.head())
    print("\nData standardization completed successfully.")