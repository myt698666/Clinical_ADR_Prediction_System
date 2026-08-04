# Random_forest_drug_target_adr_prediction
This repository contains all the data and code used within the paper 'An Interpretable Machine Learning Framework for Adverse Drug Reaction Prediction from Drug-Target Interactions'. The data and code within this repository contain all methods used and unused, the following is a breakdown of what is in each folder and how to break down this repository more simplistically.

## Folder Breakdown
The following has been explained in alphabetical order, there has been no set naming conventions so to avoid having to change code, the names have been left.

### ADR Statistical Analysis
This folder contains an unused method for final analysis, it covers the initial methods attempted to analyse ADR data from the Yellow Card Scheme (YCS) and at the start of the project where only SSRI drugs were to be looked at.

### ADR_sex_age_analysis
This was again an unused method to address how sex and age alter the ADR signals, this method showed differences in signals between different populations, however it was decided that overall signals covered the whole population. There is a code file to extract this and then all the data from when it was extracted.

### ADR_Summary
This folder provides the main ADR information used in the study that has been summarised from the data collection. There is a MedDRA key containing all the groupings of each ADR found within the yellow card scheme, all the ADR data up to when it was collected in January 2025 from when the YCS began. These two files provided the bulk of the ADR information. The other files contain analysis on this data. if the file name contains 'Normalized', this refers to the method that wasnt used, where the ADR count data was normalised with open prescribing data. All other data refers to analysis on different levels of the MedDRA hierarchy, like the files with 'grouped' in the title refers to the counts of all drugs and MedDRA categories. Then there are also disproportionality analysis results and a significance matrix, this matrix is what was used in the models for training and testing as the outputs

### All_ADR_Event_Files
This folder contains all Data in csv files for the 'event', 'case' and 'drug' data found for each drug on the YCS website. This has been put into one file in the previous folder but these are all separate here.

### Archive
This folder contains all old unused methods, useful in the building of the project but hold no relevance to the final outcome.

### chEMBL Data Extraction
During the start of the project, data originally was going to be taken from chEMBL, this folder contains all the code to extract many different values from chEMBL. The majority of these files are not required for the code to run however the information extracted for drugs can be useful for deeper understanding.

### Create Visualizations
This folder contains all the visualisations created for the paper and more to showcase differences within the data and outcomes of the models

### Data Extraction
This Folder contains all the code written to evaluate the ADR data taken from the YCS and how to extract all the data from the YCS. To extract the data from the YCS, the inspect tool was used on the website enabling the HTML code to be copied. Within the HTML code all the links to the IDAPS (ADR data) were extracted and then downloaded. This code can be re run, however if new drugs appear or the links to the IDAPS change then this code will not work. The links are currently static from when initially extracted, however runnning the code will get the most up to date data from the links extracted. The outputs of this code are in different folders are slightly messy, if more information is required, please email the authors.

### Database
This folder contains the final database that the models were trained on, including an attempted GAN to increase data size.

### Drug Activity
This folder contains drug information extracted from different sites, very little in this folder was used for the final output.

### Drug InChi Keys
This folder was the starting point for the project, enabling drug identifiers from different sites to be added to the YCS data. The InChi Keys and SMILES codes for the available drugs. There are some files in this folder out of date however, the All_drug_inchi prefixed files contain identifiers. The code also shows how to extract this.

### Drug Information
This folder provides for all the drugs where available the descriptive information in csv files. This was planned to be used in the final output however reduced performance when used. This provides common information important for drugs from pubchem.

### Drug Information Extraction
This folder contains the code to extract the drug information from the Pubchem API

### Drugs and Keys
This folder contains all the drugs keys that could be used for the project, and also the linking together of where all the information came from into one file. This file contains some missing rows where SMILES values were not found for the YCS.

### Drugs of Interest
This Folder links back the top feature importance scores to the drugs they react with. Importantly this also contains data which refers to the links between the drugs, targets and also the significant psychiatric disorder signals, this method under this folder could be done for any of the MedDRA categories, currently the best SOC model was psychiatric disorders hence the method was applied to this.

### Feature Importance
This folder contains all feature importance scores and ranking for all models created under both SOC and HLGT categories, the report only focusses on SOC.

### Full Processes
This Folder contains full processes, running this code with the whole folder downloaded will update all the proceses done, this may take time to follow all processes, potentially up to 24 hours or longer.

### Important Features Description
This folder contains the methods and data behind the validation of the top features predicted by the model against the DisGeNET database, this section is extremely important for the report in showing its usecase.

### InBirg DDPD Values
This folder contains an unused method where this database was attempted for extracting Cmax values and using these in combination with IC50 values for reactions.

### MACCS
This folder contains another unused method, this was for drug descriptions in the for of MACCS keys which provide a 166 bit values of what the drug has within it. this again reduced performance of the model hence was not used

### Open Prescribing
This folder covers the method of the open prescribing data extraction which wasn't used in the project due to reduction of data.

### Open Prescibing Data
This folder contains the data extracted and summarised from the open prescirbing data extracted, like in the previous folder this was unused in the final project

### Random Forests
This folder covers the machine learning of the project and all the model creation. This was an iterative process and all iterations of code have been included in this folder. The main file in this folder that was used was the RF5 Bayesian Optimisation. This proved to contain the best results and hence other experiments were conducted on this framework. but for the actual model RF5 Bayesian Optimisation is the main file. Also within the folders, saved models and hyperparameters values are there as well as other experiments values. There is a results folder which contains a summary of all the results from the machine learning.

### saved_models
This folder contains all of the saved models that were created in the project. 

### Sider Validation
This folder contains all the processes behind validating the difference expected between the spontateous reporting systems like the YCS and the clinical trials and drug packaging data found on SIDER. This contains the data and processes taken place.

### Timings 
this folder contains an unfinished analysis of how long each program takes and run together.

### UniProt Protein Data
This folder contains the process to extract information about the protein/genes used in the study. This method was for building a knowledgebase about the proteins rather than used in the models.

### Visuals
This folder contains some of the visuals used in the report. The visuals are out of date to the final report.

## File Break Down
Some files within this are very large so may need to be downloaded directly from http://stitch.embl.de/cgi/download.pl?UserId=bpjW7aLPmEbT&sessionId=XyCf9UzSBu3G. This takes you to the download page where you can access the chemical and targets of the chemicals/interactors. these files are around 580 MB for the interactions and 10 gb for the chemical ID files. these have been filtered but the high compute needed for filtering made it difficult to extract. If you need to run this again be prepared for it to take a long time. You will not need these files to replicate this dataset as the filtering has already been done. 

There is also a code debugging jupyter notebook, this contains all the debugging phases through the code. 

If anything else needs clarifying feel free to contact me.