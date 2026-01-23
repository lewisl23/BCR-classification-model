# Explainable machine learning classification for rheumatoid arthritis using B cell receptor sequences

This is the MSc Bioinformatics dissertation repository that uses B cell receptor (BCR) sequences 
to train classification model for the classifcation between rheumatoid arthritis (RA) patients 
and the general public. For this project, 2 machine learning models (Logistic regression and Multilayer perceptron) 
are trained and tested for their accuracy in distinguishing the patients primarily on BCR sequences.

Steps of the project:

## 1. reads_processing_pipeline
- PRESTO analysis of Illumina raw reads
- IgBLAST analysis of quality-controlled alligned reads 

The reads proccessing pipeline is created using presto (version 0.7.4) and igblast 
(version 1.21.0) with OGRDB database (version 9). These tools are organised into a pipeline using snakemake with docker image acccess containing required tools and daatabse for reproducibility. 

## 2. exploratory_and_data_cleaning
- Data cleaning and filtering
- Train test split


## 3. model_training
- logistic regression
- MLP sigmoid
- Unused MLP softmax and logistic regression (scikit-learn)

## 4. SHAP_analysis
- Analysis of feature importance in MLP model

## Note
Airr reads dataset is collected from "In Human Autoimmunity, a Substantial Component of the B Cell Repertoire Consists of Polyclonal, Barely Mutated IgG+ve B Cells" by Graeme Cowan. \
doi: https://doi.org/10.3389/fimmu.2020.00395

