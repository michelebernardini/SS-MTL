# A Semi-Supervised Multi-Task Learning Approach for Predicting Short-Term Kidney Disease Evolution

published in IEEE Journal of Biomedical and Health Informatics by M. Bernardini, L. Romeo, E. Frontoni and M.R. Amini

This work aims to propose a novel Semi-Supervised Multi-Task Learning (SS-MTL) approach for predicting short-term KD evolution on multiple General Practitioners’ Electronics Health Record data. We demonstrated that the SS-MTL approach can (i) capture the eGFR temporal evolution by imposing a temporal relatedness between consecutive time windows and (ii) exploit useful information from unlabeled patients when labeled patients are less numerous with a gain of up to 4.1 % in terms of Recall.

The code to replicate the Semi-Supervised Learning part (i.e., Self-Learning Algorithm [SLA]) of the SS-MTL approach can be found at the following link: https://github.com/aminim/tudor 

The code to replicate the Multi-task Leaerning part (i.e., Fused Group Lasso Progression Model with Logistic Loss [CFG]) can be found at the following link: https://github.com/jiayuzhou/MALSAR/tree/master/MALSAR

We tested the predictive performance of the SS-MTL approach on the novel mFIMMG dataset, publicly available at the following link: https://vrai.dii.univpm.it/content/mfimmg-dataset

Run [main_SSMTL.m](https://github.com/michelebernardini/SS-MTL/blob/master/main_SSMTL.m "main_SSMTL.m") to test the SS-MTL approach on pseudo-data.
