# Head-CT-Age-Prediction
(1) Read the instrction of uploading data to import your own Kaggle dataset to Google Drive

# About new custom dataset
(2) shape of the outputs:  
patient_slice: torch.Size([64, 1, 224, 224, 28])
age_label: torch.Size([64, 1])

# Summary of some results
(3) Dataset is 100 patients which is trained on a network for around 80 epochs.
Simple CNN model: MAE = 17.3
