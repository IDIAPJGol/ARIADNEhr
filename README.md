# ARIADNEhr
This GitHub repo presents the code needed to train and validate ARIADNEhr (Attention-based pRediction on longItudinAl Data iN Ehr). This model was published in "Carrasco-Ribelles, L.A., Cabrera-Bean, M., Khalid, S. et al.  Development of Attention-based Prediction Models for All-cause Mortality, Home Care Need, and Nursing Home Admission in Ageing Adults in Spain Using Longitudinal Electronic Health Record Data. J Med Syst 49, 17 (2025). https://doi.org/10.1007/s10916-024-02138-z". Please, cite accordingly.

*CodeToRun.py* includes all the code needed to create, train, and validate the model. Following the figures in that paper, the necessary input format is a tensor of dimensions *[Number_of_patients, Number_of_time_periods, Number_of_features]*: 
![input data format](https://github.com/idiapjgol/ARIADNEhr/blob/main/inputData.jpg?raw=true)



