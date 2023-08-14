#%% Importing libraries, loading classes and data
from sklearn.metrics import roc_auc_score, precision_score, recall_score, average_precision_score, cohen_kappa_score
from Funs import *
from Classess import *

MaxYearsOfHistory = 10 # Maximum number of time periods per patient
yearsToTarget = 1 # Number of time periods excluded from history. From observation window cutoff to outcome measurement. In the paper this was either 1 or 5.

import pickle
with open('./Data/TestTrainSets.pkl', 'rb') as f:
   [train_x, train_y, test_x, test_y] = pickle.load(f) # train_x, test_x have shape [patients, time, features]; train_y, test_y have shape [patients, label]
with open('./Data/Columns.pkl', 'rb') as f:
    [cols, categoric_cols, numeric_cols, static_cols] = pickle.load(f) # cols is the list of feature names; categoric_cols, numeric_cols, and static_cols are the names of the features of each type.

#%% Model hyperparameters
epochs = 50
batch_size = 32
propBinary = 0.5
callback = [tf.keras.callbacks.EarlyStopping(monitor='auc', min_delta = 0.001, patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor = "auc", patience = 5)
           ]
input_dims = [((MaxYearsOfHistory-yearsToTarget), len(categoric_cols)), ((MaxYearsOfHistory-yearsToTarget), len(numeric_cols)), (len(static_cols))] #categorics, numerics, statics

#%% Model definition and training
model = ARIADNEhr(input_dims=input_dims, rnn_type = "GRU", rnn_units = 256, useBiRNN_vars=False, useBiRNN_years=True, out_units=1, nn_layers = 1, dropout_rate=0.10, activation_func = "sigmoid")
model.compile(loss= two_loss_func,
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              metrics=[tf.keras.metrics.Precision(name = "precision"),tf.keras.metrics.AUC(name = "auc"),
                       tf.keras.metrics.AUC(curve = "PR", name = "auc_pr"), tf.keras.metrics.Recall(name = "recall")])
history = model.fit([train_x[:,:,np.isin(cols,categoric_cols)], train_x[:,:,np.isin(cols, numeric_cols)], train_x[:,0,np.isin(cols, static_cols)]],
                     train_y[:],
                     batch_size=batch_size,
                     epochs=epochs,
                     callbacks=callback)

model.save_weights('Weights', save_format='tf')

#%% Internal Validation
# Obtaining predictions on test set
test_y_pred, test_varsAttention, test_yearsAttention = model.predict([test_x[:,:,np.isin(cols,categoric_cols)], test_x[:,:,np.isin(cols, numeric_cols)], test_x[:, 0, np.isin(cols, static_cols)]])
# Calculate metrics using bootstrap on test set - overall
threshold = 0.5
metrics = [precision_score, roc_auc_score, average_precision_score, recall_score, cohen_kappa_score]
res_test_bootstrap = ci_auto(metrics, test_y, test_y_pred>threshold, nboots = 1000, Sample = "Test", HistoryYears = "Overall", Threshold = threshold, Output = "AllcauseMortality", PredictionWindow = "1 year")

# Calculate metrics using bootstrap on test set - stratified by number of time periods of history
HistoryInds = []
for i in range(test_x.shape[0]):
    HistoryInds.append(sum(test_x[i, :, 1] == -99)) # We first calculate the number of periods that are missing
HistoryInds = [MaxYearsOfHistory-x for x in HistoryInds] # We then calculate the number of periods with information

for i in range(max(HistoryInds)):
    if i>1:
        indiv = np.array([x==i for x in HistoryInds]) # We select the subset of patients with each number of periods with information available
        temp_y = test_y[indiv]
        res_test_bootstrap = pd.concat([res_test_bootstrap,
                                   ci_auto(metrics, temp_y, test_y_pred[indiv]>threshold, nboots = 1000, Sample = "Test",
                                           HistoryYears = i, Threshold = threshold, Output = "AllcauseMortality", PredictionWindow = "1 year")])

# Calculate metrics using bootstrap on test set - stratified by sex
males = np.squeeze(test_x[:,(MaxYearsOfHistory - yearsToTarget -1),np.where(cols == "sexe_H")]) == 1
females = np.squeeze(test_x[:,(MaxYearsOfHistory - yearsToTarget -1),np.where(cols == "sexe_H")]) == 0
res_test_bootstrap = pd.concat([res_test_bootstrap,
                           ci_auto(metrics, test_y[males], test_y_pred[males] > 0.5, nboots=1000, Sample="Test_males",
                                   HistoryYears='Overall', Threshold=threshold, Output="AllcauseMortality", PredictionWindow = "1 year")])
res_test_bootstrap = pd.concat([res_test_bootstrap,
                           ci_auto(metrics, test_y[females], test_y_pred[females] > 0.5, nboots=1000, Sample="Test_females",
                                   HistoryYears='Overall', Threshold=threshold, Output="AllcauseMortality", PredictionWindow = "1 year")])

# Calculate metrics using bootstrap on test set - stratified by age group
age_65_74 = np.squeeze(test_x[:,(MaxYearsOfHistory - yearsToTarget -1),np.where(cols == "age")]) <75
age_75_84 = np.logical_and(np.squeeze(test_x[:,(MaxYearsOfHistory - yearsToTarget -1),np.where(cols == "age")])>=75, np.squeeze(test_x[:,(10 - 1 - yearsToTarget),np.where(cols == "age")])<85)
age_p85 = np.squeeze(test_x[:,(MaxYearsOfHistory - yearsToTarget -1),np.where(cols == "age")]) >85
res_test_bootstrap = pd.concat([res_test_bootstrap,
                           ci_auto(metrics, test_y[age_65_74], test_y_pred[age_65_74] > 0.5, nboots=1000, Sample="Test_65to74",
                                   HistoryYears='Overall', Threshold=threshold, Output="AllcauseMortality", PredictionWindow = "1 year")])
res_test_bootstrap = pd.concat([res_test_bootstrap,
                           ci_auto(metrics, test_y[age_75_84], test_y_pred[age_75_84] > 0.5, nboots=1000, Sample="Test_75to84",
                                   HistoryYears='Overall', Threshold=threshold, Output="AllcauseMortality", PredictionWindow = "1 year")])
res_test_bootstrap = pd.concat([res_test_bootstrap,
                           ci_auto(metrics, test_y[age_p85], test_y_pred[age_p85] > 0.5, nboots=1000, Sample="Test_p85",
                                   HistoryYears='Overall', Threshold=threshold, Output="AllcauseMortality", PredictionWindow = "1 year")])

res_test_bootstrap.to_csv("Metrics.csv")

# Calibration plots
calibration_plot(test_y, test_y_pred, "All-cause mortality 1 years", "CalibrationPlot_AllcauseMortality_1")

#%% Attention Maps

mean_varsAttention = np.mean(test_varsAttention, axis = 0)
top_10_variables = np.argsort(mean_varsAttention[(MaxYearsOfHistory - yearsToTarget-1),:])[-10:]
variables = categoric_cols + numeric_cols
res_list = [variables[i] for i in top_10_variables]

mean_yearsAttention = np.mean(test_yearsAttention, axis = 0)

attentionMap = mean_varsAttention[:, np.argsort(mean_varsAttention[(MaxYearsOfHistory - yearsToTarget-1),:])[-10:]]
attentionMap = np.column_stack((attentionMap, mean_yearsAttention))

fig=plt.figure(figsize=(6, 4), dpi=100)
ax = plt.axes()
im = ax.imshow(np.array(attentionMap), vmin=0)
plt.title("All-cause mortality 1 years ahead")
plt.xticks(ticks=np.arange(11), labels=res_list+["Time period"], rotation=90)
plt.yticks(ticks = np.arange(MaxYearsOfHistory-yearsToTarget), labels = [1,2,3,4,5, 6, 7, 8, 9])
plt.ylabel('Time period (year of follow-up)')
plt.xlabel('Variable')
cax = fig.add_axes([ax.get_position().x1+0.005,ax.get_position().y0,0.02,ax.get_position().height])
plt.colorbar(im, cax=cax) 
plt.savefig("Attention_AllcauseMortality_1.pdf", dpi=500, bbox_inches='tight')
