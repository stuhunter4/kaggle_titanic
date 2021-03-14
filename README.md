# kaggle_titanic
The Kaggle Titanic ML competition: build a predictive model that uses passenger data to predict likelihood of survival.

<p><strong>model_01_LogisticRegression.ipynb: </strong>Feature selection and hyperparameter tuning effective, but since I dropped a NaN row from the test.csv data the submission file was incorrect.</p>
<p><strong>model_01a_LogisticRegression.ipynb: </strong>First submission with corrected test.csv data. StandardScaler not applied.  Public Score: 0.0.76315</p>
<p><strong>model_01b_LogisticRegression.ipynb: </strong>Scaled data, then loaded the saved model from initial hyperparameter tunining in model_01_LogisticRegression.ipynb.  Public Score: 0.0.77751</p>
<p><strong>model_01c_LogisticRegression.ipynb: </strong>Scaled data, then created LogisticRegression model with parameters determined from first notebook, model_01_LogisticRegression.ipynb.  Public Score: 0.0.77511</p>