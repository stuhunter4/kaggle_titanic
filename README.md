# Kaggle | Titanic - "Machine Learning from Disaster"
The Kaggle Titanic ML competition: build a predictive model that uses passenger data to predict likelihood of survival.

Current progress: From 19 submissions, of the models compared the three with the same, highest public scores were the sklearn models, Logistic Regression, Linear SVC, and SVC (submission 06, 09, and 18 respectively).  For now, I will learn more about machine learning classification before proceeding with more tests.  Given that gender_submission.csv, the default submission that assumes women survive, has a score of 0.76555, the score from my models are not very good at making predictions.

        Best Public Score: 0.77990

This was done using the following features, selected using Random Forest Classifier, then scaled with Standard Scaler before training the models:

        Pclass
        Age
        Fare
        Family_Size (SibSp + Parch)
        Age*Class (another engineered feature)
        Fare_Per_Person (Fare / Family_Size)
        Sex_female
        Title_Mr (feature extracted from 'Name' data)

Missing data and feature engineering completed in the train_transformed, test_transformed, train_transformed_solutions, and test_transformed_solutions notebooks.  Of the CSV files created for model training, the most used for best results is:

                'resources/train_transformed_2.csv' and 'resources/test_transformed_2.csv'

With the use of sklearn's GridSearchCV, hyperparameters were tuned to increase the accuracy of each high scoring sklearn model.  The first model to reach my highest public score was a Logistic Regression model with these hyperparameters:

        LogisticRegression(C=5, max_iter=50, random_state=0, solver='newton-cg', tol=10)

- - -
### Notes on each notebook and submission, in order of creation:

* **model_01_LogisticRegression.ipynb:** Feature selection and hyperparameter tuning effective, but since I dropped a NaN row from the test.csv data the submission file was incorrect.
* **model_01a_LogisticRegression.ipynb:** First submission with corrected test.csv data. StandardScaler not applied.  *Hunter_submission01.csv, Public Score: 0.76315*
* **model_01b_LogisticRegression.ipynb:** Scaled data, then loaded the saved model from initial hyperparameter tunining in model_01_LogisticRegression.ipynb.  *Hunter_submission02.csv, Public Score: 0.77751*
* **model_01c_LogisticRegression.ipynb:** Scaled data, then created LogisticRegression model with parameters determined from first notebook, model_01_LogisticRegression.ipynb.  *Hunter_submission03.csv, Public Score: 0.77511*
* **model_01d_LogisticRegression.ipynb:** Scaled data, newly transformed dataset.  Created two submissions based on grid tuned hyperparameters, which predictably had the same evaluation scores.  It may improve my score to stick with unencoded Pclass data, or continuing my original approach to replacing NaN values in Age (with median, instead of a random int within one std of the mean).  *Hunter_submission04/05.csv, Public Scores: both 0.76555*
* **model_01e_LogisticRegression.ipynb:** Scaled data forward, transformed dataset yet again.  This time, Pclass is not encoded and Age still remains with the new modification (train/test_transformed_2).  This seems to be the sweet spot for engineering the features and tuning this model's parameters.  __*Hunter_submission06.csv, Public Score: 0.77990*__
* **model_02_KNN.ipynb:** First tests with a k-nearest neighbors model.  07 uses k=5, 08 uses k=13  *Hunter_submission07/08.csv, Public Scores: 0.76315, 0.76794*
* **model_03_LinearSVC.ipynb:** Two tests with linear support vector classification, with differently tuned hyperparameters.  First test ties with high score.  __*Hunter_submission09/10.csv, Public Scores: 0.77990, 0.77751*__
* **model_01f_LogisticRegression.ipynb:** Had a thought to include all features, without using Random Forest Classifier to select features.  Returned to the best performing model and hyperparameters for this test.  Score was inferior to selected features.  *Hunter_submission11.csv, Public Score: 0.70574*
* **model_04_NeuralNetwork.ipynb:** First Neural Network test, from TensorFlow.  100 neurons/functions  *Hunter_submission12.csv, Public Score: 0.75837*
* **model_05_DeepLearning.ipynb:** Two Deep Learning tests, the first with three hidden layers at 100 neurons each.  Second test with three hidden layers of 16 neurons; 'softmax' activation function and 'categorical_crossentropy' loss function used instead of 'sigmoid' and 'binary_crossentropy' options  *Hunter_submission13/14.csv, Public Scores: 0.77511, 0.76076*
* **model_04a_NeuralNetwork.ipynb:** Having settled on 'softmax' and 'categorical_crossentropy', another two tests are done with 100 neurons, then 80 neurons.  These are the most successful TensorFlow models I attempted to implement.  *Hunter_submission15/16.csv, Public Scores: 0.76315, 0.76794*
* **model_06_SGD.ipynb:** First test using a stochastic gradient descent classifier.  *Hunter_submission17.csv, Public Score: 0.72009*
* **model_07_SVC.ipynb:** First test using support vector classification.  Ties with the high scores from LinearSVC and LogisticRegression  __*Hunter_submission18.csv, Public Score: 0.77990*__
* **model_01g_LogisticRegression.ipynb:** Back to testing Logistic Regression after redoing features according to a notebook I was learning from.  *Hunter_submission19.csv, Public Score: 0.56459*
* **model_01h_LogisticRegression.ipynb:** Like the preceding notebook, uses transformed features according to another person's notebook tips.  This time, however, I used Random Forest to do feature selection.  The score improved.  *Hunter_submission20.csv, Public Score: 0.77272*