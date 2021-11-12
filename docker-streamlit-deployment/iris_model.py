# Import dependencies
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble  import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn.metrics import f1_score

class DataPrep:
    def __init__(self, iris_df):
        self.iris_df = iris_df

    def data_prep_for_model(self):
        # Convert the target character labels into numeric labels
        self.iris_df['species'] = self.iris_df.species.map( {'setosa': 0, 'virginica': 1, 'versicolor': 2} ).astype(int)
        
        # Shuffle the dataset
        self.iris_df = shuffle(self.iris_df, random_state=42)

        # Split data into test and train
        X_train, X_test, y_train, y_test = train_test_split(self.iris_df.drop(columns=['species']), self.iris_df['species'],test_size=0.10, random_state=42)

        return X_train, X_test, y_train, y_test

class BuildModel:
    classifier = RandomForestClassifier(n_estimators=10)
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test  = X_test
        self.y_train = y_train
        self.y_test = y_test

    def train_model(self):
        self.classifier.fit(self.X_train, self.y_train)
        cv_score    = cross_val_score(self.classifier, self.X_train, self.y_train, cv=5)
        y_pred      = self.classifier.predict(self.X_test)
        f1score     = f1_score(y_test, y_pred, average='weighted')
        
        return cv_score.mean(), cv_score.std(), f1score

    def save_model(self, model_name):
        self.model_name = model_name
        pickle.dump(self.classifier, open(self.model_name, 'wb'))
        return

if __name__ == "__main__":
   iris_df = pd.read_csv('./iris.csv')
   
   # Prepare data for modelling
   prepared_df = DataPrep(iris_df)
   X_train, X_test, y_train, y_test = prepared_df.data_prep_for_model()
   print('There are {} samples in the training set and {} samples in the test set'.format(X_train.shape[0], X_test.shape[0]))
   
   # Build Model
   iris_model = BuildModel(X_train, X_test, y_train, y_test)
   cv_score_mean, cv_score_std, f1score = iris_model.train_model()
   print("Accuracy : %0.2f (+/- %0.2f)" % (cv_score_mean, cv_score_std * 2))
   print("F1 Score : ", f1score)  

   if f1score > 0.85:
      iris_model.save_model('iris_model.pkl')
      print("Model Saved Sucessfully!")