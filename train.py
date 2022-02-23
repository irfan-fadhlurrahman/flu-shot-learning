import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline

from sklearn.multioutput import MultiOutputClassifier
from sklearn.multioutput import ClassifierChain
from xgboost import XGBClassifier

def impute(X):
    X = X.copy()

    for col in list(X.columns):
        if X[col].isnull().sum() > 0:
           X[col] = X[col].fillna(X[col].mode()[0])

    return X

def final_model(n=4):
    # read the dataset
    PATH = "/mnt/c/Project/flu-shot-learning/"
    features = pd.read_csv(PATH + 'dataset/training_set_features.csv')
    targets = pd.read_csv(PATH + 'dataset/training_set_labels.csv')
    test = pd.read_csv(PATH + 'dataset/test_set_features.csv')
    submission = pd.read_csv(PATH + 'dataset/submission_format.csv')

    # concat the features and targets
    train = pd.concat(
        [features.drop('respondent_id', axis=1),
        targets.drop('respondent_id', axis=1)],
        axis=1
    )

    # split train dataset
    X_train = train.drop(['h1n1_vaccine', 'seasonal_vaccine'], axis=1)
    y_train = train[['h1n1_vaccine', 'seasonal_vaccine']].values

    # test dataset
    X_test = test.drop('respondent_id', axis=1)

    # save features names
    feature_names = {}
    for feature in list(X_train.columns):
        feature_names[feature] = list(X_train[feature].unique())

    # impute missing values
    imputer = FunctionTransformer(impute)
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # convert the features into a dictionary
    X_train = X_train.to_dict(orient='records')
    X_test = X_test.to_dict(orient='records')

    # one-hot encode with DictVectorizer
    preprocessor = DictVectorizer(sparse=False)
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    # classifier
    clf = XGBClassifier(max_depth=6, random_state=1)

    # fit the model
    model = MultiOutputClassifier(clf)
    model.fit(X_train, y_train)

    # predict the test set labels
    y_pred_test = model.predict_proba(X_test)

    # add predictions to submission file
    submission['h1n1_vaccine'] = y_pred_test[0][:, 1]
    submission['seasonal_vaccine'] = y_pred_test[1][:, 1]
    submission.to_csv(PATH + f"submission/submission_{n}.csv", index=False)
    print(f"submission_{n} has been saved")

    return imputer, preprocessor, model, feature_names

def save_pipeline(pipeline_dict):
    PATH = "/mnt/c/Project/flu-shot-learning/saved-pipeline/"
    with open(PATH + "saved_pipeline.bin", "wb") as file_in:
         pickle.dump(pipeline_dict, file_in)
    print("the pipeline has been saved")

def main():
    # final model
    imputer, preprocessor, model, feature_names = final_model(n=4)
    pipeline_dict = {
        "imputer": imputer,
        "preprocessor": preprocessor,
        "model": model,
        "features": feature_names
    }

    # saved the pipeline and model
    save_pipeline(pipeline_dict)

if __name__ == '__main__':
    main()




























#end of code
