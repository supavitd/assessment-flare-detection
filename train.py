import argparse
from glob import glob
import os
import sys

import joblib
import numpy as np
from skimage.io import imread
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from tqdm import tqdm

from utils import ImageTransformer


models = [
    GridSearchCV(SVC(), {
        'kernel': ('linear', 'rbf', 'sigmoid'),
        'C': [0.1, 1, 10]
    }),
    GridSearchCV(KNeighborsClassifier(), {
        'n_neighbors': [3,5,7,9],
        'weights': ('uniform', 'distance')
    }),
    GridSearchCV(RandomForestClassifier(), {
        'n_estimators': [50, 100, 200, 300, 500],
    }),
]


def read_dataset(data_path):
    categories = ['good', 'flare']
    X = []
    Y = []
    for i, category in enumerate(categories):
        for img in glob(data_path+'/training/'+category+'/*'):
            raw_image = imread(img)
            X.append(raw_image)
            Y.append(i)
            
    return np.array(X),np.array(Y)


def create_pipeline(estimator):
    return Pipeline([
        ('image_transfomer', ImageTransformer()),
        ('scaler', StandardScaler()),
        ('pca', PCA(0.95)),
        ('estimator', estimator)
    ])


def train_test_model(X_train, X_test, y_train, y_test, estimator, metric):
    """
    Train and test our `estimator` using the `metric`.
    `metric` is a function which takes y_true as first arg
    and y_pred as second arg. All sklearn metrics have this signature.
    """
    clf = create_pipeline(estimator)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return metric(y_test, y_pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='train', 
                                    description='Train our lens flare detection model')
    parser.add_argument('Data', metavar='DATA_DIR', type=str, help='Path to the training data')
    args = parser.parse_args()
    data_path = args.Data
    if not os.path.isdir(data_path):
        print('The given path does not exist')
        sys.exit()

    X, y = read_dataset(data_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        random_state=14, 
                                                        shuffle=True)
    best_model = None
    best_accuracy = 0
    for m in tqdm(models, desc='Model training'):
        accuracy = train_test_model(X_train, X_test, y_train, y_test, m, accuracy_score)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = m

    # Now that we have our best model. We train using the whole data.
    print('Model picked. Final training...')
    clf = create_pipeline(best_model)
    clf.fit(X, y)
    joblib.dump(clf, 'flare_detection.joblib')
    print('Training success!')

