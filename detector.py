import argparse
import os
import sys

import joblib
from skimage.io import imread

from utils import ImageTransformer

model_filename = 'flare_detection.joblib'


def predict(test_images):
    """
    Detect whether there is lens flare in the given images
    using our model `flare_detection.joblib.z`.
    """
    clf = joblib.load(model_filename)
    return clf.predict(test_images)    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('test_images', metavar='Test image', nargs='*')
    args = parser.parse_args()

    test_files = args.test_images

    if len(test_files) == 0:
        print('No images given')
        sys.exit()

    if not all([os.path.isfile(f) for f in test_files]):
        print('Cannot find image files')
        sys.exit()

    if not os.path.isfile(model_filename):
        print('No model found. Please train the model using train.py')
        sys.exit()

    X_test = [imread(img) for img in test_files]
    results = predict(X_test)
    print(*results, sep='\n')


