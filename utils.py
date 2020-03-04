import numpy as np
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.transform import resize
from sklearn.base import BaseEstimator, TransformerMixin


class ImageTransformer(BaseEstimator, TransformerMixin):
    """
    Transform RGB images into features as follows
    1. Turn them gray and resize to 64x64 by default
    2. Use Hog to extract features
    """
    def __init__(self, size=(64,64)):
        self.size = size

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        re_X = [resize(rgb2gray(x), self.size) for x in X]
        return np.array([hog(img) for img in re_X])

