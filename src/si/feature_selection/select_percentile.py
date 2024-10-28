from typing import Callable

import numpy as np

from si.base.transformer import Transformer
from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification


class SelectPercentile(Transformer):

    def __init__(self, percentile, score_func: Callable = f_classification,**kwargs):

        super().__init__(**kwargs)
        self.score_func = score_func
        self.percentile = percentile
        self.F = None
        self.p = None

    def _fit(self, dataset: Dataset):

        self.F, self.p = self.score_func(dataset)
        return self

    def _transform(self, dataset: Dataset) -> Dataset:

        num_features_to_select = int(len(self.F) * self.percentile / 100)
        idxs = np.argsort(self.F)[-num_features_to_select:]
        
        features = np.array(dataset.features)[idxs]
        return Dataset(X=dataset.X[:, idxs], y=dataset.y, features=list(features), label=dataset.label)