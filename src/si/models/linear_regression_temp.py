import numpy as np
from si.base.model import ModelA
from si.data.dataset import dataset
from si.metrics import mse

class RidgeRegression (Model):

    def __init__(self, l2_penality, alpha, max_iter: int, patience: int, scale: bool):

        self.l2_penality = l2_penality
        self.alpha = alpha
        self.max_iter = max_iter
        self.patience= patience
        self.scale = scale
        
    def _fit(self, Dataset):
        
        pass


    def _predict(self, Dataset):


        pass


    def _score(self, Dataset):
        return mse(self.theta,self.thera_zero)
    

    def cost(self, Dataset):


        pass

