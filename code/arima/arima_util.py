import pandas as pd 
import numpy as np 
import pmdarima as pm
from pmdarima import auto_arima
from pmdarima import model_selection
from statsmodels.tsa.statespace.sarimax import SARIMAX 


class Arima_Impl:
    '''
    This class implements Arima time series forecasting.
    '''
    
    def __init__(self, df_tr):
        '''
        df_tr: dataframe, training data
        '''
        self.df_tr = df_tr
    
    def find_best_orders(self, print_summary = True, **params):
        '''
        get the best order and seasonal order from the auto_arima funciton
        '''
        #get best orders using auto_arima function
        self.stepwise_fit = auto_arima(self.df_tr['y'], **params) 
        self.order = self.stepwise_fit.order
        self.seasonal_order = self.stepwise_fit.seasonal_order

        if print_summary:
            print(self.stepwise_fit.summary())
            
            
    def sm_train(self):   
        self.model = SARIMAX(self.df_tr['y'],  
                            order = self.order,  
                            seasonal_order = self.seasonal_order).fit() 
        
        
    def predict(self,n_of_months,col_name):
        
        '''
        n_of_months: integer, number of months of data to predict, ex: 12 (months)
        col_name: string, column name for pandas dataframe
        '''

        self.predictions = self.model.predict(start = len(self.df_tr['y']),  
                          end = len(self.df_tr['y'])-1 +  n_of_months,  
                          typ = 'levels').rename(col_name) 
        

