import pandas as pd 
import numpy as np 
from fbprophet import Prophet
from sklearn.model_selection import ParameterGrid

import code.tools as tl

def generate_all_params(params_grid):
    '''
    Generate all combinations of parameters
    '''
    
    all_params = ParameterGrid(params_grid)
  
    #for params in all_params:
        #print(params)
        
    print('Total Possible Models',len(list(ParameterGrid(params_grid))))
    return all_params


        
def hyperparam_tuning(all_params, df_tr, df_tst,  start_date, end_date, period, freq):
    '''
    This function finds the best parameter set for fbprophet model
    
    all_params: ParameterGrid object
    df_tr, df_tst: training data and testing data
    
    peroid: integer, predict period
    freq: 'D', 'MS', 'Y'
    
    
    start_date and end_date: string, tuning period
    

    '''
    MAPE_monthly_rev = [] # store the mape of monthly revenue or each combination of params
    min_MAPE = 1000

    
    # evaluate all parameters
    for params in all_params:
        prophet = Prophet_Impl(df_tr)  
        prophet.train(**params)
        prophet.predict(period,freq)

        #calculate monthly mape
        df_pred = tl.slice_df(prophet.forecast,start_date, end_date)
        mape = tl.calc_mape(df_pred, df_tst,'M',start_date, end_date)
        MAPE_monthly_rev.append(mape)

        #find min mape and best parameters
        if min_MAPE > mape:
            min_MAPE = mape
            best_params = params

    tuning_results = pd.DataFrame(all_params)
    tuning_results['monthly_revenue_mape'] = MAPE_monthly_rev



    print('best parameters', best_params)
    print('MAPE for monthly_revenue', min_MAPE)
    
    return best_params, tuning_results

        
        

class Prophet_Impl:
    
        
    def __init__(self, df_tr, df_tst = None):
        '''
        df_tr: dataframe, training data
        '''
        self.df_tr = df_tr
        self.df_tst =df_tst
        
    def train(self,**params):
        '''
        train the model with parameters
        '''
        self.model = Prophet(**params).fit(self.df_tr) #fit model with training data
        
        
    def predict(self, periods, freq):
        '''
        period: integer, length of future time, ex: 365 (days)
        freq: character, 'D', 'MS' (month start), 'Y'
        '''
        future = self.model.make_future_dataframe(periods= periods, freq = freq)
        self.forecast = self.model.predict(future)
        
        
        

 
        
        
   
        
