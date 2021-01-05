# This file has resusable funtions in Fbprophet and Arima model for forecasting purpose

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data(name):
    '''
    name: string, name of the csv file, file has to be in the current directory
 
    
    return
    ______
    df: dataframe
    
    '''
    df = pd.read_csv(name)
    return df
    


def prepare_data(df, ds_col, y_col):
    '''
    rename the data and target column into 'ds' and 'y'
    
    ds_col: string, column name
    y_col: string, column name
    '''
    df = df.copy()
    df = df.rename(columns = {ds_col:'ds', y_col:'y'})
    df['ds']= pd.to_datetime(df['ds'])
    
    return df




def group_by_date(df, date_type, start_date, end_date):
    '''
    group dataframe by date
    
    df: has column 'ds'
    
    '''
    df = df.copy()
    df['ds']= pd.to_datetime(df['ds']) # make sure 'ds' is in datetime datatype
    
    df = df[(df['ds'] >= start_date ) & (df['ds'] <= end_date)] 

    if date_type == "Y":
        df = df.groupby(df['ds'].dt.to_period('Y')).sum()
    elif date_type == "M":
        df = df.groupby(df['ds'].dt.to_period('M')).sum()
    else:
        df = df.groupby(df['ds'].dt.to_period('D')).sum()
        
   
    df = df.reset_index()
    df['ds']= pd.to_datetime(df.ds.astype(str))
    
    return df

def slice_df(df, start_date, end_date, selected_col=[]):
    '''
    selects the desired date frame and columns from the dataframe
    
    selected_col: list, list of column names
    '''
    df = df.copy()
    if len(selected_col) != 0:
        df = df[selected_col]
        
    mask = (df['ds'] >= start_date) & (df['ds'] <= end_date)
    return df[mask]





def train_test_split(df, train_end_date,test_end_date):
    '''
    df has a column of 'ds' which stores datetime variable
    '''
    train = (df['ds'] <= train_end_date)
    test = (df['ds'] > train_end_date) & (df['ds'] <= test_end_date )

    df_tr = df.loc[train]
    df_tst = df.loc[test]
    
    print("train shape",df_tr.shape)
    print("test shape",df_tst.shape)
    
    return df_tr, df_tst





def calculate_monthly_revenue(df, start_date, end_date):
    ''' 
    calculate monthly revenue from start_date to end_date

    '''
    df = df[(df['ds'] >= start_date ) & (df['ds'] <= end_date)] 
    df = df.groupby(df['ds'].dt.to_period('m')).sum() # group by 'YYYY-MM'
    return df





def mean_absolute_percentage_error(y_pred,y_true): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))



def calc_mape(df_pred, df_true, date_type, start_date, end_date):
    '''
    This function can first group by data and then calc mape
    
    
    df_pred: dataframe, has a column of 'yhat' that stores predicted 
             monthly revenue a column of 'ds' (datetime)
    df_true: dataframe, has a column of 'y' that stores true monthly  
             revenue, a column of 'ds' (datetime)
    '''
   
    df_pred = group_by_date(df_pred, date_type, start_date, end_date)
    df_true = group_by_date(df_true, date_type, start_date, end_date)
    return  mean_absolute_percentage_error(df_true['y'], df_pred['yhat'])


def make_dataframe(arr, df_true):
    '''
    convert prediction array to a dataframe 
    
    this function will be useful do convert the prediction data from arima model
    
    arr: numpy array
    df_true: pandas dataframe, which has the desired data frame as 'ds'
    '''
    if type(arr) is np.ndarray:
        df = pd.DataFrame({'ds':df_true.ds, 'yhat': arr})

    return df



def plot_train_and_test(df_tst, df_tr):
    '''
    plot line graphs of test data and training data into a graph
    
    df_pred: dataframe, has column 'ds' (Datetime variable), 'y'
    df_tr: dataframe, training data, has columns 'ds', 'y'
    '''
    f, ax = plt.subplots(figsize=(16,5)) #make empty graph
    df_tst.plot(kind='line',x='ds', y='y', color='green', label='Test', ax=ax)
    df_tr.plot(kind='line',x='ds',y='y', color='blue',label='Train', ax=ax)
    
    plt.legend()    
    plt.title(' y Train and Test')
    plt.show()


def plot_predict_and_actual(df_pred, df_tr):
    '''
    plot line graphs of prediction value and actual value into a graph
    
    df_pred: dataframe, has column 'ds' (Datetime variable), 'yhat'
    df_tr: dataframe, training data, has columns 'ds', 'y'
    '''
    
    f, ax = plt.subplots(figsize=(16,5)) #make empty graph
    df_pred.plot(kind='line',x='ds', y='yhat', color='orange', label='Prediction', ax=ax)
    df_tr.plot(kind='line',x='ds',y='y', color='blue',label='Actual', ax=ax)
    
    plt.legend()    
    plt.title(' y Forecast and Actual')
    plt.show()

    
    
    

def plot_predict_vs_actual(df_pred, df_true, date_type, start_date, end_date):
    '''
    plot line graphs of prediction value and actual value into a graph
    
    df_pred: has column 'ds' (Datetime variable), 'yhat'
    df_true: has columns 'ds', 'y'
    
    
    Note: Graph for yearly data is weird
    '''
    
    df_pred = group_by_date(df_pred, date_type, start_date, end_date)
    df_true = group_by_date(df_true, date_type, start_date, end_date)
    print(df_pred.shape, df_true.shape)
    
    f, ax = plt.subplots(figsize=(15,10)) #make empty graph
    df_pred.plot(kind='line',x='ds', y='yhat', color='orange', label='Prediction', ax=ax)
    df_true.plot(kind='line',x='ds',y='y', color='blue',label='Actual', ax=ax)
    
    
    plt.legend()    
    plt.title( start_date[:4] + ' y Forecast vs Actual')
    plt.show()



    

def save_to_file(df,col_names,  filename, groupby = False, date_type = None, start_date = None, end_date = None ):
    '''
    save data to csv file with columns:year, month, yhat
    
    df: the dataframe with predicted data
    col_names: list of strings, desired columns in the dataframe to save to output
    '''
    
    df = df.copy()
    df = df[col_names]
    if groupby:
        df = group_by_date(df, date_type, start_date, end_date)
        
    
    df['year'] = df.ds.dt.year # create a separate column for 'year'
    df['month'] = df.ds.dt.month # create a separate column for 'month'
    
    
    # choose the desired columns and save to file
    df[['year','month','yhat']].to_csv(filename, index = False)
    
    
    
    
    