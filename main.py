import streamlit as st
import pandas as pd
import numpy as np

header = st.beta_container()
dataset = st.beta_container()
features = st.beta_container()
modelTraining = st.beta_container()

DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
         'streamlit-demo-data/uber-raw-data-sep14.csv.gz')
@st.cache
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data

with header:
    st.title('Welcome to my data science project')
    st.text('In this project I look into the the interactions of taxis in NYC....')


with dataset:
    st.header('NYC Taxi Dataset')
    st.text('I found the dataset on blablabla.com....')
    # Create a text element and let the reader know the data is loading.
    data_load_state = st.text('Loading data...')
    # Load 10,000 rows of data into the dataframe.
    data = load_data(10000)
    # Notify the reader that the data was successfully loaded.
    data_load_state.text('Loading data...done!')
    st.write(data.head(10))
    df = pd.DataFrame(data)
    st.write(df.columns)

    st.map(df)

with features:
    st.header('The features I created')


with modelTraining:
    st.header('Time to train model')
    st.text('Here you can get the hyperparamters of the model to see how the  performance can be improved')

