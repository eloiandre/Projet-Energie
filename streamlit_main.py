import streamlit as st
import pandas as pd
import geopandas as gpd
import gdown
import plotly.graph_objects as go
import plotly.express as px
import os
from PIL import Image

def import_files():
    # Use requests to download the CSV file
    url_csv = "https://drive.google.com/uc?export=download&id=1uW-GP29vVrHz-Whh5LYCaCDckuqnCbGj"
    output_csv = "data.csv"
    
    response = requests.get(url_csv)
    if response.status_code == 200:
        with open(output_csv, 'wb') as file:
            file.write(response.content)
        df = pd.read_csv(output_csv)
    else:
        st.error("Failed to download CSV file.")
        df = None




import_files()
st.title("Heeey")
st.sidebar.title("Sommaire")
pages=["Definition du Projet","Exploration", "DataVizualization", "Modélisation"]
page=st.sidebar.radio("Aller vers", pages)
