import streamlit as st
import pandas as pd
import geopandas as gpd
import gdown
import plotly.graph_objects as go
import plotly.express as px
import os
from PIL import Image
@st.cache_data
def import_files():
    #Use the direct download link from Google Drive
    url_csv = "https://drive.google.com/uc?export=download&id=1--2Tsgm3InoAqYkzKlvq0ylJ8JcBmjNU"
    output_csv = "data.csv"

    gdown.download(url_csv, output_csv, quiet=False)
    df = pd.read_csv(output_csv)
    

    url_geojson = "https://raw.githubusercontent.com/eloiandre/Projet-Energie/main/regions.geojson"
    geojson = gpd.read_file(url_geojson)
    return(df,geojson)
def show_exploration():
    st.write('### Exploration')
    st.dataframe(df.head(10))
    st.write(f"Dimensions du DataFrame: {df.shape}")
    st.dataframe(df.describe())
df,geojson=import_files()
st.title("Projet Energie")
st.sidebar.title("Sommaire")
pages=["Definition du Projet","Exploration", "DataVizualization", "Modélisation"]
page=st.sidebar.radio("Aller vers", pages)
if page==pages[1]:
    show_exploration()