import streamlit as st
import pandas as pd
import geopandas as gpd
import gdown
import plotly.graph_objects as go
import plotly.express as px
import os
from PIL import Image

def import_files():
    # Use the direct download link from Google Drive
    #https://drive.google.com/file/d/1uW-GP29vVrHz-Whh5LYCaCDckuqnCbGj/view?usp=drive_link
    url_csv = "https://drive.google.com/uc?export=download&id=1uW-GP29vVrHz-Whh5LYCaCDckuqnCbGj"
    output_csv = "data.csv"
    gdown.download(url_csv, output_csv, quiet=False)
    df = pd.read_csv(output_csv)




import_files()
st.title("Heeey")
st.sidebar.title("Sommaire")
pages=["Definition du Projet","Exploration", "DataVizualization", "Modélisation"]
page=st.sidebar.radio("Aller vers", pages)
