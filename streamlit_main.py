import streamlit as st
import pandas as pd
import geopandas as gpd
import gdown
import plotly.graph_objects as go
import plotly.express as px
import os
from PIL import Image


st.title("Projet  Energie")
st.sidebar.title("Sommaire")
pages=["Definition du Projet","Exploration", "DataVizualization", "Modélisation"]
page=st.sidebar.radio("Aller vers", pages)
