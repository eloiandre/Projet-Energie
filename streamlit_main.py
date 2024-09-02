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
#fonction pour la page Définition
def show_definition():
    st.write('## Definition du projet :')
    st.write('« Constater le phasage entre la consommation et la production énergétique, au niveau national et au niveau régional (risque de black out notamment) »')
    st.info("Depuis 2022 en France, on parle de plus en plus de sécurité d'approvisionnement électrique. \
    Le parc nucléaire est de moins en moins renouvelé; laissant de la place au développement du parc à énergies renouvelables.")
    
    image_compo = Image.open('composition_nucleaire.png')
    st.image(image_compo)
    st .write('Depuis 2020, le poids du nucléaire est de 66,5% de la production globale. Les énergies renouvelables quant à elles comptent pour 25.1%.')

    piechart = Image.open('piechart.png')
    st.image(piechart)
    st.write("Les énergies renouvelables prennent de plus en plus de place. Une telle évolution des types d'énergies pose le risque de 'blackout', \
              à savoir des coupures d'approvisionnement sur tout ou partie du territoire. \
              Aujourd'hui, il est très difficile de stocker l'énergie. Il faut alors s'assurer que la production subvienne bien à la demande à chaque instant.")
    
    prod_type = Image.open('production_par_type.png')
    st.image(prod_type)
    st.write("Ce projet porte sur l'analyse de données extraites par une application nommée eCO²mix [link] https://www.rte-france.com/eco2mix. \
             Le défi est de pouvoir entraîner un modèle capable de prédire la consommation en éléctricité, par demie heure et par région.")
    st.markdown("""
    <style>
            .right-align {
			    text-align: right;
		    }
	</style>
	""", unsafe_allow_html=True)

    st.markdown(
    """
    Plusieurs aspects seront observés:
    - Analyse au niveau régional pour en déduire une prévision de consommation
    - Analyse par filière de production : énergie nucléaire / renouvelable
    - Focus sur les énergies renouvelables (où sont- elles implantées ?)
    """
    )
    st.markdown('<p class="right-align">Membres du groupe: Eloi Andre, Pierre Valmont, Siyamala Rollot, Léa Henry-Beaupied,  </p>', unsafe_allow_html=True)
    st.markdown('<p class="right-align">Date: Septembre 2024</p>', unsafe_allow_html=True)


df,geojson=import_files()
st.title("Projet2 Energie")
st.sidebar.title("Sommaire")
pages=["Definition du Projet","Exploration", "DataVizualization", "Modélisation"]
page=st.sidebar.radio("Aller vers", pages)
if page ==pages[0]:
    show_definition()
if page==pages[1]:
    show_exploration()