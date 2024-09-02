import streamlit as st
import pandas as pd
import geopandas as gpd
import gdown
import plotly.graph_objects as go
import plotly.express as px
import os
from PIL import Image
st.set_page_config(layout="wide")
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
def show_exploration():
    st.title('Exploration')
    st.info('Nous avons dans un premier temps extrait le fichier initial, auquel nous avons ensuite ajouté les températures trouvées sur le site [link] https://meteo.data.gouv.fr.')
    with st.expander('**Dataset initial**'):
        """
        Le fichier initial contient 32 colonnes et 2 108 840 lignes. Dans ce fichier, nous disposons, par demie heure et par région:
        - quantité d'électricité en MW consommée
        - quantité d'électricité en MW produite, par type d'énergie
        - les taux de couverture (TCO) par type d'énergie, en pourcentage
        - les taux de charge (TCH) par type d'énergie, en pourcentage
        - les échanges d'électricité entre régions, en MW
        """
    

        if st.checkbox('Afficher un extrait du DataFrame'):
            st.dataframe(df.head(10))
            st.dataframe(df.describe().round(2))
        st.write("Toutes les variables sont de type numérique, à l'exception de la variable eolien et libelle_region. \
             Nous remarquons des écarts de consommation très importants, pouvant varier de 703 à 15 338 MW. \
             Sur la variable ech_physique, nous observons des valeurs positives et des valeurs négatives. Une valeur est positive lorsque \
             la région en question reçoit de l'électricité. Une valeur est négative lorsque la région transfère de l'électricité.")
        st.dataframe(df.isna().sum()*100/len(df))
        st.write('Les variables TCO et TCH comportent beaucoup de manquants (entre 69 et 82%), idem pour les variables stockage.\
             Nous ne garderons pas ces variables pour la suite du projet')
        st.write('Les différentes actions effectuées sur ce fichier:')
        st.write('**Suppressions**')
        """
        - supression des données avant 2020 car manque de données tco et tch
        - suppression des colonnes vides: 'column_30', 'stockage_batterie', 'destockage_batterie','eolien_terrestre','eolien_offshore'
        - suppression des 12 premières lignes vides du dataframe
        - les doublons lors du passage en heures d'été ont été supprimés
        
        """
        st.write('**Conversions**')
        """
        - variable 'date_heure' en format datetime
        - variable eolien en float
        - variable code_insee en string
        
        """

        st.write('**Remplacements**')
        """
        - encodage de la colonne 'nature', puis remplacée par la variable 'definitif'
        - mise à zéro de la variable nucléaire pour les régions sans centrales : Ile de France, Pays de la Loire, Provence-Alpes-Côte-d'Azur, \
        Bretagne, Bourgogne Franche Comté
        - mise à zéro des NaN dans la variable pompage
        - gestion des données incohérentes: tch hydraulique > 200%

        """

        st.write('**Enrichissements**')
        """
        - ajout des colonnes année, mois, jour et jour de la semaine
        - ajout des colonnes saison et type_jour qui seront ensuite encodées
        
        """

df,geojson=import_files()
st.title("Projet2 Energie")
st.sidebar.title("Sommaire")
pages=["Definition du Projet","Exploration", "DataVizualization", "Modélisation"]
page=st.sidebar.radio("Aller vers", pages)
if page ==pages[0]:
    show_definition()
if page==pages[1]:
    show_exploration()