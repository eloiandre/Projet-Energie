import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import gdown
import plotly.graph_objects as go
import plotly.express as px
import os
import pickle 
from sklearn.model_selection import train_test_split
from PIL import Image
from sklearn.base import BaseEstimator, TransformerMixin 
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, date
import scipy.stats as stats

st.set_page_config(layout="wide")

# Déclarer la classe heures_sinus
class heures_sinus(BaseEstimator, TransformerMixin):
    def __init__(self, datetime_col):
        self.datetime_col = datetime_col

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        self.datetime = pd.to_datetime(X[self.datetime_col])

        #Collections des annes,saison, mois, jours et heures depuis datetime
        X['annee'] = self.datetime.dt.year
        X['mois'] = self.datetime.dt.month
        X['jour'] = self.datetime.dt.day
        X['jour_ouvre']=X[self.datetime_col].dt.dayofweek<=5
        X['jour_sem']=X[self.datetime_col].dt.dayofweek

        #Transformation des heures minutes
        self.heures = self.datetime.dt.hour + self.datetime.dt.minute / 60 # passage des heures en dizaines de minutes
        X['heures_sin'] = np.sin(2 * np.pi * self.heures / 24)
        X['heures_cos'] = np.cos(2 * np.pi * self.heures / 24)
        X['heures'] = self.heures
        #X.drop(['date_heure'], axis=1, inplace=True)

        #Transformation des saisons
        self.saison=pd.cut(X['mois'], bins = [0,2,5,8,11,12],
                           labels = [1,2,3,4,1],
                           right = True, include_lowest=True, ordered = False).astype(int)
        X['saison_sin'] = np.sin(2 * np.pi * self.saison / 4)
        X['saison_cos'] = np.cos(2 * np.pi * self.saison / 4)
        X['saison']=self.saison
        return X

    def inverse_transform(self, X, y=None):
        self.sin_heures = X['heures_sin']
        self.cos_heures = X['heures_cos']
        heures = np.arctan2(self.sin_heures, self.cos_heures) * 24 / (2 * np.pi)
        heures = heures % 24
        minutes = (heures - np.floor(heures)) * 60
        heures_int = np.floor(heures).astype(int)
        minutes_int = np.floor(minutes).astype(int)
        X['date_heure'] = pd.to_datetime(
            X['annee'].astype(str) + '-' +
            X['mois'].astype(str) + '-' +
            X['jour'].astype(str) + ' ' +
            heures_int.astype(str) + ':' +
            minutes_int.astype(str)
        )
        X.drop(['annee', 'mois', 'jour', 'heures_sin', 'heures_cos','saison_sin','saison_cos'], axis=1, inplace=True)
        return X
    
@st.cache_data
def import_files():
    
    # Télécharger le fichier CSV principal
    url_csv = "https://drive.google.com/uc?export=download&id=1--2Tsgm3InoAqYkzKlvq0ylJ8JcBmjNU"
    output_csv = "data.csv"
    if not os.path.exists(output_csv):
        gdown.download(url_csv, output_csv, quiet=False)
    df = pd.read_csv(output_csv)
    #st.write("Fichier CSV principal téléchargé et chargé.")

    # Télécharger le fichier des températures
    url_temperature_csv = "https://drive.google.com/uc?export=download&id=1dmNMpWNhQuDyPxu0f4Un_wE38iDcOcuY"
    output_temperature_csv = "temperature.csv"
    if not os.path.exists(output_temperature_csv):
        gdown.download(url_temperature_csv, output_temperature_csv, quiet=False)
    temperature = pd.read_csv(output_temperature_csv)
    #st.write("Fichier CSV des températures téléchargé et chargé.")

    # Télécharger le fichier GeoJSON
    url_geojson = "https://raw.githubusercontent.com/eloiandre/Projet-Energie/main/regions.geojson"
    geojson = gpd.read_file(url_geojson)
    #st.write("Fichier GeoJSON téléchargé et chargé.")

    # Télécharger le fichier des features
    url_features = "https://raw.githubusercontent.com/eloiandre/Projet-Energie/main/feature_importances.csv"
    df_features = pd.read_csv(url_features, index_col=0)
    #st.write("Fichier des features téléchargé et chargé.")

    # URL du fichier Google Drive (utiliser l'ID de fichier dans le lien)

    # Télécharger le scaler depuis Google Drive
    url = 'https://drive.google.com/uc?id=17fVK3rUA47E6mO6GWHd4RxxTPHJ63il_'
    output = 'y_scaler.pkl'
    gdown.download(url, output, quiet=False)

    # Ouvrir le fichier et charger le scaler
    with open(output, 'rb') as f:
        y_scaler = pickle.load(f)
    #st.write('scaler telechargé')

    # Télécharger et charger le modèle
    url_model = "https://drive.google.com/uc?export=download&id=1-7_N8OZF4QfzDjAhVOjArFMrEcpL87z6"
    output_model = "model.pkl"
    if not os.path.exists(output_model):
        gdown.download(url_model, output_model, quiet=False)
    with open(output_model, 'rb') as file:
        model = pickle.load(file)


    #Télecharger ddes 5 premieres lignes de df
    url_head = "https://raw.githubusercontent.com/eloiandre/Projet-Energie/becae1f88ae5650712a044e77c86f3efe29d705d/df_head.csv"
    df_head = pd.read_csv(url_head,index_col=0)

    #Télecharger les NA
    url_na='https://raw.githubusercontent.com/eloiandre/Projet-Energie/main/df_na_percentage.csv'
    df_na=pd.read_csv(url_na)
    return df,df_head,df_na,geojson, temperature, df_features, model, y_scaler
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
def show_exploration():
    st.title('Exploration')
    st.info('Nous avons dans un premier temps extrait le fichier initial, auquel nous avons ensuite ajouté les températures trouvées sur le site https://meteo.data.gouv.fr.')
    st.write(temperature.head(10))
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
            st.write(df_head)
        st.write('memo : cree une table describe')
        #st.dataframe(df.describe().round(2))
        st.write("Toutes les variables sont de type numérique, à l'exception de la variable eolien et libelle_region. \
             Nous remarquons des écarts de consommation très importants, pouvant varier de 703 à 15 338 MW. \
             Sur la variable ech_physique, nous observons des valeurs positives et des valeurs négatives. Une valeur est positive lorsque \
             la région en question reçoit de l'électricité. Une valeur est négative lorsque la région transfère de l'électricité.")
        #st.dataframe(df_na)
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
    with st.expander('**Dataset température**'):
        """
        - ce fichier est le résultat d'une consolidation de plusieurs fichiers de température de météo France
        - changement de la variable date_heure au format datetime
        - passage de la région en type string

        """
        if st.checkbox('Afficher un extrait du DataFrame'):
            st.write(temperature.head(10))

@st.cache_data
def monthly_2022():### adaptation de la df pour le tracé de cartes
    df_2022 = df[df['annee'] == 2022].copy()
    df_2022['mois'] = pd.to_datetime(df_2022['date_heure']).dt.month
    df_2022.drop(columns=['date_heure'], inplace=True)
    productions_columns = ['thermique', 'nucleaire', 'eolien', 'solaire', 'hydraulique', 'pompage', 'bioenergies']
    df_2022 = df_2022[['code_insee_region', 'libelle_region', 'mois', 'consommation'] + productions_columns]
    df_2022['total_production'] = df_2022[productions_columns].sum(axis=1)
    df_2022_grouped = df_2022.groupby(['code_insee_region', 'libelle_region', 'mois']).sum().reset_index()
    return df_2022_grouped
def carte_prod(df_2022):
    # Calculer la production totale nationale pour chaque type de production
    total_production = df_2022['total_production'].sum()
    thermique_production = df_2022['thermique'].sum()
    nucleaire_production = df_2022['nucleaire'].sum()
    eolien_production = df_2022['eolien'].sum()
    solaire_production = df_2022['solaire'].sum()
    hydraulique_production = df_2022['hydraulique'].sum()
    bioenergies_production = df_2022['bioenergies'].sum()

    # La production par défaut affichée sera 'total_production'
    fig2 = px.choropleth(
        df_2022.reset_index(),  # Réinitialiser l'index pour que les colonnes soient accessibles
        geojson=geojson,  # Utiliser le fichier GeoJSON
        locations='code_insee_region',  # Colonne contenant les codes INSEE des régions
        featureidkey='properties.code',  # Le champ dans le GeoJSON correspondant aux codes INSEE
        color='total_production',  # Afficher la production totale par défaut
        color_continuous_scale='YlOrRd',
        hover_name='libelle_region',  # Afficher le nom de la région lors du survol
        labels={'total_production': 'Production Totale'}
    )

    # Limiter la carte à la France uniquement
    fig2.update_geos(
        projection_type="mercator",  # Utilisation de la projection Mercator adaptée à la France
        showcoastlines=False,  # Désactiver les lignes de côte
        showland=False,  # Désactiver l'affichage des terres en dehors du GeoJSON
        showframe=False,  # Désactiver le cadre de la carte
        fitbounds="locations",  # Adapter la carte aux frontières du GeoJSON (France)
        lataxis_range=[41, 51],  # Limiter la latitude (France métropolitaine)
        lonaxis_range=[-5, 10]   # Limiter la longitude (France métropolitaine)
    )

    # Ajouter des annotations (texte par défaut)
    annotation = dict(
        x=0.25, y=0.95, xref="paper", yref="paper",
        text=f"Production Totale : {total_production:,} MW",  # Valeur par défaut
        showarrow=False, font=dict(size=14, color="black"),
        align="left", bgcolor="white", bordercolor="black", borderwidth=2
    )

    fig2.update_layout(
        annotations=[annotation]
        
    )

    # Ajouter des boutons de filtre pour sélectionner les différentes sources de production
    fig2.update_layout(
        updatemenus=[
            dict(
                buttons=[
                    dict(
                        args=[
                            {"z": [df_2022['total_production']]},
                            {"annotations": [dict(annotation, text=f"Production Totale : {total_production:,} MW")]}
                        ],
                        label="Totale",
                        method="update"
                    ),
                    dict(
                        args=[
                            {"z": [df_2022['thermique']]},
                            {"annotations": [dict(annotation, text=f"Thermique : {thermique_production:,} MW")]}
                        ],
                        label="Thermique",
                        method="update"
                    ),
                    dict(
                        args=[
                            {"z": [df_2022['nucleaire']]},
                            {"annotations": [dict(annotation, text=f"Nucléaire : {nucleaire_production:,} MW")]}
                        ],
                        label="Nucléaire",
                        method="update"
                    ),
                    dict(
                        args=[
                            {"z": [df_2022['eolien']]},
                            {"annotations": [dict(annotation, text=f"Éolien : {eolien_production:,} MW")]}
                        ],
                        label="Éolien",
                        method="update"
                    ),
                    dict(
                        args=[
                            {"z": [df_2022['solaire']]},
                            {"annotations": [dict(annotation, text=f"Solaire : {solaire_production:,} MW")]}
                        ],
                        label="Solaire",
                        method="update"
                    ),
                    dict(
                        args=[
                            {"z": [df_2022['hydraulique']]},
                            {"annotations": [dict(annotation, text=f"Hydraulique : {hydraulique_production:,} MW")]}
                        ],
                        label="Hydraulique",
                        method="update"
                    ),
                    dict(
                        args=[
                            {"z": [df_2022['bioenergies']]},
                            {"annotations": [dict(annotation, text=f"Bioénergies : {bioenergies_production:,} MW")]}
                        ],
                        label="Bioénergies",
                        method="update"
                    )
                ],
                direction="down",  # Créer un menu déroulant
                showactive=True
            )
        ]
    )

    # Redimensionner l'image
    fig2.update_layout(
        autosize=False,
        width=1000,  # Largeur de l'image en pixels
        height=900,  # Hauteur de l'image en pixels pour l'étirement vertical
        margin={"r": 0, "t": 50, "l": 0, "b": 0},  # Réduire les marges pour maximiser l'espace
        
    )
    fig2.update_layout(
    title={
        'text': "Production electrique par source en 2022",
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    })

    st.plotly_chart(fig2, use_container_width=True)
def aggreg_period():
    df['mois'] = pd.to_datetime(df['date_heure']).dt.month
    df['jour_semaine'] = pd.to_datetime(df['date_heure']).dt.dayofweek
    df['heure'] = pd.to_datetime(df['date_heure']).dt.hour
    # Agrégations
    df_agg_mois = df[['consommation', 'temperature', 'mois']].groupby('mois').mean().reset_index()
    df_agg_jour_semaine = df[['consommation', 'temperature', 'jour_semaine']].groupby('jour_semaine').mean().reset_index()
    df_agg_heure = df[['consommation', 'temperature', 'heure']].groupby('heure').mean().reset_index()
    return(df_agg_mois,df_agg_jour_semaine,df_agg_heure)
def plot_conso_vs_temp(df_agg_mois,df_agg_jour_semaine,df_agg_heure):
        # Créer une figure
    fig = go.Figure()

    # Traces pour l'agrégation par mois
    fig.add_trace(go.Bar(
        x=df_agg_mois['mois'],
        y=df_agg_mois['consommation'],
        name='Consommation (Mois)',
        marker_color='skyblue',
        visible=True,  # Initialement visible
        yaxis='y1'
    ))
    fig.add_trace(go.Scatter(
        x=df_agg_mois['mois'],
        y=df_agg_mois['temperature'],
        name='Température (Mois)',
        mode='lines+markers',
        marker=dict(color='red'),
        line=dict(color='red'),
        visible=True,  # Initialement visible
        yaxis='y2'
    ))

    # Traces pour l'agrégation par jour de la semaine
    fig.add_trace(go.Bar(
        x=df_agg_jour_semaine['jour_semaine'],
        y=df_agg_jour_semaine['consommation'],
        name='Consommation (Jours de la semaine)',
        marker_color='skyblue',
        visible=False,  # Masqué au départ
        yaxis='y1'
    ))
    fig.add_trace(go.Scatter(
        x=df_agg_jour_semaine['jour_semaine'],
        y=df_agg_jour_semaine['temperature'],
        name='Température (Jours de la semaine)',
        mode='lines+markers',
        marker=dict(color='red'),
        line=dict(color='red'),
        visible=False,  # Masqué au départ
        yaxis='y2'
    ))

    # Traces pour l'agrégation par heure
    fig.add_trace(go.Bar(
        x=df_agg_heure['heure'],
        y=df_agg_heure['consommation'],
        name='Consommation (Heures)',
        marker_color='skyblue',
        visible=False,  # Masqué au départ
        yaxis='y1'
    ))
    fig.add_trace(go.Scatter(
        x=df_agg_heure['heure'],
        y=df_agg_heure['temperature'],
        name='Température (Heures)',
        mode='lines+markers',
        marker=dict(color='red'),
        line=dict(color='red'),
        visible=False,  # Masqué au départ
        yaxis='y2'
    ))

    # Mise en page avec menu déroulant
    fig.update_layout(
        title={
        'text': 'Consommation et Température selon différentes agrégations',
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'
        },
        xaxis=dict(title='Période'),
        yaxis=dict(
            title='Consommation (MW)',
            titlefont=dict(color='skyblue'),
            tickfont=dict(color='skyblue')
        ),
        yaxis2=dict(
            title='Température (°C)',
            titlefont=dict(color='red'),
            tickfont=dict(color='red'),
            overlaying='y',  # Superpose le deuxième axe y sur le premier
            side='right'  # Place le deuxième axe y sur la droite
        ),
        updatemenus=[  # Ajouter un menu déroulant
            dict(
                buttons=[
                    dict(
                        label='Par Mois',
                        method='update',
                        args=[{'visible': [True, True, False, False, False, False]},  # Montrer les traces par mois
                            {'title': 'Consommation et Température par Mois',
                            'xaxis': {'title': 'Mois'}}]
                    ),
                    dict(
                        label='Par Jours de la semaine',
                        method='update',
                        args=[{'visible': [False, False, True, True, False, False]},  # Montrer les traces par jours de la semaine
                            {'title': 'Consommation et Température par Jours de la semaine',
                            'xaxis': {'title': 'Jour de la semaine'}}]
                    ),
                    dict(
                        label='Par Heures',
                        method='update',
                        args=[{'visible': [False, False, False, False, True, True]},  # Montrer les traces par heures
                            {'title': 'Consommation et Température par Heures',
                            'xaxis': {'title': 'Heure'}}]
                    )
                ],
                direction='down',  # Menu déroulant vers le bas
                showactive=True
            )
        ]
    )
    st.plotly_chart(fig, use_container_width=True)
def carte_conso():
    # Make a copy of df to avoid modifying the global df
    df_grouped = df.groupby(['code_insee_region', 'libelle_region']).agg({'consommation': 'sum'}).reset_index()

    fig = px.choropleth(
        df_grouped,  # Use the grouped DataFrame
        geojson=geojson,  
        locations='code_insee_region',  
        featureidkey='properties.code',  
        color='consommation',  
        color_continuous_scale='YlOrRd',
        hover_name='libelle_region', 
        labels={'consommation': 'Consommation'}
    )

    # Limiter la carte à la France uniquement
    fig.update_geos(
        projection_type="mercator",  
        showcoastlines=False,  
        showland=False,  
        showframe=False,  
        fitbounds="locations",  
        lataxis_range=[41, 51],  
        lonaxis_range=[-5, 10]   
    )

    # Redimensionner l'image
    fig.update_layout(
        autosize=True,  # Utiliser autosize pour s'assurer que la carte s'adapte bien
        width=1000,  
        height=900,  
        margin={"r":0,"t":50,"l":0,"b":0},  # Laisser de la marge en haut pour le titre
        title={
            'text': "Consommation electrique en 2022",
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )

    # Affichage dans Streamlit
    st.plotly_chart(fig, use_container_width=True)
def show_data_viz():
    st.write('### DataVisualization2')
    conso_temp()
    st.write('En été la consommation suit un cycle defini par les jours ouvrés et jours de weekend plutot stable. En hiver la consommation et en opposition avec la temperature,\
              une vague de froid en janvier 2021 engendre un pic de consommation.')
    st.write('Inversemet en janvier 2022 une vague de chaleur engendre une baisse conséquente de la consommation.')
    carte_conso()
    carte_prod(monthly_2022())
    a, b, c = aggreg_period()
    plot_conso_vs_temp(a,b,c)
    plot_conso_region()
    plot_box_energie_conso()
def conso_temp():
    # Calcul des moyennes nationales par date/heure
    df_national = df[['date_heure', 'consommation', 'temperature']].groupby('date_heure').mean()
    df_national.reset_index(inplace=True)
    
    # Ajout des colonnes pour le lissage sur 24 heures (48 intervalles de 30 minutes)
    df_national['mois'] = pd.to_datetime(df_national['date_heure']).dt.month
    df_national['consommation_lisse'] = df_national['consommation'].rolling(window=48).mean()
    df_national['temperature_lisse'] = df_national['temperature'].rolling(window=48).mean()

    # Création de la figure avec deux axes Y
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_national['date_heure'], y=df_national['consommation_lisse'],
                             mode='lines', name='Consommation',
                             line=dict(color='blue'),
                             yaxis='y1'))  
    fig.add_trace(go.Scatter(x=df_national['date_heure'], y=df_national['temperature_lisse'],
                             mode='lines', name='Température',
                             line=dict(color='red'),
                             yaxis='y2'))  # Associe cette courbe à l'axe y2

    # Mise à jour de la mise en page pour ajuster la largeur et afficher deux axes Y
    fig.update_layout(
        title='Consommation et Température',
        xaxis=dict(title='Date/Heure'),
        yaxis=dict(
            title='Consommation (MW)',
            titlefont=dict(color='blue'),
            tickfont=dict(color='blue')
        ),
        yaxis2=dict(
            title='Température (°C)',
            titlefont=dict(color='red'),
            tickfont=dict(color='red'),
            overlaying='y',  # Superpose le deuxième axe y sur le premier
            side='right'  # Place le deuxième axe y sur la droite
        ),
        legend=dict(x=0.1, y=1.1, orientation='h'),
        autosize=False,
        width=1200,  # Augmentez la largeur du graphique
        height=500,  # Ajustez la hauteur du graphique si nécessaire
        
    )

    # Afficher la figure dans l'application Streamlit
    st.plotly_chart(fig, use_container_width=True)
def tableaux_modeles():
    data = {
        "Modèle": ["RandomForestRegressor", "XGBoostRegressor", "Decision Tree Regressor", "Régression linéaire"],
        "Taille test": [0.3, 0.3, 0.3, 0.3],
        "Max Depth": [10, 10, 10, None],
        "n_estimators": [200, 200, None, None],
        "MSE": [187145, 40173, 288024, 1374712.81],
        "RMSE": [432.60, 200.43, 536.67, 1172.48],
        "MAE": [318.53, 133.65, 408.60, 861.51],
        "R2 score Train": [0.957, 0.994, 0.934, 0.679],
        "R2 score Test": [0.956, 0.990, 0.933, 0.682]
    }
    # Créer un DataFrame
    df_results = pd.DataFrame(data)
    # Afficher le DataFrame dans Streamlit
    st.dataframe(df_results)
def plot_feature_importance():
    fig = go.Figure()
    df_sorted=df_features.sort_values(by=['Importance'],ascending=True)
    fig.add_trace(go.Bar(
        y=df_sorted.Feature,
        x=df_sorted.Importance,
        orientation='h'
    ))

    fig.update_layout(
        xaxis_title="Importance",
        yaxis_title="Feature",
        title="Feature Importances",
        margin=dict(l=150, r=50, t=50, b=50),
        width=800,
        height=600,
    )
    st.plotly_chart(fig, use_container_width=True)
def split_dataset(df):
    col_to_keep=['code_insee_region','date_heure','consommation','temperature']
    df=df[col_to_keep]
    df.date_heure=pd.to_datetime(df.date_heure)
    target='consommation'
    X=df.drop(target,axis=1)
    y=df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=151)
    return(X_train, X_test, y_train, y_test)
def plot_comparaison(y_test, y_pred, num_values=50):
    # Créer un DataFrame avec y_test et y_pred
    df_result = pd.DataFrame({
        'y_test': y_test,
        'y_pred': y_pred
    })

    # Prendre les 50 premières valeurs
    df_scatter = df_result.head(num_values).reset_index()

    # Créer la figure
    fig = go.Figure()

    # Ajouter les points de y_test
    fig.add_trace(go.Scatter(
        y=df_scatter['y_test'],
        x=df_scatter.index,
        mode='markers',
        marker=dict(color='blue'),
        name='y_test'
    ))

    # Ajouter les points de y_pred
    fig.add_trace(go.Scatter(
        y=df_scatter['y_pred'],
        x=df_scatter.index,
        mode='markers',
        marker=dict(color='red'),
        name='y_pred'
    ))

    # Mettre à jour le layout de la figure
    fig.update_layout(
        title=f'Écart entre test et prévisions pour les {num_values} premières valeurs',
        xaxis_title='Echantillon',
        yaxis_title='Valeur s(MW)'
    )

    # Afficher la figure
    st.plotly_chart(fig, use_container_width=True)
def create_result_df(y_test,y_pred,X_test):
    df_result=pd.concat([y_test.round(0),y_pred],axis=1)
    df_result = df_result.rename(columns={'y_test': 'consommation'})
    df_result = df_result.merge(df, how='left', left_index=True, right_index=True)
    st.write(df_result.head())
    col_to_keep=['y_pred','consommation_x','code_insee_region','date','heure','date_heure']
    df_result=df_result[col_to_keep]
    df_result = df_result.rename(columns={'consommation_x': 'consommation','y_pred':'prevision'})
    df_result['date_heure'] = pd.to_datetime(df_result['date_heure'])
    return(df_result)
def intro_model(X_train,y_train):
    st.write('### Modéles :')
    st.write('## Objectif : Prédire la consommation par région')
    st.write('# Tableau comparatif de nos modéles :')
    tableaux_modeles()
    st.write("Les modéles Random Forest , XGboost et Decision Tree ont les meilleures performances. Mais le score trop élevé des deux premiers\
             ressembe à du suraprentissage. Nous selectionons donc le 'Decision Tree Regressor' pour son score un peu plus faible et sa simplicitée")
    st.write('Notre modéles prend en compte la region, le temps et la temperature pour estimer la consommation regionale.')
    # Afficher X_train dans la première colonne
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("Exemple de nos données d'entrainement:")
        st.write(X_train.head())

    # Afficher y_train dans la seconde colonne
    with col2:
        st.write('Cible :')
        st.write(y_train.head())

    st.write("## Transformation des données :" )
    st.write("Pour que notre modéle considére les heures comme cyclique nous effectuons une transformation sinusoidale des heures:")
    image_heures_sin=Image.open('heures.jpg')
    st.image(image_heures_sin,width=500)

    st.write("# Pipeline : ")
    image_pipeline=Image.open('pipeline.png')
    st.image(image_pipeline)
    st.write("Le pipeline extrait l'année, la saison, le moi et le jour de la semaine de la variable 'date_heure' et decompose les heures en cos et sin.\
              Les heures et la temperatures sont considérés comme des variables numeriques, les autres variables sont considérés comme categorielle.\
             Le pipeline prévoit aussi une decomposition des saisons en cosinus et sinus comme pour les heures mais ces variables n'ont pas donné de bons\
              résultats lors de l'apprentissage et ne seront pas utilisées.   ")
def prediction(X_test,y_train,y_test):
    # Mise à l'échelle de y_train et y_test
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).ravel()

    # Prédiction 
    y_pred_scaled = model.predict(X_test)

    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_pred = pd.Series(y_pred, index=X_test.index, name='y_pred')
    y_test = pd.Series(y_test.values, index=X_test.index, name='y_test')

    df_result=create_result_df(y_test,y_pred,X_test).round(0)
    
    return df_result
def reel_vs_predict_interactive(df_result):
    # Échantillonner 1000 lignes pour alléger la charge
    df_result_sample = df_result.sample(n=1000, random_state=42)

    # Extraire les mois, jours de la semaine et heures
    df_result_sample['mois'] = df_result_sample['date_heure'].dt.month
    df_result_sample['jour_semaine'] = df_result_sample['date_heure'].dt.dayofweek
    df_result_sample['heure'] = df_result_sample['date_heure'].dt.hour

    # Créer une figure vide initiale
    fig = go.Figure()

    # Ajouter les barres pour les mois (valeur initiale)
    fig.add_trace(go.Bar(x=df_result_sample['mois'], y=df_result_sample['consommation'], name='Consommation', marker=dict(color='light blue')))
    fig.add_trace(go.Bar(x=df_result_sample['mois'], y=df_result_sample['prevision'], name='Prévision', marker=dict(color='blue')))

    # Créer le menu déroulant pour choisir le type de graphique
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=[
                    dict(
                        args=[{'x': [df_result_sample['mois'], df_result_sample['mois']], 
                               'y': [df_result_sample['consommation'], df_result_sample['prevision']]}],
                        label="Par Mois",
                        method="restyle"
                    ),
                    dict(
                        args=[{'x': [df_result_sample['jour_semaine'], df_result_sample['jour_semaine']], 
                               'y': [df_result_sample['consommation'], df_result_sample['prevision']]}],
                        label="Par Jour de la Semaine",
                        method="restyle"
                    ),
                    dict(
                        args=[{'x': [df_result_sample['heure'], df_result_sample['heure']], 
                               'y': [df_result_sample['consommation'], df_result_sample['prevision']]}],
                        label="Par Heure",
                        method="restyle"
                    )
                ],
                direction="down",
                showactive=True
            )
        ]
    )

    # Mettre à jour les éléments de mise en page
    fig.update_layout(
        title='Consommation réelle vs prédite par différentes dimensions',
        xaxis_title='Choix',
        yaxis_title='Consommation (MW)',
        legend_title_text='Type',
        barmode='group',
        width=800,
        height=500
    )

    # Afficher le graphique dans Streamlit
    st.plotly_chart(fig, use_container_width=True)


    # Échantillonner 1000 lignes pour alléger la charge
    df_result_sample = df_result.sample(n=1000, random_state=42)

    # Extraire l'heure depuis la colonne date_heure
    df_result_sample['heure'] = df_result_sample['date_heure'].dt.hour

    # Fondre les colonnes 'consommation' et 'prevision' pour la comparaison
    df_melted = pd.melt(df_result_sample, id_vars=['heure'], value_vars=['consommation', 'prevision'],
                        var_name='Type', value_name='Consommation')

    # Créer le graphique en barres
    fig = px.bar(df_melted, x='heure', y='Consommation', color='Type', barmode='group',
                 labels={'heure': 'Heure de la Journée', 'Consommation': 'Consommation (MW)'},
                 title='Consommation réelle vs prédite par heure de la journée')

    # Ajuster les paramètres du graphique
    fig.update_layout(
        title='Consommation réelle vs prédite par heure de la journée',
        xaxis_title='Heure de la Journée',
        yaxis_title='Consommation (MW)',
        legend_title_text='Type'
    )

    # Afficher le graphique dans Streamlit
    st.plotly_chart(fig, use_container_width=True)
def plot_prediction_vs_real(df_result):
    # Ajouter une colonne pour les noms de région à partir du dictionnaire
    df_result['region'] = df_result['code_insee_region'].map(region_dict)

    # Créer un graphique scatter avec plotly
    fig = px.scatter(
        df_result, 
        x='prevision', 
        y='consommation', 
        color='region', 
        labels={
            'prevision': 'Prédiction (MW)',
            'consommation': 'Consommation Réelle (MW)',
            'region': 'Région'
        },
        title='Prédiction vs Consommation Réelle par Région',
        color_continuous_scale='Viridis',
        template='plotly_white'
    )

    # Ajouter la ligne diagonale en rouge
    min_value = min(df_result['consommation'].min(), df_result['prevision'].min())
    max_value = max(df_result['consommation'].max(), df_result['prevision'].max())

    fig.add_trace(
        go.Scatter(
            x=[min_value, max_value], 
            y=[min_value, max_value],
            mode='lines',
            line=dict(dash='dash', color='red'),
            showlegend=False
        )
    )

    # Afficher le graphique dans Streamlit
    st.plotly_chart(fig, use_container_width=True)
def plot_residus(df_result):
    #reduction de la taille des données:
    df_result = df_result.sample(n=1000, random_state=42)
    y_test=df_result['consommation']
    y_pred=df_result['prevision']
    residuals=df_result['residus']
    # Créer le subplot 2x2
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Graphique de Dispersion des Résidus",
                                                        "Histogramme des Résidus",
                                                        "Comparaison Valeurs Réelles vs. Prédites",
                                                        "QQ Plot des Résidus"))

    # 1er graphique : Dispersion des résidus
    fig.add_trace(go.Scatter(x=y_pred, y=residuals, mode='markers',
                             name='Résidus', marker=dict(color='blue')),
                  row=1, col=1)
    # Ligne horizontale à y=0
    fig.add_trace(go.Scatter(x=[min(y_pred), max(y_pred)], y=[0, 0],
                             mode='lines', line=dict(color='red', dash='dash'),
                             showlegend=False), row=1, col=1)

    # 2e graphique : Histogramme des résidus
    fig.add_trace(go.Histogram(x=residuals, nbinsx=100, name='Résidus',
                               marker_color='green'), row=1, col=2)

    # 3e graphique : Comparaison Valeurs Réelles vs. Prédites
    fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers',
                             name='Réelles vs Prédites', marker=dict(color='purple')),
                  row=2, col=1)
    # Ligne diagonale
    fig.add_trace(go.Scatter(x=[min(y_test), max(y_test)], y=[min(y_test), max(y_test)],
                             mode='lines', line=dict(color='red', dash='dash'),
                             showlegend=False), row=2, col=1)

    # Quantiles théoriques de la distribution normale
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")

    # Points du QQ plot
    fig.add_trace(go.Scatter(x=osm, y=osr, mode='markers', name='QQ Plot', marker=dict(color='orange')), row=2, col=2)
    
    # Ligne de référence dans le QQ plot
    fig.add_trace(go.Scatter(x=osm, y=slope * osm + intercept, mode='lines', 
                             line=dict(color='red', dash='dash'), showlegend=False), row=2, col=2)
    # Mettre à jour la mise en page
    fig.update_layout(height=800, width=1000, title_text="Analyse des Résultats du Modèle",
                      showlegend=False)

    # Afficher le graphique dans Streamlit
    st.plotly_chart(fig, use_container_width=True)
def plot_box_energie_conso():
    # Réduire les données si elles sont trop volumineuses (par exemple, prendre un échantillon)
    df_sampled = df.sample(frac=0.1)  # Prendre un échantillon de 10 % des données pour accélérer le rendu

    # Créez une liste des données à tracer avec l'échantillon
    data = [
        go.Box(y=df_sampled['consommation'], name='Consommation', width=0.8),
        go.Box(y=df_sampled['thermique'], name='Thermique', width=0.8),
        go.Box(y=df_sampled['nucleaire'], name='Nucléaire', width=0.8),
        go.Box(y=df_sampled['solaire'], name='Solaire', width=0.8),
        go.Box(y=df_sampled['eolien'], name='Éolien', width=0.8),
        go.Box(y=df_sampled['hydraulique'], name='Hydraulique', width=0.8),
        go.Box(y=df_sampled['bioenergies'], name='Bioénergies', width=0.8)
    ]

    # Créez la figure
    fig = go.Figure(data=data)

    # Mise à jour de la mise en page
    fig.update_layout(
        title={
            'text': 'BOXPLOT Energie vs Consommation en MW',
            'x': 0.5,
            'xanchor': 'center',
        },
        xaxis_title='Type',
        yaxis_title='MW',
        xaxis=dict(
            tickvals=list(range(len(data))),
            ticktext=['Consommation', 'Thermique', 'Nucléaire', 'Solaire', 'Éolien', 'Hydraulique', 'Bioénergies'],
            tickangle=-25
        ),
        boxmode='group',
        height=800,
        width=800
    )
    st.plotly_chart(fig, use_container_width=True)

def show_model():
    X_train,X_test,y_train,y_test = split_dataset(df)
    intro_model(X_train,y_train)

    df_result=prediction(X_test,y_train,y_test)
    plot_comparaison(df_result['consommation'],df_result['prevision'])

    st.write('## Feature Importance :')
    #st.write(df_features.head())
    plot_feature_importance()
    split_dataset(df)
    
    #courbe d'aprentissage
    courbe_apprentissage=Image.open('Courbe_apprentissage.jpg')
    st.image(courbe_apprentissage)

    #consommation vs prediction
    reel_vs_predict_interactive(df_result)

    # etude des residus
    st.write('# Etude des residus')
    # residus exprimés en pourcentage
    df_result['residus']=((df_result['consommation']-df_result['prevision'])/df_result['consommation'])*100
    st.write(df_result.head())
    st.write("### on va surement remplacer par l'image du rapport" )
    plot_residus(df_result)
def show_conclusion():
    st.write("### Conclusion")
    st.write("\n")
    st.write("\n")

    st.write("""
                          
     La capacité de puissance électrique instantanée délivrable en France est d’environ 150GW. 
     Sur la période d’étude, la consommation maximum instantanée en France a été de 88.5GW. 
     Notre modèle a un résidu avec une médiane de 2GW (4GW maximum dans quelques cas). 
             
     Nous pouvons donc conclure qu'actuellement, les risques de blackout sont écartés, bien que des pics de consommation puissent être observés, 
      particulièrement durant les mois d'hiver.    
    """)  
          
     

    st.write("\n")
    st.write ("""
        Notre modèle Decision Tree Regressor montre une bonne capacité de généralisation avec un R² élevé. Les écarts moyens entre les prédictions 
        et les valeurs réelles sont raisonnables, mais certaines erreurs significatives suggèrent qu'il serait nécessaire de réexaminer les valeurs
        extrêmes pour améliorer la précision globale.
        """)
    st.write("\n")
           
    st.write("#### Perspective d'évolution ")

    st.write(""" 
        Amélioration des prévisions : Intégrer les prévisions météorologiques fournies par des API de Météo France permettrait d'anticiper les fluctuations 
        de consommation d'énergie, particulièrement en fonction des variations de température ou des événements climatiques extrêmes.
             

        Optimisation de la production d’énergies renouvelables : L'intégration des sources intermittentes comme l’éolien et le solaire dans les modèles
        permettrait de mieux anticiper la production d'énergie renouvelable en fonction des conditions météorologiques (ensoleillement, vitesse du vent, etc.), 
        améliorant ainsi l'ajustement entre la production et la consommation d'énergie. 
             
        Approfondissement des analyses des erreurs : En tenant compte des effets météorologiques et des effets calendaires. Cela permettrait de mieux comprendre 
        les résidus, et de corriger ces biais pour affiner encore davantage les modèles de prédiction. 
    """)
    
    st.write("\n")
    st.write("\n")

    st.write("#### Critique ")

    st.write(""" 
             
        Un modèle de Machine Learning avec des séries temporelles aurait pu mieux capturer la saisonnalité et les tendances liées aux données historiques de 
        consommation d'énergie. Ces modèles sont particulièrement efficaces pour gérer les variations saisonnières et améliorer les prévisions à long terme.
        
     """)

def plot_conso_region():
    df_tot = df.groupby(['mois', 'libelle_region','annee'])['consommation'].sum().reset_index()

    fig = px.bar (df_tot,
                x ='mois',
                y = 'consommation',
                animation_frame ='annee',
                color = 'libelle_region',
                hover_name='libelle_region',
                labels={'mois': 'Mois', 'consommation': 'Consommation totale (MW)'}
                )


    fig.update_layout(plot_bgcolor= 'white')
    fig.update_layout(
        title={
            'text': "Conso totale par région sur la période 2020-2023",
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        })

    fig["layout"].pop("updatemenus")
    st.plotly_chart(fig, use_container_width=True)
regions_dict = {
    11: "Île-de-France",
    24: "Centre-Val de Loire",
    27: "Bourgogne-Franche-Comté",
    28: "Normandie",
    32: "Hauts-de-France",
    44: "Grand Est",
    52: "Pays de la Loire",
    53: "Bretagne",
    75: "Nouvelle-Aquitaine",
    76: "Occitanie",
    84: "Auvergne-Rhône-Alpes",
    93: "Provence-Alpes-Côte d'Azur"
}
def show_prediction():
    st.write("## Predictions")

    # Interface pour sélectionner la région avec un identifiant unique (key)
    selected_region_name = st.selectbox(
        "Sélectionnez une région :", 
        list(region_dict.values()), 
        key="region_selectbox"
    )
    selected_region_code = [code for code, name in region_dict.items() if name == selected_region_name][0]

    # Interface pour ajuster la température (curseur de -10 à 30) avec un identifiant unique
    selected_temperature = st.slider(
        "Ajustez la température (°C) :", 
        min_value=-10, max_value=30, value=15, 
        key="temperature_slider"
    )

    # Interface pour sélectionner la date avec un identifiant unique
    selected_date = st.date_input(
        "Sélectionnez une date :", 
        value=date.today(), 
        key="date_input"
    )

    # Sélection de l'heure (curseur avec pas de 30 minutes) avec un identifiant unique
    selected_time = st.slider(
        "Sélectionnez l'heure :", 
        min_value=0.0, max_value=23.5, step=0.5, value=12.0, 
        format="%.1f", 
        key="time_slider"
    )
    hours, minutes = divmod(selected_time * 60, 60)  # Convertit l'heure en heures et minutes


    # Combinaison de la date et de l'heure
    selected_datetime = datetime.combine(selected_date, datetime.min.time()) + timedelta(hours=hours, minutes=minutes)

    # Format final : 'YYYY-MM-DD HH:MM:SS+00:00'
    selected_datetime_formatted = selected_datetime.strftime("%Y-%m-%d %H:%M:%S+00:00")

    # Affichage du format de la date et heure pour le modèle
    #st.write(f"Date et heure formatées pour le modèle : {selected_datetime_formatted}")

    data={
        'code_insee_region':[selected_region_code],
        'date_heure':[selected_datetime_formatted],
        'temperature':[selected_temperature]
    }
    #st.write(data)
    data_df=pd.DataFrame(data)
    data_df['date_heure']=pd.to_datetime(data_df['date_heure'])
    #st.write (data_df.dtypes)
    #st.write(data_df)
    pred_scaled=model.predict(data_df)[0]
    pred=y_scaler.inverse_transform([[pred_scaled]])[0][0]
    st.write(f"Le {selected_date} à {int(hours):02}:{int(minutes):02} en {selected_region_name} :" )
    st.header(f"Prédiction : {round(pred,2)} Mw")


def main():
    st.title("Projet Energie")
    
    st.sidebar.title("Sommaire")
    pages=[ "👋 Définition du projet", "🔍Exploration des données", " 📊 Data visualisation", " 🧩 Modélisation", "🔮 Prédiction", "📌Conclusion"]
    page=st.sidebar.radio("Aller vers", pages)
    st.sidebar.markdown("""  
        <br><br>  
        **Cursus**: Data Analyst  
        **Format** : Bootcamp  
        **Mois** : Juillet 2024  
        <br>  
        **Membres du Groupe** :  
        <ul>
            <li>Léa HENRY</li>  
            <li>Pierre VALMONT</li>  
            <li>Siyamala ROLLOT</li>  
            <li>Eloi ANDRE</li>  
        </ul>
        """, unsafe_allow_html=True)
    
    if page ==pages[0]:
        show_definition()
    if page==pages[1]:
        show_exploration()
    if page==pages[2]:
        show_data_viz()
    if page==pages[3]:
        show_model()
    if page==pages[4]:
        show_prediction()
    if page==pages[5]:
        show_conclusion()

# debut du code
#importation de tous les fichiers necessaire
df,df_head,df_na,geojson,temperature,df_features,model,y_scaler=import_files()

#creaction d'un dictionnaire des ferions
region_dict = df.set_index('code_insee_region')['libelle_region'].to_dict()
# Télécharger la df_head
main()