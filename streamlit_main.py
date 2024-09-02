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
def carte_prod(df_2022, geojson):
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
        title="Production électrique par source"
    )

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
def conso_vs_temp(df_agg_mois,df_agg_jour_semaine,df_agg_heure):
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
        title='Consommation et Température selon différentes agrégations',
        xaxis=dict(title='Période'),
        yaxis=dict(
            title='Consommation (MWh)',
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
def carte_conso(geojson):

    df=df.groupby(['code_insee_region','libelle_region']).agg({'consommation':'sum'})

    fig = px.choropleth(
        df.reset_index(), 
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
        autosize=False,
        width=1000,  
        height=900,  
        margin={"r":0,"t":0,"l":0,"b":0}  
    )
    st.plotly_chart(fig, use_container_width=True)
def show_data_viz():
    st.write('### DataVisualization')
    conso_temp()
    st.write('En été la consommation suit un cycle defini par les jours ouvrés et jours de weekend plutot stable. En hiver la consommation et en opposition avec la temperature,\
              une vague de froid en janvier 2021 engendre un pic de consommation.')
    st.write('Inversemet en janvier 2022 une vague de chaleur engendre une baisse conséquente de la consommation.')
    carte_conso(geojson)
    #carte_prod(monthly_2022(),geojson)
    #a, b, c = aggreg_period()
    #conso_vs_temp(a,b,c)
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
            title='Consommation (MWh)',
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
df,geojson=import_files()
st.title("Projet2 Energie")
st.sidebar.title("Sommaire")
pages=["Definition du Projet","Exploration", "DataVizualization", "Modélisation"]
page=st.sidebar.radio("Aller vers", pages)
if page ==pages[0]:
    show_definition()
if page==pages[1]:
    show_exploration()
if page==pages[2]:
    show_data_viz()