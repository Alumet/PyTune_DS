import streamlit as st
from PIL import Image


def create():
    st.title('PYTUNE')
    st.header('Introduction')

    st.text("Dans le cadre de notre formation de « Data Scientist » nous réalisons un projet fil rouge qui matérialise "
            "l’ensemble des méthodes \napprises au cours des neuf mois de la formation. Notre projet s’intitule « "
            "Recommandations musicales — prédire les musiques les \nplus appréciées». Le but ici sera de "
            "prédire le degré d’appréciation d’une musique à partir d’événements d’écoute. Pour cela, \nnous disposons "
            "d’un triplet de jeu de données provenant du concours Kaagle Nowplayingrs Datasets  ainsi que du jeux de "
            "donnée \nlastFM_1K proposé par le MTG (Musique technologie groupe) de l’université Pompeu Fabra de "
            "Barcelone:")

    st.write("https://www.kaggle.com/chelseapower/nowplayingrs?select=user_track_hashtag_timestamp.csv")
    st.write("https://www.upf.edu/web/mtg/lastfm360k")

    st.header('Explorations des modèles')

    st.text("Pour chaque modèle exploré, nous présenterons succinctement la théorie sur laquelle il repose puis nous "
            "commenterons les \ndifférentes étapes d’entrainement ainsi que les résultats obtenus. L’évaluation se "
            "fera à travers les deux métriques retenues, \nl’AUC calculé sur l’ensemble du dataset et le NDGC calculé "
            "sur un ensemble de 50 recommandations. A chaque fois, le score final \nretenu est la moyenne des scores "
            "obtenue sur l’ensemble des utilisateurs. La validité du modèle sera évaluée à partir de cinq "
            "\nutilisateurs factices dont les goûts sont très tranchés (Jazz, classique, pop, rock et rap). On notera "
            "la qualité des \nrecommandations complètes (ensemble des items) et des recommandations filtrées (sous "
            "ensemble des items ne comprenant pas les \nitems déjà rencontré par l’utilisateur).")

    st.header('Comparaison des modèles')

    image = Image.open('data/result.png')

    st.image(image, caption='Récapitulatif des scores obtenus pour les différents modèles explorés')

    st.header('Conclusion')

    st.text("Dans le cadre de notre projet fil rouge, plusieurs modèles de systèmes de recommandation ont été testés "
            "et évalués suivant \ndeux métriques (l’AUC et le NDGC) ainsi que sur un panel de cinq utilisateurs "
            "factices. Il ressort une domination des modèles de \ntype collaborative filtering. Ils se révèlent en "
            "effet plus performants pour capter les goûts spécifiques des utilisateurs. En \nparticulier, la notation "
            "implicite présente dans notre jeu de données a donné l’avantage au modèle implicit ALS. \n\nDans le temps "
            "imparti pour ce projet, il n’a malheureusement pas été possible d’explorer tout le potentiel des modèles "
            "hybrides. \nIl serait ainsi intéressant de chercher à optimiser le modèle LightFM. Par ailleurs, "
            "il n’a pas été possible d’explorer des \nsystèmes de recommandation basés sur les réseaux de neurones. Ces "
            "modèles seraient ainsi une suite logique à notre projet. \n\nUne nouvelle tendance des systèmes de "
            "recommandation (de films ou de musiques) est de laisser l’utilisateur choisir le style \nd’items qu’il "
            "désire se voir recommander. Ainsi, par exemple, le « flow » proposé par Deezzer permet à l’utilisateur "
            "de choisir \nentre plusieurs ambiances (Motivation, chill, focus, party, …). À l’aide de certaines "
            "track_features comme la danceability ou \nle tempo, il serait possible d’adapter un modèle pour proposer "
            "le même type de service. \n\nEnfin, nous pourrions explorer la possibilité d’avoir un curseur contrôlant le "
            "niveau d’éclectisme des recommandations, ceci afin \nd’éviter le phénomène de bulles hermétiques dans "
            "lesquelles les modèles testés ont tendance à enfermer les utilisateurs.")
