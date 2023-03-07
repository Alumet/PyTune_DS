import streamlit as st
import utils.model as model
from utils.tools import youtube_search


def display_reco(reco_list, col):
    labels = list()
    ids = list()
    duplicates = list()

    df_track = st.session_state['df']['track']

    for el in reco_list:
        a = df_track[df_track['track_id'] == el]
        track = a['track_name'].values[0]
        artist = a['artist_name'].values[0]

        if artist not in duplicates:
            label = f'{artist}  ---->  {track}'
            ids.append(el)
            duplicates.append(artist)
            st.button(label=label, on_click=youtube_search, args=[label], key=label + col)

        if len(duplicates) >= 10:
            break


def get_reco_from_model(option, user_id):

    if 'filtered' in option:
        filter = True
    else:
        filter = False

    if 'Content based' in option:

        vect = st.session_state['fake_users']['vect'][user_id]

        mcb = model.ContentBased()
        return mcb.get_reco(vect, user_id, filter=filter)

    elif 'ALS_Implicit' in option:

        als = model.ALSImplicit()
        return als.get_reco(user_id, filter=filter)

    elif 'KNN' in option:

        knn = model.KNN()
        return knn.get_reco(user_id, filter=filter)

    elif 'SVD' in option:

        svd = model.SVD()
        return svd.get_reco(user_id, filter=filter)

    else:
        return []


def create():

    st.title('COMPARAISON DES MODELES')

    st.header("Choisir le type d'utilisateur")

    option_user = st.selectbox('',
                               ('JAZZ', 'CLASSIQUE', 'POP', 'ROCK', 'RAP'))

    users2id = {'JAZZ': 1001,
                'CLASSIQUE': 1002,
                'POP': 1003,
                'ROCK': 1004,
                'RAP': 1005}

    user_id = users2id[option_user]

    st.header("Choisir les modèles à comparer")

    col1, col2 = st.columns(2)

    with col1:
        option_1 = st.selectbox('MODELE 1',
                                ('Content based',
                                 'KNN',
                                 'ALS_Implicit',
                                 'SVD',
                                 'Content based (filtered)',
                                 'KNN (filtered)',
                                 'ALS_Implicit (filtered)',
                                 'SVD (filtered)'),
                                key='col_1')

        reco = get_reco_from_model(option_1, user_id)
        display_reco(reco, 'col_1')

    with col2:
        option_2 = st.selectbox('MODELE 2',
                                ('Content based',
                                 'KNN',
                                 'ALS_Implicit',
                                 'SVD',
                                 'Content based (filtered)',
                                 'KNN (filtered)',
                                 'ALS_Implicit (filtered)',
                                 'SVD (filtered)'),
                                key='col_2')
        reco = get_reco_from_model(option_2, user_id)
        display_reco(reco, 'col_2')
