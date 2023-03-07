import streamlit as st
import implicit
import numpy as np
from scipy.sparse import coo_matrix
import random

from utils.tools import youtube_search


def create_artist_selection(artist_list):
    user_selection = st.session_state['user']['artist_selection']
    artist_list = [x for x in artist_list if x not in user_selection]
    if st.session_state['user']['n_artist_item'] < 7:
        random.shuffle(artist_list)
    for i, col in enumerate(st.columns(4)):

        with col:
            for el in artist_list[i * 5:i * 5 + 5]:
                label = el.split('&')[0][:30]
                side = '. ' * int(((30 - len(label)) / 2))
                label = side + ' ' + label + ' ' + side
                st.button(label=label, on_click=artist_add, args=[el])


def artist_add(selected_artist):
    df_artist = st.session_state['df']['artist']

    id_item = df_artist[df_artist['artist_name'] == selected_artist]['artist_id'].unique()[0]
    st.session_state['user']['artist_selection'].append(selected_artist)

    st.session_state['coo_artist']['weight'] = np.append(st.session_state['coo_artist']['weight'], np.array(100))
    st.session_state['coo_artist']['item'] = np.append(st.session_state['coo_artist']['item'], np.array(id_item))
    st.session_state['coo_artist']['user'] = np.append(st.session_state['coo_artist']['user'], np.array(1001))
    st.session_state['user']['n_artist_item'] += 1

    if st.session_state['user']['n_artist_item'] >= 3:
        mat_artist = coo_matrix((st.session_state['coo_artist']['weight'],
                                 (st.session_state['coo_artist']['item'],
                                  st.session_state['coo_artist']['user'])
                                 ))

        model = implicit.als.AlternatingLeastSquares(factors=30,
                                                     use_native=True,
                                                     use_cg=True,
                                                     calculate_training_loss=True,
                                                     num_threads=-1,
                                                     iterations=10)

        model.fit(mat_artist)

        nb_reco = st.session_state['user']['n_artist_item'] + 20
        reco = [x[0] for x in model.recommend(1001, mat_artist.tocsr().T, nb_reco)]
        new_list = [df_artist[df_artist['artist_id'] == x]['artist_name'].values[0] for x in reco]
        st.session_state['selection'] = new_list

        st.session_state['user']['similar_user'] = [x[0] for x in model.similar_users(1001) if x[0] != 1001][:3]

        st.session_state['user']['user_reco'] = list()


def track_add(track_id):
    st.session_state['user']['n_track_item'] += 1
    st.session_state['user']['liked'].append(track_id)

    st.session_state['coo_music']['weight'] = np.append(st.session_state['coo_music']['weight'], np.array(100))
    st.session_state['coo_music']['item'] = np.append(st.session_state['coo_music']['item'], np.array(track_id))
    st.session_state['coo_music']['user'] = np.append(st.session_state['coo_music']['user'], np.array(1001))

    st.session_state['user']['user_reco'].remove(track_id)


def track_remove(track_id):
    st.session_state['user']['disliked'].append(track_id)
    st.session_state['user']['user_reco'].remove(track_id)


def train_music_model():
    with st.spinner('Training model...'):
        mat_music = coo_matrix((st.session_state['coo_music']['weight'],
                                (st.session_state['coo_music']['item'],
                                 st.session_state['coo_music']['user'])
                                ))

        model = implicit.als.AlternatingLeastSquares(factors=40,
                                                     use_native=True,
                                                     use_cg=True,
                                                     calculate_training_loss=False,
                                                     num_threads=-1,
                                                     iterations=20)

        model.fit(mat_music)

        st.session_state['model'] = (model, mat_music)


def get_reco():
    n_track = st.session_state['user']['n_track_item']
    lt = st.session_state['last_trained']

    if 'model' not in st.session_state or (n_track >= 20 and n_track % 5 == 0 and lt < n_track):
        train_music_model()
        st.session_state['last_trained'] = n_track

    model, mat_music = st.session_state['model']

    reco = list()

    if st.session_state['user']['n_track_item'] < 50:

        for user_id in st.session_state['user']['similar_user']:
            reco += model.recommend(user_id, mat_music.tocsr().T, 10)

    if st.session_state['user']['n_track_item'] > 20:
        fact = st.session_state['user']['n_track_item']/50
        reco += [(x[0], x[1] * fact) for x in model.recommend(1001, mat_music.tocsr().T, 50)]

    if 30 < st.session_state['user']['n_track_item'] < 50:
        st.session_state['user']['similar_user'] = [x[0] for x in model.similar_users(1001) if x[0] != 1001][:3]

    reco.sort(key=lambda x: x[1])

    if len(st.session_state['user']['user_reco']) <= 0:
        reco_single = list()
        for el, _ in reco:
            if el not in reco_single:
                reco_single.append(el)

    else:
        old = st.session_state['user']['user_reco']
        old += [x[0] for x in reco]

        reco_single = list()
        for el in old:
            if el not in reco_single:
                reco_single.append(el)

    reco_single = [x for x in reco_single if x not in st.session_state['user']['liked']]
    reco_single = [x for x in reco_single if x not in st.session_state['user']['disliked']]

    st.session_state['user']['user_reco'] = reco_single


def display_reco():
    labels = list()
    ids = list()
    duplicates = list()

    df_track = st.session_state['df']['track']

    for el in st.session_state['user']['user_reco']:
        a = df_track[df_track['track_id'] == el]
        track = a['track_name'].values[0]
        artist = a['artist_name'].values[0]

        if artist not in duplicates:
            labels.append(f'{artist}  ---->  {track}')
            ids.append(el)
            duplicates.append(artist)

        if len(duplicates) >= 10:
            break

    col1, col2, col3 = st.columns([6, 1, 1])

    with col1:
        for label in labels:
            st.button(label=label, on_click=youtube_search, args=[label], key=label)

    with col2:
        for track_id in ids:
            st.button(label="J'aime", on_click=track_add, args=[track_id], key=track_id)

    with col3:
        for track_id in ids:
            st.button(label="Je n'aime pas", on_click=track_remove, args=[track_id], key=track_id)


def create():
    st.title('PYTUNE LIVE', anchor=None)
    st.header('1/2 - Choisir les artists que vous aimez (minimum 7)', anchor=None)

    st.progress(min([st.session_state['user']['n_artist_item'] / 7, 1.0]))

    create_artist_selection(st.session_state['selection'])

    if st.session_state['user']['n_artist_item'] >= 7:
        st.header('2/2 - Vos recommandations personalisées', anchor=None)
        st.progress(min([st.session_state['user']['n_track_item'] / 20, 1.0]))
        get_reco()
        display_reco()
