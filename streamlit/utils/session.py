import streamlit as st
import pandas as pd
import numpy as np
import pickle

base_list = ['The Beatles', 'The Rolling Stones', 'Elton John', 'Mariah Carey','Madonna', '2Pac', 'Michael Jackson',
             'Taylor Swift', 'Stevie Wonder', 'Nirvana', 'Whitney Houston', 'Eminem','Elvis Presley', 'Miles Davis',
             'U2', 'Usher', 'Prince', 'Rihanna', 'Billy Joel', 'Korn', 'Billie Holiday', 'Snoop Dogg',
             'Louis Armstrong', 'Rammstein', 'Linkin Park', 'Diana Krall', 'Wolfgang Amadeus Mozart',
             'Ludwig Van Beethoven']


@st.cache
def load_csv():
    file = 'data/streamlit_df.csv'
    df_track_user = pd.read_csv(file, index_col=0)
    df_artist = df_track_user.groupby(['artist_id']).first().reset_index()
    df_track = df_track_user.groupby(['track_id']).first().reset_index()

    file = 'data/streamlit_df_track_vector.csv'
    df_track_vect = pd.read_csv(file, index_col=0)

    df_fake = pd.read_csv(f'data/fake_user/last_fm_fake_user(1001)_jazz.csv', index_col=0)

    files = ['last_fm_fake_user(1002)_classic.csv',
             'last_fm_fake_user(1003)_pop.csv',
             'last_fm_fake_user(1004)_rock.csv',
             'last_fm_fake_user(1005)_rap.csv']

    for file in files:
        df_temp = pd.read_csv(f'data/fake_user/{file}', index_col=0)
        df_temp['rating'] = 100
        df_fake = pd.concat([df_fake, df_temp])

    return df_track_user, df_artist, df_track, df_track_vect, df_fake


def create():
    if 'df' not in st.session_state:
        df_track_user, df_artist, df_track, df_track_vect, df_fake = load_csv()

        st.session_state['df'] = {'df': df_track_user,
                                  'artist': df_artist,
                                  'track': df_track,
                                  'track_vect': df_track_vect,
                                  'fake': df_fake
                                  }

    if 'user' not in st.session_state:
        st.session_state['user'] = {'n_artist_item': 0,
                                    'n_track_item': 0,
                                    'artist_selection': [],
                                    'similar_user': [],
                                    'user_reco': [],
                                    'liked': [],
                                    'disliked': []}

    if 'coo_artist' not in st.session_state:
        df = st.session_state['df']['df']
        df_gb = df.groupby(['user_id', 'artist_id']).mean().reset_index()

        st.session_state['coo_artist'] = {'item': np.array(df_gb[['artist_id']].values).T[0],
                                          'user': np.array(df_gb[['user_id']].values).T[0],
                                          'weight': np.array(df_gb[['rating']].values).T[0]}

    if 'coo_music' not in st.session_state:
        df = st.session_state['df']['df']
        st.session_state['coo_music'] = {'item': np.array(df[['track_id']].values).T[0],
                                         'user': np.array(df[['user_id']].values).T[0],
                                         'weight': np.array(df[['rating']].values).T[0]}

    if 'selection' not in st.session_state:
        st.session_state['selection'] = base_list

    if 'last_trained' not in st.session_state:
        st.session_state['last_trained'] = 0

    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = 'Page_0'

    if 'fake_users' not in st.session_state:
        st.session_state['fake_users'] = {'vect': pickle.load(open("data/user_vect.p", "rb"))}
