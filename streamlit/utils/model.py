import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics.pairwise import cosine_similarity

import implicit
from scipy.sparse import coo_matrix

from surprise import KNNBasic

import pickle


class ContentBased:

    def __init__(self):
        df_track = st.session_state['df']['track_vect']

        self.df_res = pd.DataFrame()
        self.df_res['track_id'] = df_track['track_id']

        vect = df_track.drop(columns=['track_id', 'artist_name', 'track_name']). \
            apply(lambda r: tuple(r), axis=1).apply(np.array).values
        self.track_vect = np.array([x for x in vect])

    def get_reco(self, vec, user_id, n=200, filter=False):
        vec = np.array([x for x in vec[0]])
        res = cosine_similarity([vec], self.track_vect)[0]
        self.df_res['cosine_similarity'] = res

        if filter:
            self.filter_reco(user_id)

        if n >= 0:
            res = self.df_res.sort_values(by=['cosine_similarity'], ascending=False).iloc[0:n]

        return res['track_id'].values

    def filter_reco(self, user_id):
        df = st.session_state['df']['fake']
        track_list = set(df[df['user_id'] == user_id]['track_id'].values)
        f = self.df_res['track_id'].apply(lambda x: x not in track_list)
        self.df_res = self.df_res[f]


class ALSImplicit:

    def __init__(self):
        self.model = pickle.load(open("utils/pretrained/ALS_model.p", "rb"))
        self.mat = pickle.load(open("utils/pretrained/ALS_mat.p", "rb"))

    def get_reco(self, user_id, n=200, filter=False):
        track_ids = self.model.recommend(user_id, self.mat.tocsr().T, n)
        reco = [x[0] for x in track_ids]

        if filter:
            df = st.session_state['df']['fake']
            track_list = set(df[df['user_id'] == user_id]['track_id'].values)
            reco = [x for x in reco if x not in track_list]

        return reco


class KNN:

    def __init__(self):
        self.model = pickle.load(open("utils/pretrained/KNN_model.p", "rb"))

    def get_reco(self, user_id, n=200, filter=False):
        scores = []
        track_ids = []
        user_ids = []

        df = st.session_state['df']['fake']
        track_list = set(df[df['user_id'] == user_id]['track_id'].values)

        for track_id in st.session_state['df']['track']['track_id'].unique():
            score = self.model.predict(user_id, track_id, r_ui=1)[3]

            if filter and track_id not in track_list:
                user_ids.append(user_id)
                track_ids.append(track_id)
                scores.append(score)
            elif not filter:
                user_ids.append(user_id)
                track_ids.append(track_id)
                scores.append(score)

        recommendation = pd.DataFrame({'user_id': user_ids, 'track_id': track_ids, 'score': scores})

        if n >= 0:
            recommendation = recommendation.sort_values(by=['score'], ascending=False).iloc[:n]

        return recommendation['track_id'].values


class SVD:

    def __init__(self):
        self.model = pickle.load(open("utils/pretrained/SVD_model.p", "rb"))

    def get_reco(self, user_id, n=200, filter=False):
        scores = []
        track_ids = []
        user_ids = []

        df = st.session_state['df']['fake']
        track_list = set(df[df['user_id'] == user_id]['track_id'].values)

        for track_id in st.session_state['df']['track']['track_id'].unique():
            score = self.model.predict(user_id, track_id, r_ui=1)[3]

            if filter and track_id not in track_list:
                user_ids.append(user_id)
                track_ids.append(track_id)
                scores.append(score)
            elif not filter:
                user_ids.append(user_id)
                track_ids.append(track_id)
                scores.append(score)

        recommendation = pd.DataFrame({'user_id': user_ids, 'track_id': track_ids, 'score': scores})

        if n >= 0:
            recommendation = recommendation.sort_values(by=['score'], ascending=False).iloc[:n]

        return recommendation['track_id'].values
