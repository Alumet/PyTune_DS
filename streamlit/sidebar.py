import streamlit as st
import page_als_full
import page_home
import page_model_comparison


def set_page(arg):
    st.session_state['current_page'] = arg


def create():
    # Sidebar creation

    st.sidebar.title('PYTUNE')
    st.sidebar.text('Projet file rouge\nFormation Data Scientist\nDataScientest')
    st.sidebar.button(label='DESCRIPTION', on_click=set_page, args=['Page_0'])

    st.sidebar.header('COMPARAISON DES MODELES')
    st.sidebar.text("Evaluation des modèles sur cinq \nutilisateurs factices aux goûts \nmusicaux bien tranchés ("
                    "Jazz', \n'Classique', 'Pop', 'Rock', 'Rap')")
    st.sidebar.button(label='LANCER', on_click=set_page, args=['Page_1'], key='Page_1')

    st.sidebar.header('MODELE LIVE')
    st.sidebar.text('Demonstration live basée sur un \nmodèle ALS_implicit')
    st.sidebar.button(label='LANCER', on_click=set_page, args=['Page_2'], key='Page_2')

    st.sidebar.subheader('\nCredits')
    st.sidebar.write("[Yann HUET](https://www.linkedin.com/in/huetyann)\n[Jiayang GAO]("
                     "https://www.linkedin.com/in/jiayang-gao-4622a558/)")

    if st.session_state['current_page'] == 'Page_0':
        page_home.create()
    elif st.session_state['current_page'] == 'Page_1':
        page_model_comparison.create()
    elif st.session_state['current_page'] == 'Page_2':
        page_als_full.create()

