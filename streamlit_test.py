###############################################################################
# Libraries
###############################################################################
import streamlit as st
import base64

# import networkx as nx
# import hypernetx as hnx
# import matplotlib.pyplot as plt
# import random
# import numpy as np
# import numba
# import pandas as pd

# local
from utils import *

###############################################################################
# Configure Page and Format
###############################################################################

st.set_page_config(
    page_title="Multimorbidity Calculations",
    page_icon="https://www.england.nhs.uk/w\
        p-content/themes/nhsengland/static/img/favicon.ico",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        # "Get Help": "https://www.england.nhs.uk/allocations/",
        # "Report a bug": "<github url>",
        # "About": "[https://www.england.nhs.uk/allocations/]\
        # (https://www.england.nhs.uk/allocations/)",
    },
)
padding = 1
st.markdown(
    f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
    }} </style> """,
    unsafe_allow_html=True,
)

st.set_option("deprecation.showPyplotGlobalUse", False)

# render svg image
def render_svg(svg):
    """Renders the given svg string."""
    b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    html = r'<img src="data:image/svg+xml;base64,%s"/>' % b64
    st.write(html, unsafe_allow_html=True)


# NHS Logo
svg = """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 16">
            <path d="M0 0h40v16H0z" fill="#005EB8"></path>
            <path d="M3.9 1.5h4.4l2.6 9h.1l1.8-9h3.3l-2.8 13H9l-2.7-9h-.1l-1.8 9H1.1M17.3 1.5h3.6l-1 4.9h4L25 1.5h3.5l-2.7 13h-3.5l1.1-5.6h-4.1l-1.2 5.6h-3.4M37.7 4.4c-.7-.3-1.6-.6-2.9-.6-1.4 0-2.5.2-2.5 1.3 0 1.8 5.1 1.2 5.1 5.1 0 3.6-3.3 4.5-6.4 4.5-1.3 0-2.9-.3-4-.7l.8-2.7c.7.4 2.1.7 3.2.7s2.8-.2 2.8-1.5c0-2.1-5.1-1.3-5.1-5 0-3.4 2.9-4.4 5.8-4.4 1.6 0 3.1.2 4 .6" fill="white"></path>
          </svg>
"""
render_svg(svg)


###############################################################################
# SIDEBAR
###############################################################################

st.sidebar.subheader("Random Patient Generator")

view_choice = st.sidebar.selectbox(
    "What would you like to view?",
    (
        "Population hypergraph calculations",
        "Future disease prediction",
        "Most likely cause(s) of disease",
    ),
)


num_patients = st.sidebar.slider(
    "Number of patients to generate", min_value=1, max_value=20
)
num_dis = st.sidebar.slider("Number of diseases to generate", min_value=1, max_value=5)

if st.sidebar.checkbox("Show Maximum Number of Edges"):
    st.sidebar.write(
        # "With ",
        # num_patients,
        # "patients and ",
        "With",
        num_dis,
        "potential diseases there could be up to... ",
    )
    # Calculate the number of possible directed edges
    st.sidebar.subheader("Standard Directed Graph")

    max_edges = num_dis * (num_dis - 1)
    st.sidebar.write(
        max_edges,
        "edges in a standard directed graph",
    )

    # Calculate the number of possible undirected hyperedges
    st.sidebar.subheader("Hyperedges (Undirected Hypergraph)")
    max_hyperedges = N_max_hyperedges(n_diseases=num_dis)
    st.sidebar.write(
        max_hyperedges,
        "hyperedges in an undirected hypergraph",
    )

    # Calculate the number of possible directed hyperedges (b-hypergraphs)
    st.sidebar.subheader("Hyperarcs (Directed Hypergraph)")
    max_hyperarcs = N_max_hyperarcs(n_diseases=num_dis, b_hyp=True)
    st.sidebar.write(
        max_hyperarcs,
        "hyperarcs in a b-hypergraph",
    )

edge_list, dis_list, final_prog_df = patient_maker(
    num_dis=num_dis, num_patients=num_patients, max_deg=num_dis
)

binmat, conds_worklist, idx_worklist = create_worklists(len(dis_list), edge_list)

###############################################################################
# POPULATION HYPERGRAPH CALCULATIONS
###############################################################################

if view_choice == "Population hypergraph calculations":

    st.title("Hypergraphs for Multimorbidity")
    st.markdown("Last Updated 24th February 2023")
    st.markdown(
        "_This applet is a prototype which generates fictious patient data to show how hypergraphs can be used to explore multimorbidity._"
    )

    motivation_tab, tab1, tab2 = st.tabs(
        [
            "Why Hypergraphs for Multimorbidity",
            "Undirected Hypergraph",
            "Directed Hypergraph",
        ]
    )

    ###############################################################################
    # MOTIVATION TAB = WHY HYPERGRAPHS FOR MULTIMORBIDITY
    ###############################################################################

    # TODO: Explain multimorbidity and hypergraphs in this section
    motivation_tab.subheader("Multimorbidity")
    motivation_tab.markdown(
        """Multimorbidity is defined as having 2 or more long term chronic
        health conditions simultaneously. Multimorbidity is 
        associated with increased health service utilisation 
        \cite{casselle2018epidemiology}.

People are living longer and so the population is ageing. This is due to 
improvements in lifestyle through things such as having healthier diets, exercising and not smoking. Treatment improvements mean that more people are surviving acute conditions. The number of hospitals and other healthcare services are not increasing fast enough to keep up with this rising ageing population and increased health service utilisation required by patients with multimorbidity, so healthcare services are being burdened. 

Frameworks and policies in healthcare management are designed for individual conditions, rather than multiple. Patients with multiple long-term conditions can be difficult to treat as medications may interact and conditions require different management strategies. If departments could be placed more optimally distally when multimorbidity occurs, access for patients could be easier and quicker. Which is especially important for those with frailties. Additionally, having departments closer together allows them to communicate to better integrate treatment plans for their in common patients.

Approximately 1 in 4 patients in primary care within the UK had multiple chronic conditions in 2018 \cite{casselle2018epidemiology}. Unfortunately these multimoribidities reduce quality of life and can increase mortality. Even more concerning is that in 2015 54\% of people over 65 years old had multimorbidity, and in just 20 years this is expected to rise to 67.8\% \cite{kingston2018projections}.

One study using data from around 14 million patients in 2012 found that hypertension, depression/anxiety and then chronic pain were the most prevalent multimorbidity conditions. They also found than females (30\%) were more likely to have multimorbidity than males (24.4\%). And those with a low socioeconomic status had 4.2\% more people with multimorbidity than the highest socioeconomic status group. Additionally, around half of GP consultations and hospital admissions were for multimorbidity. 78.7\% of prescriptions were for multimorbidity \cite{casselle2018epidemiology}.

As multimorbidity becomes more prevalent its important that research in multimorbidity develops so that strategies can be backed up to change health frameworks and policies, to prevent accumulation of conditions, better manage those with multiple condition and reduce the burden on healthcare."""
    )

    motivation_tab.subheader("Graphs and Hypergraphs")
    motivation_tab.latex(
        r"""
        \mathcal{G}
        
    """
    )

    ###########################################################################
    # TAB1 = UNDIRECTED HYPERGRAPH
    ###########################################################################

    if tab1.checkbox("Show patients individually as undirected"):
        tab1.write(final_prog_df)

    tab1.header("Undirected Hypergraph")

    tab1.subheader("Visual representation:")
    tab1.write("Note: Self-connections are not considered with undirected.")
    draw_undirected_hypergraph(edge_list, tab1)

    tab1.subheader("Hyperedge weight calculation:")

    tab1.subheader("Eigenvector Centrality:")

    tab1.text("Most important diseases...")

    ###########################################################################
    # TAB2 = DIRECTED HYPERGRAPH
    ###########################################################################

    tab2.header("Directed Hypergraph")
    if tab2.checkbox("Show the list of each patient's final hyperarc"):
        tab2.write(final_prog_df)
    tab2.subheader("Visual representation:")

    # Draw b hypergraph from randomly generated patients
    draw_b_hypergraph(dis_list, edge_list, tab2)

    tab2.subheader("Hyperarc weight calcuations:")
    tab2.markdown(
        "Hyperarcs are calculated using the hyperedge calculations from the \
            `Undirected Hypergraph` tab"
    )

    tab2.subheader("RandomWalk Probability Transition Matrix:")

    tab2.write("#### Successor Transition Matrix")

    tab2.write("#### Predecessor Transition Matrix")

    tab2.subheader("PageRank:")

    tab2.write("#### Successor PageRank")
    tab2.text("Disease most likely to be successor diseases...")

    tab2.write("#### Predecessor PageRank")
    tab2.text("Disease most likely to be predecessor diseases...")

    ###########################################################################
    # TODO: TAB3 = DUAL HYPERGRAPH
    ###########################################################################

    # tab3.subheader("Hyperedge weight calculation:")
    # tab3.subheader("Eigenvector Centrality:")
    # tab3.text("Most important sets of diseases...")


elif view_choice == "Future disease prediction":
    st.markdown("_Page under construction_ 👷")
    # TODO: Implement


elif view_choice == "Cause(s) of disease":
    st.markdown("_Page under construction_ 👷")
    # TODO: Implement
