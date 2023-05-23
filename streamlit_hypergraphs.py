###############################################################################
# Libraries and Imports
###############################################################################
import streamlit as st
import base64
from src import numpy_utils, progressions
import tab0_mot_tab
import tab1_undirect
import tab2_direct
from string import ascii_uppercase as auc


###############################################################################
# Configure Page and Format
###############################################################################

st.set_page_config(
    page_title="Multimorbidity Calculations",
    page_icon="https://www.england.nhs.uk/wp-content/themes/nhsengland/static/img/favicon.ico",  # noqa: E501
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        # "Get Help": "https://www.england.nhs.uk/allocations/",
        # "Report a bug": "<github url>",
        # "About": "[https://www.england.nhs.uk/]\
        # (https://www.england.nhs.uk/)",
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
"""  # noqa: E501
render_svg(svg)


###############################################################################
# SIDEBAR
###############################################################################

# st.sidebar.subheader("Random Patient Generator")

view_choice = st.sidebar.selectbox(
    "What would you like to view?",
    (
        "Population hypergraph calculations",
        "Most likely disease successors",
        "Most likely disease predecessors",
    ),
)


num_patients = st.sidebar.slider(
    "Number of patients to generate", min_value=5, max_value=20
)
num_dis = st.sidebar.slider("Number of diseases", min_value=2, max_value=5)

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
    st.sidebar.subheader("Standard Graph (Undirected)")

    max_edges_undir = int((num_dis * (num_dis - 1)) / 2)
    st.sidebar.write(
        max_edges_undir,
        "edges in a standard undirected graph",
    )

    # Calculate the number of possible undirected hyperedges
    st.sidebar.subheader("Hyperedges (Undirected Hypergraph)")
    max_hyperedges = numpy_utils.N_max_hyperedges(n_diseases=num_dis)
    st.sidebar.write(
        max_hyperedges,
        "hyperedges in an undirected hypergraph",
    )

    # Calculate the number of possible directed edges
    st.sidebar.subheader("Standard Graph (Directed)")

    max_edges = num_dis * (num_dis - 1)
    st.sidebar.write(
        max_edges,
        "edges in a standard directed graph (without self-loops)",
    )

    st.sidebar.write(
        max_edges + num_dis,
        "edges in a standard directed graph (with self-loops)",
    )

    # Calculate the number of possible directed hyperedges (b-hyperarcs)
    st.sidebar.subheader("B-Hyperarcs (Directed Hypergraph)")
    max_hyperarcs = numpy_utils.N_max_hyperarcs(n_diseases=num_dis, b_hyp=True)
    st.sidebar.write(
        max_hyperarcs,
        "B-hyperarcs in a hypergraph (with self-loops)",
    )

    # Calculate the number of possible bf-hyperarcs
    st.sidebar.subheader("BF-Hyperarcs (Directed Hypergraph)")
    max_bf_arcs = numpy_utils.N_max_hyperarcs(n_diseases=num_dis, b_hyp=False)
    st.sidebar.write(
        max_bf_arcs,
        "BF-hyperarcs in a hypergraph (with self-loops)",
    )

edge_list, dis_list, final_prog_df, all_progs = numpy_utils.patient_maker(
    num_dis=num_dis, num_patients=num_patients, max_deg=num_dis
)

binmat, conds_worklist, idx_worklist = numpy_utils.create_worklists(
    len(dis_list), edge_list
)


###############################################################################
# POPULATION HYPERGRAPH CALCULATIONS
###############################################################################

if view_choice == "Population hypergraph calculations":
    st.title("Hypergraphs for Multimorbidity")

    numpy_utils.display_markdown_from_file("markdown_text/prototype.txt", st)

    mot_tab, tab1, tab2 = st.tabs(
        [
            "Why Hypergraphs for Multimorbidity",
            "Undirected Hypergraph",
            "Directed Hypergraph",
        ]
    )

    ###########################################################################
    # MOTIVATION TAB = WHY HYPERGRAPHS FOR MULTIMORBIDITY
    ###########################################################################

    tab0_mot_tab.tab0_motiv(mot_tab)

    ###########################################################################
    # TAB1 = UNDIRECTED HYPERGRAPH
    ###########################################################################

    tab1_undirect.tab1_undirected(
        tab1,
        final_prog_df,
        num_dis,
        edge_list,
        dis_list,
    )

    ###########################################################################
    # TAB2 = DIRECTED HYPERGRAPH
    ###########################################################################

    tab2_direct.tab2_directed(
        tab2,
        final_prog_df,
        dis_list,
        edge_list,
        all_progs,
        num_dis,
    )

    ###########################################################################
    # TODO: TAB3 = DUAL HYPERGRAPH
    ###########################################################################

    # tab3.subheader("Hyperedge weight calculation:")
    # tab3.subheader("Eigenvector Centrality:")
    # tab3.text("Most important sets of diseases...")


elif view_choice == "Most likely disease successors":
    node_labels = [*auc][:num_dis]
    st.markdown("_Page under construction_ 👷")
    st.subheader("Most likely successor diseases")
    st.markdown(
        "On this page we demonstrate how the directed hypergraphs "
        "created by the population data could be used to show which "
        "diseases are likely to succeed a current disease or set of "
        "diseases. This could provide health utility to clinical "
        "practioners be providing them with a next possible observed "
        "disease to help inform treatment plans."
    )
    # TODO: Implement

    st.markdown(
        "We first need to find the dual of the hypergraph $H$. The dual "
        "hypergraph $H^{*}$ can be formed by taking the original hypergraph's "
        "incidence matrix and transposing. This swaps the nodes and edges "
        "such that $H^{*}$ has $m$ nodes and $n$ edges, so the single "
        "disease sets (originally nodes in $H$) become edges linking the "
        "different sets of diseases (originally edges in $H$) and the edges "
        "in $H$ become the nodes of $H^{*}$."
    )

    st.markdown(
        "Given the fictitious population generated with this applet, "
        "you may input a single disease or disease set to find out "
        "which diseases are likely to be observed next. "
        "This input should be in the format _$dis_1$_, _$dis_2$_, ...,"
        "_$dis_{n-1}$_ where $dis_n$ is from the list:"
    )
    st.markdown(f"{node_labels}")
    st.markdown("For example, you could input `A,C`.")
    dis_input = st.text_input("Enter your disease/disease set here 👇")
    n_progressions = st.slider(
        "Number of progressions to return:",
        min_value=1,
        max_value=5,
    )
    max_degree = st.slider(
        "Maximum number degree of diseases to generate:",
        min_value=1,
        max_value=5,
    )
    st.write(dis_input)

    dir_inc_mat_df = progressions.np_inc_mat(edge_list, dis_list, st)
    st.dataframe(dir_inc_mat_df)
    inc_mat_arr = dir_inc_mat_df.values
    st.write(inc_mat_arr)

    # Need hyperarc dataframe with cols:
    # Hyperarc e.g. 'A -> B' | Degree | Centrality
    # Where the centrality is the dual eigenvector centrality

    # hyperarc_centrality = centrality.eigenvector_centrality(incidence_matrix,
    #                                                       mort1_hyperarc_weights,
    #                                                       mort1_node_weights,
    #                                                       rep="dual",
    #                                                       tolerance=1e-6,
    #                                                       max_iterations=1000,
    #                                                       weight_resultant=True,
    #                                                       random_seed=None)

    # n_conds = n_diseases*[2] + [len(d.split(",")) + 1 for d in mort1_hyperarc_titles[n_diseases:]]

    # hyperarc_evc = pd.DataFrame({
    #     "Disease": mort1_hyperarc_titles,
    #     "Degree":n_conds,
    #     "Eigenvector Centrality": np.round(hyperarc_centrality,3)},)
    # hyperarc_evc.sort_values(by="Degree", ascending=True).reset_index(drop=True)

    progressions.generate_forward_prog(
        str(dis_input), hyperarc_evc, n_progressions, max_degree
    )

elif view_choice == "Most likely disease predecessors":
    st.markdown("_Page under construction_ 👷")
    # TODO: Implement

# TODO: NUMBA explained (why we need it and the 3 worklists)
# Link to Github repository

st.markdown("-" * 50)
st.text("Last Updated 22nd May 2023 \t\t\t\t\t Version 0.1.0")
