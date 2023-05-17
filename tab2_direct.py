###############################################################################
# Libraries and Imports
###############################################################################
import os
import streamlit as st
import pandas as pd
import numpy as np
from scipy import linalg

# from sklearn import preprocessing
from string import ascii_uppercase as auc

# local
from src import build_model, centrality, centrality_utils, weight_functions
from src import numpy_utils

##############################################################################


def tab2_directed(
    tab2, final_prog_df, dis_list, edge_list, binmat, conds_worklist, all_progs
):
    tab2.header("Directed Hypergraph")
    tab2.subheader("_Page is Work in Progress_ 👷")
    if tab2.checkbox("Show the list of each patient's final hyperarc"):
        tab2.write(final_prog_df)
    tab2.subheader("Visual population representation:")

    # Draw b hypergraph from randomly generated patients
    numpy_utils.draw_b_hypergraph(dis_list, edge_list, tab2)
    tab2.subheader("Predecessor and Successor Disease Importance")
    tab2.markdown(
        "In this section we will go through how to calculate "
        "the importance of each disease as a predecessor "
        "or a successor, using left Eigenvector centrality on "
        "the directed hypergraph to obtain PageRank scores. "
        "Here we define a predecessor as a disease that is observed "
        "before another, and a successor is a disease that is observed "
        "after another. See overview image below."
    )

    pagerank_image = numpy_utils.add_image(
        image_path="images/PageRank.png", width=700, height=380
    )
    tab2.image(
        pagerank_image,
        caption="PageRank overview for hypergraphs.",
    )

    tab2.subheader("Hyperarc weight calcuations:")
    tab2.markdown(
        "Hyperarcs are calculated using the hyperedge calculations from the "
        "`Undirected Hypergraph` tab. From those calculations we get the "
        "following hyperedge weights:"
    )

    # NOTE: the undirected hypergraph hyperedge calcs don't include
    # single/selfedges however the directed hypergraph hyperedge calcs do
    # but this we'll use the build_model etc files for these instead?
    soren_dice_df = numpy_utils.soren_dice_create_df(edge_list, dis_list)
    tab2.dataframe(soren_dice_df)

    tab2.markdown(
        "Once we have the hyperedge weights we can calculate the "
        "hyperarc weights. The weighting of the hyperarcs considers disease "
        "prevalence by weighting its prevalence among other children and of "
        "its parenting hyperedge, which overall is weighted by the prevalence "
        " of the hyperedge itself."
    )

    with tab2.expander("How to calculate hyperarc weights?"):
        st.markdown("The equation:")

        st.markdown(
            r"""
                $$\mathcal{K}(h_i) = \{h_j \hspace{2pt} : \hspace{2pt} p(h_j)=p(h_i)\}$$
                """
        )

        st.markdown(
            "is the set of all the hyperarc children of the parent "
            "hyperedge $p(h_{i})$. Where $\mathcal{K}(h_i)$ is the set of "
            "siblings for hyperarc $h_{i}$. And $W(p(h_i))$ is the hyperedge "
            "weight of the parent $e_i$. Then we define the weight for hyperarc "
            "$h_i$ as"
        )

        st.markdown(
            r"""$$w(h_i) = W(p(h_i)) \frac{C(h_i)}{\sum_{h_j \in \mathcal{K}(h_i)}C(h_j)}$$"""
        )

        st.markdown(
            "Note that $W(\cdot)$ represents a _hyperedge_ "
            "weight while $w(\cdot)$ represents a "
            "_hyperarc_ weight."
        )
        examp_hyperedge = soren_dice_df.iloc[0, 0]

        examp_hyperedge_we = soren_dice_df.iloc[0, 4]

        examp_hyperarc = numpy_utils.print_hyperarc_str(examp_hyperedge)
        examp_hyperedge = ", ".join(map(str, examp_hyperedge))

        st.markdown(
            f"As an example let's calculate the hyperarc weight of hyperarc {{{examp_hyperarc}}}:"
        )

        st.markdown(
            f"The weight of the parent hyperedge {{{examp_hyperedge}}} of hyperarc {{{examp_hyperarc}}} is {examp_hyperedge_we}."
        )

        st.markdown(f"$W(p({{{examp_hyperarc}}}))$ = {examp_hyperedge_we}.")
        st.markdown(
            "Then we need to count the number of times the hyperarc "
            "occurs (the raw prevalence). The Table below shows the "
            "prevalence for each hyperarc within the population."
        )

        # Get a count for the number of times each of the hyperarcs appear
        hyperarcs_dups_list = list()
        for i in range(len(all_progs)):
            ind_progs = all_progs[i]
            for hyparc in range(len(ind_progs)):
                ind_prog_i = ind_progs[hyparc]
                hyperarcs_dups_list.append(tuple(ind_prog_i))

        hyparc_count_dict = {}
        for item in hyperarcs_dups_list:
            if item in hyparc_count_dict:
                hyparc_count_dict[item] += 1
            else:
                hyparc_count_dict[item] = 1

        # Convert the counts to a DataFrame
        hyperarc_count_df = pd.DataFrame(
            list(hyparc_count_dict.items()), columns=["Hyperarc", "Count"]
        )

        for i, row in hyperarc_count_df.iterrows():
            hyperarc_comma = hyperarc_count_df.iloc[i, 0]
            hyperarc_comma = ", ".join(map(str, hyperarc_comma))
            last_comma_idx = hyperarc_comma.rfind(",")
            if last_comma_idx != -1:
                hyperarc_comma = (
                    hyperarc_comma[:last_comma_idx]
                    + " -> "
                    + hyperarc_comma[last_comma_idx + 1 :]
                )
            hyperarc_count_df.iloc[i, 0] = hyperarc_comma

        st.dataframe(hyperarc_count_df)

        # Dataframe with hyperarc and occurence
        # st.markdown(f"")

    tab2.subheader("RandomWalk Probability Transition Matrix:")

    tab2.write("#### Successor Transition Matrix")

    tab2.write("#### Predecessor Transition Matrix")

    tab2.subheader("PageRank:")

    tab2.write("#### Successor PageRank")
    tab2.text("Disease most likely to be successor diseases...")

    tab2.write("#### Predecessor PageRank")
    tab2.text("Disease most likely to be predecessor diseases...")
