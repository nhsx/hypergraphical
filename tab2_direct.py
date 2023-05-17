###############################################################################
# Libraries and Imports
###############################################################################
import os
import streamlit as st
import pandas as pd
import numpy as np
from scipy import linalg
import itertools

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
            "occurs (the raw prevalence)."
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
            hyperarc_count_df.iloc[i, 0] = numpy_utils.print_hyperarc_str(
                hyperarc_comma
            )

        # st.dataframe(hyperarc_count_df)

        # Find the number of times the example hyperarc occurs
        hyperarc_count_row = hyperarc_count_df[
            hyperarc_count_df["Hyperarc"] == examp_hyperarc
        ]
        examp_hyperarc_count = hyperarc_count_row.iloc[0, 1]

        st.markdown(
            f"The raw prevalence of hyperarc {{{examp_hyperarc}}} is {examp_hyperarc_count}."
        )

        st.markdown(f"$C({{{examp_hyperarc}}})$ = {examp_hyperarc_count}.")

        st.markdown(
            "Next we need to count the number of times the hyperarc "
            "children of the parent hyperedge occurs. In other words we need "
            "to find the siblings for the hyperarc and the number of "
            "times they occur."
        )

        st.markdown("The sibling hyperarcs are:")
        # Print of siblings
        elems = examp_hyperedge.split(", ")
        arc_siblings = list(itertools.permutations(elems, len(elems)))

        sibs_list = list()
        for hyp in arc_siblings:
            sib = numpy_utils.print_hyperarc_str(hyp)
            sibs_list.append(sib)

            st.write(sib)

        # Count of hyperarc siblings
        hyperedge_par_list = list()
        for i, hyp_edge in hyperarc_count_df.iterrows():
            arc = hyperarc_count_df.iloc[i, 0]
            par_edge = arc.replace("->", ",").replace(" ", "")
            par_edge_list = par_edge.split(",")
            par_edge_set = set(par_edge_list)
            hyperedge_par_list.append(par_edge_set)

        hyperarc_count_df["Parent Hyperedge"] = hyperedge_par_list
        hyperarc_count_df["Siblings Count"] = [None] * len(hyperarc_count_df)
        hyperarc_count_df["Parent Hyperedge Weight"] = [None] * len(hyperarc_count_df)
        hyperarc_count_df["w(h_i)"] = [None] * len(hyperarc_count_df)

        for i, row in hyperarc_count_df.iterrows():
            # Count the sum of the parent edges
            par_edge = hyperarc_count_df.iloc[i, 2]
            matching_par_rows = hyperarc_count_df[
                hyperarc_count_df["Parent Hyperedge"] == par_edge
            ]
            par_sum = matching_par_rows["Count"].sum()
            hyperarc_count_df.iloc[i, 3] = par_sum

            # Get the hyperedge weights from the soren_dice df
            soren_row = soren_dice_df[
                soren_dice_df["Hyperedge"] == tuple(sorted(par_edge))
            ]
            if len(soren_row) > 0:
                hyperarc_count_df.iloc[i, 4] = soren_row.iloc[0, 4]

        # This is the sibling count of the examp hyperarc
        examp_sib_count = hyperarc_count_df[
            hyperarc_count_df["Hyperarc"] == examp_hyperarc
        ].iloc[0, 3]

        st.markdown(f"These siblings occur {examp_sib_count} times.")

        st.markdown(r"""$$\sum_{h_j \in \mathcal{K}(h_i)}C(h_j)$$ is equal to:""")
        st.markdown(f"{examp_sib_count}")

        st.markdown("We can then calculate this hyperarc weight $w(h_{i})$ as:")
        st.markdown(
            f"{examp_hyperedge_we}({examp_hyperarc_count}/{examp_sib_count})={examp_hyperedge_we*(examp_hyperarc_count/examp_sib_count)}"
        )

        st.markdown(
            "The Table below gives the counts and weight "
            "values for all of the hyperarcs:"
        )

        hyperarc_count_df["w(h_i)"] = hyperarc_count_df["Parent Hyperedge Weight"] * (
            hyperarc_count_df["Count"] / hyperarc_count_df["Siblings Count"]
        )

        st.dataframe(hyperarc_count_df)

    tab2.subheader("RandomWalk Probability Transition Matrix:")

    tab2.write("#### Successor Transition Matrix")

    tab2.write("#### Predecessor Transition Matrix")

    tab2.subheader("PageRank:")

    tab2.write("#### Successor PageRank")
    tab2.text("Disease most likely to be successor diseases...")

    tab2.write("#### Predecessor PageRank")
    tab2.text("Disease most likely to be predecessor diseases...")
