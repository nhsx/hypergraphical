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
        "following hyperedge weights (undirected):"
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
            f"The weight of the parent hyperedge {{{examp_hyperedge}}} of hyperarc {{{examp_hyperarc}}} is {round(examp_hyperedge_we, 2)}."
        )

        st.markdown(f"$W(p({{{examp_hyperarc}}}))$ = {round(examp_hyperedge_we,2)}.")
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
            f"{round(examp_hyperedge_we,2)}({examp_hyperarc_count}/{examp_sib_count})={round(examp_hyperedge_we*(examp_hyperarc_count/examp_sib_count),2)}"
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

    tab2.markdown(
        "Random walks (a particular case of Markov random chain) "
        "can be used in this application with random steps being "
        "taken from one node to another, where each step is "
        "completely independent for the last step. The behaviour "
        "is determined by a transition probability matrix "
        "$\mathcal{P}$, where the column defines the start node "
        "and the row defines the end node. Here we can use random "
        "walks to find the probability of transitioning to another "
        "disease state based on a transition matrix e.g. "
        "probability of B to follow A."
    )

    with tab2.expander(
        "General equation for calculating the probability of transitioning "
        "from one node to another"
    ):
        st.markdown(
            r"""$$p(u,v) = \sum_{e \in \mathcal{E}} w(e) \frac{h(u,e)}{d(u)} \frac{h(v,e)}{\delta(e)}$$"""
        )

        st.markdown("Where")
        st.markdown("- $h(u,e) = 1$ if hyperedge $e \in E$ connects the nodes")
        st.markdown("- $d$ is the node degree function")
        st.markdown("- $\delta$ the edge degree function")

    tab2.write("#### Successor Transition Matrix")

    tab2.markdown(
        "We can find which disease is likely to be observed "
        "following other diseases. This is done by calculating "
        "the successor transition matrix. In the case of successor "
        "detection we only consider the transition from a tail to "
        "a head."
    )

    with tab2.expander("Successor Transition Matrix Equation"):
        st.markdown(
            "We can use the equation below to calculate the "
            "probability of their being a transition from node $u$ to "
            "node $v$ which can be formulated into a successor "
            "transition matrix."
        )

        st.markdown(
            r"""$$p(u,v) = \sum_{e \in \mathcal{E}} w(e) \frac{m_-(u,e)}{d_-(u)} \frac{m_+(v,e)}{\delta_+(e)}.$$"""
        )
        st.markdown("Where")
        st.markdown("- $u$ is the current node position")
        st.markdown("- $v$ is the node to be transitioned to")
        st.markdown(
            "- $m_-(u,e)$ = 1 if node $u$ has a edge $e$ stemming from it "
            "(a tail is connected to node $u$)"
        )
        st.markdown(
            "- $m_+(v,e)$ = 1 if node $v$ has the head of an edge $e$ "
            "connected to it"
        )
        st.markdown("- $d_-(u)$ = the sum of all possible contributions to $u$")
        st.markdown(
            "- $\delta_+ (e)$ = the number of nodes connected to the edge "
            "$e$ via the edges head (this will always be 1 in b-hypergraphs)"
        )
        st.markdown(
            r"""- $\frac{m_-(u,e)}{d_-(u)} \frac{m_+(v,e)}{\delta_+(e)}$ is the row normaliser"""
        )

    with tab2.expander("Example - calculate of the successor transition probability"):
        st.markdown(
            "First, we need to calculate the non-normalised "
            "probability of transitioning from one node to another."
        )
        st.markdown(
            r"""This is done by taking the sum of the hyperarc weights $w(e)$ for all possible node pairs $$\sum_{e \in \mathcal{E}} w(e).$$"""
        )

        # If a hyperarc has u in the tail and v in the head then get the sum of those hyperarc weights
        hyperarc_weights_df = hyperarc_count_df[["Hyperarc", "w(h_i)"]]
        # st.dataframe(hyperarc_weights_df)

        nn_succ_trans_df = pd.DataFrame(columns=dis_list)
        nn_succ_trans_df["Node"] = dis_list
        nn_succ_trans_df.set_index("Node", inplace=True)
        nn_succ_trans_df = nn_succ_trans_df.fillna(0)

        # get a list of all possible pairs
        all_dis_pairs = list(itertools.permutations(dis_list, 2))
        for pair in all_dis_pairs:
            for i, row in hyperarc_weights_df.iterrows():
                tail = hyperarc_weights_df.iloc[i, 0].split("->")[0]
                head = hyperarc_weights_df.iloc[i, 0].split("->")[1]
                if pair[0] in tail and pair[1] in head:
                    # if u in head and v in tail:
                    # sum the hyperarc weights for this particular transition
                    # st.markdown(
                    #     f"{pair[0]} in tail: {{{tail}}} and {pair[1]} in head: {{{head}}}"
                    # )
                    nn_succ_trans_df.loc[pair[0], pair[1]] += hyperarc_weights_df.iloc[
                        i, 1
                    ]

        st.markdown("__Example__")
        st.markdown(
            f"As an example let's calculate the probability of transitioning from node {all_dis_pairs[1][0]} to {all_dis_pairs[1][1]}."
        )

        st.markdown(
            f"First we find all the hyperarcs that have {all_dis_pairs[1][0]} in their tail and {all_dis_pairs[1][1]} in their head component and their corresponding weights:"
        )

        examp_succ_hyps = list()
        examp_succ_hyp_weights = list()
        for i, row in hyperarc_weights_df.iterrows():
            tail = hyperarc_weights_df.iloc[i, 0].split("->")[0]
            head = hyperarc_weights_df.iloc[i, 0].split("->")[1]
            if all_dis_pairs[1][0] in tail and all_dis_pairs[1][1] in head:
                examp_succ_hyps.append(hyperarc_weights_df.iloc[i, 0])
                examp_succ_hyp_weights.append(hyperarc_weights_df.iloc[i, 1])

        examp_succ_df = pd.DataFrame(
            {"Hyperarc": examp_succ_hyps, "Weight": examp_succ_hyp_weights}
        )
        st.dataframe(examp_succ_df)

        st.markdown(
            f"The probability of transitioning from {all_dis_pairs[1][0]} to "
            f"{all_dis_pairs[1][1]} is the sum of the weights of these "
            f"hyperarcs: {examp_succ_df['Weight'].sum()}."
        )

        st.markdown("__All node pairs__")
        st.markdown(
            "If we do this for all possible node combinations we get "
            "the non-normalised successor transition matrix:"
        )
        st.dataframe(nn_succ_trans_df.round(2))

        succ_trans_df = nn_succ_trans_df.div(nn_succ_trans_df.sum(axis=1), axis=0)
        succ_trans_df = succ_trans_df.round(2)
        # st.dataframe(succ_trans_df)

    tab2.markdown(
        "Following the steps above we get the normalised successor "
        "transition matrix, where each row sums to 1:"
    )
    tab2.dataframe(succ_trans_df)  # .style.highlight_max(axis=0))
    tab2.write("#### Predecessor Transition Matrix")

    tab2.subheader("PageRank:")

    tab2.write("#### Successor PageRank")
    tab2.text("Disease most likely to be successor diseases...")

    tab2.write("#### Predecessor PageRank")
    tab2.text("Disease most likely to be predecessor diseases...")
