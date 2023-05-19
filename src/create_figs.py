import matplotlib.pyplot as plt

# import matplotlib
# import numpy as np
# import pandas as pd
# import seaborn as sns


def pagerank_scatter(suc_col, pred_col, dis_col, tab):
    """
    Create a scatterplot comparing successor PageRank to predecessor PageRank
    to observe whether diseases are more likely to be a successor or
    predecessor.

    Args:
        suc_col (df col) : PageRank successor column from dataframe.

        pred_col (df col) : PageRank predecessor column from dataframe.

        dis_col (df col) : Column of dataframe with disease labels.

    Return:
        plt figure : Scatterplot of successor and predecessor PageRank

    """
    # fig = plt.figure()
    fig, ax = plt.subplots(figsize=(2.5, 2.5))

    # Triangles
    x = [-0.01, 0.99, -0.01]
    y = [-0.01, 0.99, 0.99]

    ax.fill(x, y, color="orange", alpha=0.1)
    pred_loc_y = pred_col.max() + 0.04
    ax.text(0.001, pred_loc_y, "Predecessor", color="orange", fontsize=5)

    ax.fill(y, x, color="green", alpha=0.1)
    succ_loc_x = suc_col.max() + 0.06
    suc = "Successor"
    ax.text(succ_loc_x, 0.01, suc, color="green", rotation=90, fontsize=5)

    scat_cols = range(0, len(dis_col))
    ax.scatter(x=suc_col, y=pred_col, s=200, c=scat_cols, alpha=0.5)

    for x, y, label in zip(suc_col, pred_col, dis_col):
        ax.text(x, y, label, va="center", ha="center", fontsize=6)

    ax.set_xlim(-0.01, suc_col.max() + 0.1)
    ax.set_ylim(-0.01, pred_col.max() + 0.1)
    ax.plot(
        (-0.01, suc_col.max() + 0.1),
        (-0.01, suc_col.max() + 0.1),
        ls="--",
        color="blue",
    )
    ax.text(
        suc_col.max(),
        pred_col.max() + 0.08,
        "Transitive",
        ha="center",
        va="center",
        rotation=0,
        color="blue",
        fontsize=5,
    )
    ax.set_xlabel = "Successor PageRank"
    ax.set_ylabel = "Predecessor PageRank"
    ax.tick_params(labelsize=5)

    tab.pyplot()
