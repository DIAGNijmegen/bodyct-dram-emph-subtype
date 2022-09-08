""""
 Adjusted based on https://github.com/fcakyon/confplot
"""
from typing import List
from typing import Optional
from typing import Tuple

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from matplotlib.collections import QuadMesh
from matplotlib.figure import Axes
from matplotlib.figure import Figure
from matplotlib.text import Text
from pandas import DataFrame
from sklearn.metrics import confusion_matrix


def get_new_fig(name: str, fig_size: int) -> Tuple[Figure, Axes]:
    """Init graphics"""
    fig = plt.figure(name, (fig_size, fig_size))
    ax = fig.gca()
    ax.cla()
    return fig, ax


def config_cell_text_and_colors(
    array_df: np.ndarray,
    row_idx: int,
    col_idx: int,
    text: Text,
    face_colors,
    posi: int,
    font_size: int,
):
    text_add = []
    text_del = []
    cell_val = array_df[row_idx][col_idx]
    total_val = array_df[-1][-2]
    cell_per = (float(cell_val) / total_val) * 100
    curr_column = array_df[:, col_idx]
    n_rows = len(curr_column)
    font_prop = fm.FontProperties(weight='bold', size=font_size)
    # last line and/or last column
    if (col_idx >= (n_rows - 1)) or (row_idx == (n_rows - 1)):
        if col_idx == n_rows:
            per_ok = cell_val * 100
            per_err = 0
        elif cell_val != 0:
            if col_idx == n_rows - 1 and row_idx == n_rows - 1:
                tot_rig = sum([array_df[i][i] for i in range(array_df.shape[0] - 1)])
            elif col_idx == n_rows - 1:
                tot_rig = array_df[row_idx][row_idx]
            else:
                tot_rig = array_df[col_idx][col_idx]
            per_ok = (float(tot_rig) / cell_val) * 100
            per_err = 100 - per_ok
        else:
            per_ok = per_err = 0

        per_ok_s = ['%.2f%%' % per_ok, '100%'][per_ok == 100]

        # text to DEL
        text_del.append(text)

        # text to ADD

        text_kwargs = [
            dict(color='r', ha="center", va="center", gid='sum', fontproperties=font_prop),
            dict(color='lime', ha="center", va="center", gid='sum', fontproperties=font_prop),
            dict(color='salmon', ha="center", va="center", gid='sum', fontproperties=font_prop),
        ]
        if col_idx == row_idx:
            text_fmt = ['%d' % cell_val, per_ok_s, 'Accuracy']
        elif col_idx == n_rows:
            text_fmt = ['', per_ok_s, '']
        else:
            text_fmt = ['%d' % cell_val, per_ok_s, '%.2f%%' % per_err]

        text_pos = [(text._x, text._y - 0.3), (text._x, text._y), (text._x, text._y + 0.3)]
        for i in range(len(text_pos)):
            text_add.append(
                dict(x=text_pos[i][0], y=text_pos[i][1], text=text_fmt[i], kw=text_kwargs[i])
            )

        # set background color for sum cells (last line and last column)
        carr = [0.27, 0.30, 0.27, 1.0]
        if (col_idx >= n_rows - 1) and (row_idx == n_rows - 1):
            carr = [0.17, 0.20, 0.17, 1.0]
        face_colors[posi] = carr

    else:
        if cell_per > 0:
            txt = '%d\n%.2f%%' % (cell_val, cell_per)
        else:
            txt = '0'
        text.set_text(txt)
        text.set_color('r')
    return text_add, text_del


def insert_overall_metrics(df_cm: pd.DataFrame):
    """insert total column and line (the last ones)"""
    sum_col = []
    for c in df_cm.columns:
        sum_col.append(df_cm[c].sum())
    sum_row = []
    for row in df_cm.iterrows():
        sum_row.append(row[1].sum())
    f1 = []
    for i in range(0, len(df_cm.index)):
        _p = df_cm.iat[i, i] / sum_col[i]
        _q = df_cm.iat[i, i] / sum_row[i]
        _f1 = 2 * (_p * _q) / (_p + _q)
        f1.append(0 if _f1 != _f1 else _f1)

    df_cm['Precision'] = sum_row
    df_cm['F1-Score'] = f1

    sum_col.append(np.sum(sum_row))
    sum_col.append(np.average(f1))
    df_cm.loc['Recall'] = sum_col


def pretty_plot_confusion_matrix(
    df_cm: pd.DataFrame,
    annot: bool = True,
    cmap: str = "Oranges",
    font_size: int = 11,
    line_width: float = 0.5,
    cbar: bool = False,
    fig_size: int = 7,
):
    df_cm = df_cm.T

    # create "Total" column
    insert_overall_metrics(df_cm)

    # this is for print always in the same window
    fig, ax = get_new_fig('Conf matrix default', fig_size)
    fig.patch.set_facecolor('white')

    ax = sn.heatmap(
        df_cm,
        annot=annot,
        annot_kws={"size": font_size},
        linewidths=line_width,
        ax=ax,
        cbar=cbar,
        cmap=cmap,
        linecolor='w',
        fmt='.2f',
    )

    # set tick labels rotation
    ax.set_xticklabels(ax.get_xticklabels(), rotation=-30, fontsize=10, ha='left')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=25, fontsize=10)

    # axis settings: avoiding cutting off edges
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)

    # turn off all the ticks
    for t in ax.xaxis.get_major_ticks():
        t.tick1line.set_visible(False)
        t.tick2line.set_visible(False)
    for t in ax.yaxis.get_major_ticks():
        t.tick1line.set_visible(False)
        t.tick1line.set_visible(False)

    # face colors list
    quad_mesh = ax.findobj(QuadMesh)[0]
    face_colors = quad_mesh.get_facecolors()

    # iter in text elements
    array_df = df_cm.to_numpy()
    text_add = []
    text_del = []
    posi = -1  # from left to right, bottom to top.
    for text in ax.collections[0].axes.texts:  # ax.texts:
        pos = np.array(text.get_position()) - [0.5, 0.5]
        y_idx = int(pos[1])
        x_idx = int(pos[0])
        posi += 1
        # set text
        txt_res = config_cell_text_and_colors(
            array_df, y_idx, x_idx, text, face_colors, posi, font_size
        )

        text_add.extend(txt_res[0])
        text_del.extend(txt_res[1])

    # remove the old ones
    for item in text_del:
        item.remove()
    # append the new ones
    for item in text_add:
        ax.text(item['x'], item['y'], item['text'], **item['kw'])

    # titles and legends
    ax.set_title('Confusion matrix')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predict')
    plt.tight_layout()


def plot_confusion_matrix_from_data(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[int],
    columns: Optional[List[str]] = None,
    annot: bool = True,
    cmap: str = "Greys",
    font_size: int = 11,
    line_width: float = 0.5,
    cbar: bool = False,
    fig_size: int = 7,
):
    """
    plot confusion matrix function with y_true (actual values) and y_pred (predictions),
    without a confusion matrix yet
    """
    if columns is None:
        columns = [f"class {i}" for i in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    df_cm = DataFrame(cm, index=columns, columns=columns)
    pretty_plot_confusion_matrix(
        df_cm,
        annot=annot,
        cmap=cmap,
        fig_size=fig_size,
        cbar=cbar,
        font_size=font_size,
        line_width=line_width,
    )

    return plt.gcf()
