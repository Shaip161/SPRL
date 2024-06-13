from transformers import AutoModelForSequenceClassification, AutoTokenizer
from ferret.benchmark import Benchmark
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import table
import seaborn as sns
from collections import Counter, defaultdict
from matplotlib.colors import LinearSegmentedColormap

"""
get_dataframe, deduplicate_column_names, get_colormap, show_table, style_heatmap copied and slightly modified (to fix relevant bug) from the Ferret library. The original code can be found at https://github.com/g8a9/ferret .
"""

def get_dataframe(explanations):
    """Convert explanations into a pandas DataFrame.

    Args:
        explanations (List[Explanation]): list of explanations

    Returns:
        pd.DataFrame: explanations in table format. The columns are the tokens and the rows are the explanation scores, one for each explainer.
    """
    scores = {e.explainer: e.scores for e in explanations}
    scores["Token"] = explanations[0].tokens
    table = pd.DataFrame(scores).set_index("Token").T
    return table

def deduplicate_column_names(df):
    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()

    column_counts = Counter(df_copy.columns)

    new_columns = list()
    seen_names = defaultdict(int)
    for column in df_copy.columns:

        count = column_counts[column]
        if count > 1:
            new_columns.append(f"{column}_{seen_names[column]}")
            seen_names[column] += 1
        else:
            new_columns.append(column)

    df_copy.columns = new_columns
    return df_copy


def get_colormap(format):

    if format == "blue_red":
        return sns.diverging_palette(240, 10, as_cmap=True)
    elif format == "white_purple":
        return sns.light_palette("purple", as_cmap=True)
    elif format == "purple_white":
        return sns.light_palette("purple", as_cmap=True, reverse=True)
    elif format == "white_purple_white":
        colors = ["white", "purple", "white"]
        return LinearSegmentedColormap.from_list("diverging_white_purple", colors)
    else:
        raise ValueError(f"Unknown format {format}")


def show_table(
    explanations, remove_first_last, style, **style_kwargs
):
    """Format explanation scores into a colored table.

    Args:
        explanations (List[Explanation]): list of explanations
        apply_style (bool): apply color to the table of explanation scores
        remove_first_last (bool): do not visualize the first and last tokens, typically CLS and EOS tokens

    Returns:
        pd.DataFrame: a colored (styled) pandas dataframed
    """

    # Get scores as a pandas DataFrame
    table = get_dataframe(explanations)

    if remove_first_last:
        table = table.iloc[:, 1:-1]

    # add count as prefix for duplicated tokens
    table = deduplicate_column_names(table)
    if not style:
        return table.style.format("{:.2f}")

    if style == "heatmap":
        subset_info = {
            "vmin": style_kwargs.get("vmin", -1),
            "vmax": style_kwargs.get("vmax", 1),
            "cmap": style_kwargs.get("cmap", get_colormap("blue_red")),
            "axis": None,
            "subset": None,
        }
        return style_heatmap(table, [subset_info])
    else:
        raise ValueError(f"Style {style} is not supported.")

def style_heatmap(df: pd.DataFrame, subsets_info):
    """Style a pandas DataFrame as a heatmap.

    Args:
        df (pd.DataFrame): a pandas DataFrame
        subsets_info (List[Dict]): a list of dictionaries containing the style information for each subset of the DataFrame. Each dictionary should contain the following keys: vmin, vmax, cmap, axis, subset. See https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html#Building-Styles for more information.

    Returns:
        pd.io.formats.style.Styler: a styled pandas DataFrame
    """

    style = df.style
    for si in subsets_info:
        style = style.background_gradient(**si)

    # Set stick index
    style = style.set_sticky(axis="index")

    style = style.format("{:.2f}")
    return df


name = "roberta-large-mnli"
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained(name)

bench = Benchmark(model, tokenizer)
explanations = bench.explain('He listens impassively . He was/were sentient.', target=2)
explanation_evaluations = bench.evaluate_explanations(explanations, target=2)

df = show_table(explanations, True, "heatmap")
with open('explain_listen_sentient.html', 'w') as f:
    f.write(df.to_html())


df_eval = bench.show_evaluation_table(explanation_evaluations)
with open('eval_listen_sentient.html', 'w') as f:
    f.write(df_eval.to_html())