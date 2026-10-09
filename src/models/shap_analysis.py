import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

save_dir = "reports/shap"
os.makedirs(save_dir, exist_ok=True)


def make_shap_importance_table(shap_values_by_output, feature_names, output_names):
    mean_shap_val = np.mean(np.abs(shap_values_by_output), axis=0)
    return pd.DataFrame(mean_shap_val, index=feature_names, columns=output_names)


def plot_shap_barplot(importance_df, output_name, save_dir, top_n=None):
    """
    Bar plot of mean absolute SHAP values for one output.
    """

    df = importance_df[output_name].copy()
    df = df.sort_values(ascending=False)

    if top_n is not None:
        df = df.head(top_n)

    fig, ax = plt.subplots(figsize=(7, 5))

    y_pos = np.arange(len(df))

    ax.barh(y_pos, df.values)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df.index)

    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_ylabel("Input feature")
    ax.set_title(f"SHAP feature importance — {output_name}")

    plt.tight_layout()

    plt.savefig(save_dir, dpi=300, bbox_inches="tight")
    plt.close()


def plot_shap_heatmap(importance_df, save_dir):

    heatmap_df = importance_df.copy()

    fig, ax = plt.subplots(figsize=(12, 6))

    im = ax.imshow(heatmap_df.values, aspect="auto", vmin=0)

    ax.set_xticks(np.arange(len(heatmap_df.columns)))
    ax.set_yticks(np.arange(len(heatmap_df.index)))

    ax.set_xticklabels(heatmap_df.columns, rotation=45, ha="right",fontsize=17)
    ax.set_yticklabels([r"$R_{rs}$" + f"({rrs})" for rrs in heatmap_df.index], fontsize=17)
    ax.set_xlabel("Pigment", fontsize=25)
    # ax.set_ylabel(r"$R_{rs} [nm]$")
    label = r"$m(|SHAP|)\ \ [\%]$"

    cb = fig.colorbar(im, cmap=plt.cm.Greens, ax=ax)
    cb.set_label(label, fontsize=17)
    cb.ax.tick_params(labelsize=20)

    ax.set_title("")

    plt.tight_layout()

    plt.savefig(save_dir, dpi=300, bbox_inches="tight")
    return heatmap_df


# ============================================================
# Main SHAP study
# ============================================================

def run_shap_study(model, x_train, x_test, output_names):
    feature_names = x_train.columns.values
    os.makedirs(save_dir, exist_ok=True)

    background = x_train.values
    x_explain = x_test.values

    explainer = shap.GradientExplainer(model, background)

    shap_values = explainer.shap_values(x_explain)

    importance_df = make_shap_importance_table(shap_values, feature_names, output_names)

    return shap_values, importance_df
