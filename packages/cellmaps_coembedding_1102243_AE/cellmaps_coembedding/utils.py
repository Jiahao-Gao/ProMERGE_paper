import random
from scipy.stats import mannwhitneyu, gaussian_kde
import pandas as pd
import matplotlib.pyplot as plt
from cellmaps_coembedding.muse_sc.df_utils import *
import numpy as np
import seaborn as sns
import umap


def plot_kde_from_df(df, outdir, kde_figure='sim_muse.png'):
    """
    Plots the KDE (Kernel Density Estimate) of similarity scores for CORUM and non-CORUM pairs.

    :param df: A DataFrame with 'Similarity' values and 'Type' labels ('CORUM pair' or 'non-CORUM pair')
    :type df: pandas.DataFrame
    :param outdir: Output directory
    :type outdir: str
    :param kde_figure: The filename (or path) to save the resulting figure, defaults to 'sim_muse.png'
    :type kde_figure: str, optional
    :return: This function does not return anything. It saves a figure and closes the plot.
    :rtype: None
    """
    df_corum = df[df['Type'] == 'CORUM pair']['Similarity'].dropna()
    df_non_corum = df[df['Type'] == 'non-CORUM pair']['Similarity'].dropna()
    kde_corum = gaussian_kde(df_corum)
    kde_non_corum = gaussian_kde(df_non_corum)
    x_min = df['Similarity'].min()
    x_max = df['Similarity'].max()
    x = np.linspace(x_min, x_max, 200)
    plt.figure(figsize=(8, 6))
    plt.plot(x, kde_corum(x), color='green', label='CORUM pair')
    plt.plot(x, kde_non_corum(x), color='grey', label='non-CORUM pair')
    plt.xlabel('Similarity')
    plt.ylabel('Density')
    plt.ylim(bottom=0)
    plt.legend()
    plt.savefig(os.path.join(outdir, kde_figure), dpi=300, bbox_inches='tight')
    plt.close()


def plot_embedding_eval(results_df, outdir, embedding_eval_figure='embedding_eval.png'):
    """
    Plots the enrichment results (effect sizes) for various embeddings with their confidence intervals.

    :param results_df: A DataFrame with columns 'Embedding' and 'Result' (the enrichment effect sizes)
    :type results_df: pandas.DataFrame
    :param outdir: Output directory
    :type outdir: str
    :param embedding_eval_figure: The filename (or path) to save the figure, defaults to 'embedding_eval.png'
    :type embedding_eval_figure: str, optional
    :return: This function does not return anything. It saves a figure and closes the plot.
    :rtype: None
    """
    color_map = {
        "ppi": "#659AD2",
        'image': '#c30000',
        'MUSE': "#9D85BE",
        'concat': '#C8A2C8'
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    list_keys = list(set(results_df['Embedding']))
    dataset_path_keys = [x for x in list_keys if x != 'MUSE'] + ['MUSE']

    # Iterate over datasets to plot
    muse = results_df[results_df['Embedding'] == 'MUSE']['Result'].values
    for idx, embedding in enumerate(dataset_path_keys):
        subset_embedding = results_df[results_df['Embedding'] == embedding]
        mean_result = np.mean(subset_embedding['Result'].values)
        stdev = np.std(subset_embedding['Result'].values)
        bootstrap_mean_diff = muse - subset_embedding['Result'].values
        bootstrap_mean_diff_average = np.mean(bootstrap_mean_diff)
        # shift so null is true
        shifted = np.array([x - bootstrap_mean_diff_average for x in bootstrap_mean_diff])
        # p-value = how many differences are more extreme than actual mean difference if null true; add 1 so p-value
        # is not 0
        pval = float((1 + len(shifted[shifted > np.abs(bootstrap_mean_diff_average)])) / (1 + len(shifted)) + (
            1 + len(shifted[shifted < -np.abs(bootstrap_mean_diff_average)])) / (1 + len(shifted)))
        sig = False
        if embedding != 'MUSE':
            if pval < 0.05:
                sig = True

        lower = mean_result - stdev
        upper = mean_result + stdev
        if embedding == 'ppi' or embedding == 'image':
            shape = '^'
        else:
            shape = 'o'
        # Plot the horizontal bars for confidence intervals in grey
        ax.hlines(2 + 0.2 * (idx - len(dataset_path_keys) / 2), lower, upper, color='grey',
                  alpha=0.3)  # Set the color to grey
        alpha = 1.0 if embedding == "MUSE" else 0.6  # Full opacity for "CO-EMBEDDING", reduced for others
        ax.scatter(mean_result, 2 + 0.2 * (idx - len(dataset_path_keys) / 2), alpha=alpha, color=color_map[embedding],
                   marker=shape, label=embedding, s=300)
        if sig:
            ax.text(mean_result, 2 + 0.2 * (idx - len(dataset_path_keys) / 2), '*', fontsize=15, color='black',
                    ha='center', va='center')

    plt.yticks([])
    plt.xlabel('Enrichment')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlim((0, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, embedding_eval_figure), dpi=300, bbox_inches='tight')
    plt.close()


def get_embedding_eval_data(dataset_paths, edges, num_samplings, num_edges, kde_file, eval_file, common_proteins=None):
    """
    Calculates enrichment effect sizes for different embeddings based on a reference adjacency (e.g., CORUM).
    It also saves data for KDE plotting from the MUSE embedding.

    :param dataset_paths: Dictionary with keys as embedding names ('ppi', 'image', 'MUSE', etc.)
                         and values as paths to TSV files with embedding vectors. Each file must
                         be indexed by protein IDs.
    :type dataset_paths: dict
    :param edges: A DataFrame representing a square adjacency matrix (e.g., CORUM). Rows/columns
                  are protein IDs, and the entries are 1 (same complex) or 0 (different complex).
    :type edges: pandas.DataFrame
    :param num_samplings: Number of bootstrap iterations for effect size calculation
    :type num_samplings: int
    :param num_edges: Number of edges to sample for both CORUM (true) and non-CORUM (false) in each bootstrap
    :type num_edges: int
    :param kde_file: Path to CSV file where MUSE similarity scores (CORUM vs. non-CORUM) are saved
    :type kde_file: str
    :param eval_file: Path to CSV file where computed effect sizes for each embedding are saved
    :type eval_file: str
    :param common_proteins: If provided, restricts the analysis to this set of proteins. Otherwise, it uses
                            the intersection of all embeddings’ protein IDs.
    :type common_proteins: set, optional
    :return: This function does not return anything. It saves two CSV files:
             - `kde_file` with MUSE similarity scores
             - `eval_file` with effect sizes for each embedding
    :rtype: None
    """
    dataframes = {dataset: pd.read_table(dataset_paths[dataset], index_col=0) for dataset in dataset_paths.keys()}
    if 'ppi' in dataset_paths.keys() and 'image' in dataset_paths.keys():
        df1 = dataframes['ppi']
        df2 = dataframes['image']
        df_concatenated = pd.concat([df1, df2], axis=1, ignore_index=True)
        dataframes['concat'] = df_concatenated

    # Placeholder for results and eval set sizes
    results = []

    index_sets = [set(df.index) for df in dataframes.values()]
    if common_proteins is None:
        common_proteins = set.intersection(*index_sets)

    # Calculate for each latent embedding
    for modality_name, df in dataframes.items():
        eval_set = list(common_proteins.intersection(edges.index))
        filtered_edges = edges.loc[eval_set, eval_set]
        if df.shape[0] == df.shape[1]:
            sim_df = df.loc[eval_set, eval_set]
        else:
            sim_df = cosine_similarity_scaled(df.loc[eval_set])
        filtered_df = sim_df.loc[eval_set, eval_set]
        upper_tri_indices = np.triu_indices_from(filtered_df, k=1)
        sim_df_values = filtered_df.values[upper_tri_indices]
        filtered_edges_values = filtered_edges.values[upper_tri_indices]

        for _ in np.arange(num_samplings):
            true = random.choices(sim_df_values[filtered_edges_values == 1], k=num_edges)
            false = random.choices(sim_df_values[filtered_edges_values == 0], k=num_edges)
            u, p = mannwhitneyu(true, false, alternative='greater')
            effect_size = (u / (len(true) * len(false)) - 0.5) * 2
            results.append({'Embedding': modality_name, 'Result': effect_size})

        if modality_name == 'MUSE':
            true = sim_df_values[filtered_edges_values == 1]
            false = sim_df_values[filtered_edges_values == 0]
            kde_data = pd.DataFrame({
                'Similarity': np.concatenate([true, false]),
                'Type': ['CORUM pair'] * len(true) + ['non-CORUM pair'] * len(false)
            })
            kde_data.to_csv(kde_file, index=False)

    results_df = pd.DataFrame(results)
    results_df.to_csv(eval_file, index=False)


def generate_embedding_evaluation_figures(coembedding, ppi=None, image=None, outdir='.', num_samplings=1000,
                                          num_edges=1000):
    """
    Generates figures and CSV files evaluating the MUSE (co-embedding) along with optional 'ppi' and 'image' embeddings.

    Steps:
      1. Loads embeddings from the provided paths.
      2. Loads a reference adjacency matrix (CORUM).
      3. Performs enrichment evaluation using Mann-Whitney U tests.
      4. Saves similarity data for KDE (MUSE embedding) and effect sizes (all embeddings).
      5. Plots and saves a KDE figure (MUSE) and an enrichment comparison plot.

    :param coembedding: Directory path containing 'coembedding_emd.tsv' for the MUSE embedding
    :type coembedding: str
    :param ppi: Directory path containing 'ppi_emd.tsv' for the PPI-based embedding (optional)
    :type ppi: str, optional
    :param image: Directory path containing 'image_emd.tsv' for the image-based embedding (optional)
    :type image: str, optional
    :param outdir: Output directory for saving figures and CSV files
    :type outdir: str, optional
    :param num_samplings: Number of bootstrap samplings for effect size calculation
    :type num_samplings: int, optional
    :param num_edges: Number of edges to sample (CORUM vs. non-CORUM) in each bootstrap iteration
    :type num_edges: int, optional
    :return: This function does not return anything, but writes two CSV files and saves two figures:
             - 'sim_muse_data.csv' (MUSE similarity scores for CORUM vs. non-CORUM)
             - 'embedding_eval.csv' (enrichment effect sizes for each embedding)
             - 'sim_muse.png' (KDE figure)
             - 'embedding_eval.png' (enrichment comparison figure)
    :rtype: None
    """
    dataset_paths = {'MUSE': os.path.join(coembedding, 'coembedding_emd.tsv')}
    if ppi:
        dataset_paths['ppi'] = os.path.join(ppi, 'ppi_emd.tsv')
    if image:
        dataset_paths['image'] = os.path.join(image, 'image_emd.tsv')

    # CORUM
    corum_file = os.path.join(os.path.dirname(__file__), 'corum_M.tsv')
    corum_M = pd.read_csv(corum_file, sep='\t', index_col=0)

    kde_file = os.path.join(outdir, 'sim_muse_data.csv')
    eval_file = os.path.join(outdir, 'embedding_eval.csv')
    get_embedding_eval_data(dataset_paths, corum_M, num_samplings=num_samplings, num_edges=num_edges, kde_file=kde_file,
                            eval_file=eval_file)

    kde_data = pd.read_csv(kde_file)
    plot_kde_from_df(kde_data, outdir)

    embedding_eval_df = pd.read_csv(eval_file)
    plot_embedding_eval(embedding_eval_df, outdir)


def generate_umap_of_embedding(emb_file, outdir, label_map=None):
    # Load the embedding
    emb = pd.read_csv(emb_file, sep='\t', index_col=0)

    # Run UMAP
    mapper = umap.UMAP(random_state=42).fit_transform(emb.values)

    # Create a DataFrame with UMAP coordinates
    emb_mapper_annot = pd.DataFrame(
        mapper,
        index=emb.index,
        columns=['UMAP1', 'UMAP2']
    )

    # If label_map is provided, map each index (gene) to its label
    if label_map is not None:
        emb_mapper_annot['Label'] = emb_mapper_annot.index.map(label_map.get)
        # Fill missing labels with 'Other'
        emb_mapper_annot['Label'].fillna('Other', inplace=True)

        # Plot UMAP colored by label
        plt.figure(figsize=(5, 5))
        sns.scatterplot(
            x='UMAP1',
            y='UMAP2',
            data=emb_mapper_annot,
            hue='Label',
            s=10,
            linewidth=0
        )
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    else:
        # Plot UMAP without coloring by label
        plt.figure(figsize=(5, 5))
        sns.scatterplot(
            x='UMAP1',
            y='UMAP2',
            data=emb_mapper_annot,
            s=10,
            linewidth=0
        )

    sns.despine()
    plt.xticks([], [])
    plt.yticks([], [])
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')

    # Save figure
    figure_save_path = os.path.join(outdir, 'embedding_umap.svg')
    plt.savefig(figure_save_path, dpi=300, bbox_inches='tight')
    plt.close()
