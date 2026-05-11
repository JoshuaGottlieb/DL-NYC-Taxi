from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import seaborn as sns

# Custom utility functions for formatting plots
from modules.plotting_utils import (
    snake_to_title,
    snake_to_title_axes,
    snake_to_title_ticks,
    generate_title
)

# Custom utility functions for calculating statistics
from modules.statistics import compute_pairwise_associations, compute_kruskal_wallis

# Custom functions for loading data
from modules.utils import load_epoch_data

# ---- Notes ----
# There is probably some additional refactoring that can be done to extract common function logic

# ---- Plotting Functions for EDA ----
# ---- Correlations, Associations, and Kruskal-Wallis Plots ----

def correlation_heatmap(dataframe: pd.DataFrame, figsize: tuple[float, float] = (12, 6)) -> plt.Figure:
    """
    Draw side-by-side Pearson and Spearman correlation heatmaps with title-cased axis labels.

    Args:
        dataframe (pd.DataFrame):
            Input dataframe containing numeric columns to correlate.
        figsize (tuple, optional):
            Figure size if axes are created. Defaults to (12, 6).

    Returns:
        matplotlib.axes.Axes:
            List containing the two axes objects (Pearson and Spearman heatmaps).
    """
    # Create axes, two subplots side-by-side, share y axis
    fig, ax = plt.subplots(ncols = 2, figsize = figsize, sharey = True)

    # Compute absolute correlations
    pearson_corr = dataframe.corr(method = 'pearson', numeric_only = True).abs()
    spearman_corr = dataframe.corr(method = 'spearman', numeric_only = True).abs()

    # Draw heatmaps
    sns.heatmap(pearson_corr, annot = True, cbar = False, ax = ax[0], fmt = ".2f")
    sns.heatmap(spearman_corr, annot = True, cbar = False, ax = ax[1], fmt = ".2f")

    # Set titles using consistent utility
    ax[0].set_title(generate_title(x = "Pearson Correlation"))
    ax[1].set_title(generate_title(x = "Spearman Correlation"))

    # Convert tick labels to title case
    for i, a in enumerate(ax):
        # Only modify y-labels for the first axis, since sharey = True
        snake_to_title_ticks(a, y = i == 0, rotation_x = 45, rotation_y = 0)

    plt.tight_layout(pad = 4) # Padding to use shared y-labels in readable manner
    
    return fig

def plot_correlation_differences(summary_df: pd.DataFrame, figsize: tuple = (10, 6),
                                 cmap: str = "coolwarm", tick_rotation: int = 45) -> plt.Figure:
    """
    Plot heatmaps showing the differences in correlation coefficients
    (r_comparison - r_reference) between datasets.

    For each comparison dataset in `summary_df`, a heatmap is generated
    showing pairwise correlation differences between variables, arranged
    in side-by-side subplots.

    Args:
        summary_df (pd.DataFrame):
            A summary DataFrame containing the following columns:
            ['variable_1', 'variable_2', 'r_reference', 'r_comparison',
             'comparison_dataset', 'reference_dataset'].
        figsize (tuple, optional):
            Base figure size for a single heatmap; width scales with the number of subplots.
            Defaults to (10, 6).
        cmap (str, optional):
            Colormap used for visualizing correlation differences. Defaults to "coolwarm".
        tick_rotation (int, optional):
            Rotation angle for x-tick labels on the heatmaps. Defaults to 45.

    Returns:
        matplotlib.figure.Figure:
            The figure object containing one or more correlation difference heatmaps.
    """
    
    # Identify all unique comparison datasets (each will be one subplot)
    comparison_datasets = summary_df["comparison_dataset"].unique()
    n_datasets = len(comparison_datasets)

    # Create subplots: one per comparison dataset
    fig, axes = plt.subplots(
        1, n_datasets,
        figsize = (figsize[0] * n_datasets, figsize[1]),  # Scale width with number of subplots
        sharey = True  # Share y-axis labels for consistency across plots
    )

    # If there's only one dataset, wrap the single Axes object in a list for consistent iteration
    if n_datasets == 1:
        axes = [axes]

    # Loop through each comparison dataset to create individual heatmaps
    for ax, comp_name in zip(axes, comparison_datasets):
        # Subset data for the current comparison dataset
        df_plot = summary_df[summary_df["comparison_dataset"] == comp_name].copy()

        # Compute difference in correlation between comparison and reference
        df_plot["r_diff"] = df_plot["r_comparison"] - df_plot["r_reference"]

        # Pivot data into a matrix format suitable for heatmap plotting
        heatmap_data = df_plot.pivot(index = "variable_1", columns = "variable_2", values = "r_diff")

        # Plot the heatmap for correlation differences
        sns.heatmap(
            heatmap_data,
            annot = True,     # Show correlation difference values in each cell
            fmt = ".2f",      # Format annotations to two decimal places
            cmap = cmap,      # Use diverging color map centered at 0
            center = 0,       # Center color scale at zero (no difference)
            cbar = False,     # Hide colorbar for cleaner subplot layout
            ax = ax
        )
        
        # Remove axis labels for cleaner layout
        ax.set_xlabel("")
        ax.set_ylabel("")

        # Create subplot title showing comparison vs. reference dataset
        ref_name = df_plot["reference_dataset"].iloc[0]
        ax.set_title(f"{snake_to_title(comp_name)} - {snake_to_title(ref_name)}")

    # Format tick labels, only format y-axis for the first subplot due to sharey = True
    # Needs to be called outside of the loop since sns.heatmap overwrites y-ticks with sharey = True
    for i, ax in enumerate(axes):
        snake_to_title_ticks(ax, x = True, y = i == 0, rotation_x = tick_rotation, rotation_y = 0)
    
    # Adjust subplot spacing for readability
    plt.tight_layout(pad = 4)

    return fig

def association_heatmap(dataframe: pd.DataFrame, figsize: tuple = (12, 6)) -> plt.Figure:
    """
    Draw side-by-side heatmaps for Cramer's V and Cohen's Omega statistics 
    for association between categorical variables, with title-cased axis labels.

    Args:
        dataframe (pd.DataFrame):
            Input dataframe containing categorical columns.
        figsize (tuple, optional):
            Figure size if axes are created. Defaults to (12, 6).

    Returns:
        list of matplotlib.axes.Axes:
            List containing the two axes objects (Cramer's V and Cohen's Omega heatmaps).
    """
    # Create axes, two subplots side-by-side, share y axis
    fig, ax = plt.subplots(ncols = 2, figsize = figsize, sharey = True)

    # Compute pairwise Cramer's V and Cohen's Omega
    associations_df = compute_pairwise_associations(dataframe)
    
    # Pivot data for heatmaps
    pivot_dataframes = []
    for stat in ['cramers_v', 'cohens_omega']:
        pivot = associations_df.pivot(index = 'column1', columns = 'column2', values = stat)
        pivot.columns.name = None
        pivot.index.name = None
        pivot_dataframes.append(pivot)

    # Draw heatmaps
    sns.heatmap(pivot_dataframes[0], fmt = '.2g', annot = True, cbar = False, ax = ax[0])
    sns.heatmap(pivot_dataframes[1], fmt = '.2g', annot = True, cbar = False, ax = ax[1])

    # Set titles using consistent utility
    ax[0].set_title(generate_title(x = "Cramer's V"))
    ax[1].set_title(generate_title(x = "Cohen's Omega"))

    # Convert tick labels to title case
    for i, a in enumerate(ax):
        # Only configure y ticks for first axis, since sharey = True
        snake_to_title_ticks(a, y = i == 0, rotation_x = 90, rotation_y = 0)

    plt.tight_layout(pad = 2) # Padding to use shared y-labels in readable manner
    
    return fig

def plot_association_differences(datasets: List[pd.DataFrame], reference_index: int = 0,
                                 dataset_names: Optional[List[str]] = None,
                                 figsize: tuple = (10, 6), cmap: str = "coolwarm",
                                 tick_rotation: int = 90, omega_threshold: float = 0.1) -> plt.Figure:
    """
    Plot heatmaps of differences in pairwise associations (Cohen's Omega)
    between a reference dataset and comparison datasets, masking pairs
    where Cohen's Omega <= omega_threshold in both datasets.

    Args:
        datasets (List[pd.DataFrame]): List of DataFrames with categorical columns.
        reference_index (int, optional): Index of the reference dataset. Defaults to 0.
        dataset_names (List[str], optional): Names for the datasets. Defaults to "dataset_0", ...
        figsize (tuple, optional): Base figure size. Width scales with number of subplots.
        cmap (str, optional): Colormap for the heatmaps.
        tick_rotation (int, optional): Rotation of x-tick labels.
        omega_threshold (float, optional): Minimum Cohen's Omega to show a pair.

    Returns:
        matplotlib.figure.Figure: Figure object containing heatmap subplots.
    """

    # Default dataset names if none provided
    if dataset_names is None:
        dataset_names = [f"dataset_{i}" for i in range(len(datasets))]

    # Define the reference dataset and compute its pairwise associations
    ref_name = dataset_names[reference_index]
    ref_assoc = compute_pairwise_associations(datasets[reference_index])

    # Indices of datasets to compare against the reference
    comparison_indices = [i for i in range(len(datasets)) if i != reference_index]

    # Create one subplot per comparison dataset
    fig, axes = plt.subplots(
        1, len(comparison_indices),
        figsize = (figsize[0] * len(comparison_indices), figsize[1]),
        sharey = True  # Share y-axis to align variable names across subplots
    )

    # Ensure axes is iterable even for a single subplot
    if len(comparison_indices) == 1:
        axes = [axes]

    for ax, idx in zip(axes, comparison_indices):
        comp_name = dataset_names[idx]

        # Compute pairwise associations for the comparison dataset
        comp_assoc = compute_pairwise_associations(datasets[idx])

        # Merge association tables on column pairs
        merged = ref_assoc.merge(comp_assoc, on = ['column1', 'column2'], suffixes = ('_ref', '_comp'))

        # Mask pairs where both datasets have low association strength
        mask = (
            (merged['cohens_omega_ref'] <= omega_threshold) &
            (merged['cohens_omega_comp'] <= omega_threshold)
        )
        merged.loc[mask, 'cohens_omega_comp'] = np.nan

        # Compute difference in Cohen's Omega between comparison and reference datasets
        merged['cohens_omega_diff'] = merged['cohens_omega_comp'] - merged['cohens_omega_ref']

        # Reshape for heatmap plotting
        heatmap_data = merged.pivot(index = 'column1', columns = 'column2', values = 'cohens_omega_diff')

        # Plot the heatmap for the current comparison dataset
        sns.heatmap(
            heatmap_data,
            annot = True,     # Show association difference values in each cell
            fmt = ".2f",      # Format annotations to two decimal places
            cmap = cmap,      # Use diverging color map centered at 0
            center = 0,       # Center color scale at zero (no difference)
            cbar = False,     # Hide colorbar for cleaner subplot layout
            ax = ax
        )

        # Remove axis labels for cleaner presentation
        ax.set_xlabel("")
        ax.set_ylabel("")

        # Add title showing comparison direction (e.g., Dataset B - Dataset A)
        ax.set_title(f"{snake_to_title(comp_name)} - {snake_to_title(ref_name)}")

    # Format tick labels, only format y-axis for the first subplot due to sharey = True
    # Needs to be called outside of the loop since sns.heatmap overwrites y-ticks with sharey = True
    for i, ax in enumerate(axes):
        snake_to_title_ticks(ax, x = True, y = i == 0, rotation_x = tick_rotation, rotation_y = 0)
    
    # Adjust spacing between subplots for readability
    plt.tight_layout(pad = 4)

    return fig

def plot_kruskal_wallis_heatmap(dataframes: Union[pd.DataFrame, List[pd.DataFrame]],
                                continuous_var: str, categorical_vars: Optional[List[str]] = None,
                                dataset_names: Optional[List[str]] = None,
                                significance_level: float = 0.05, show_only_significant: bool = True,
                                figsize: tuple = (10, 3), cmap: str = "rocket",
                                title: Optional[str] = None, tick_rotation: int = 90) -> plt.Figure:
    """
    Plot Kruskal–Wallis H-statistics as heatmaps for one or more datasets,
    using a shared continuous variable across all datasets.

    Each dataset is plotted in its own row, sharing the same y-axis categories.

    Args:
        dataframes (pd.DataFrame | List[pd.DataFrame]): 
            Single DataFrame or list of DataFrames to analyze.
        continuous_var (str): 
            The continuous variable to test against categorical variables.
        categorical_vars (List[str], optional): 
            Categorical variables to test. If None, use all categorical columns.
        dataset_names (List[str], optional): 
            Names for datasets. Defaults to "Dataset 1", "Dataset 2", etc.
        significance_level (float, optional): 
            p-value threshold for significance masking. Defaults to 0.05.
        show_only_significant (bool, optional): 
            If True, mask results where p >= significance_level. Defaults to True.
        figsize (tuple, optional): 
            Overall figure size. Defaults to (10, 3).
        cmap (str, optional): 
            Colormap for heatmaps. Defaults to "rocket".
        title (str, optional): 
            Figure title. Defaults to generated title.
        tick_rotation (int, optional): 
            Rotation angle for x-axis tick labels. Defaults to 90.

    Returns:
        matplotlib.figure.Figure: Figure containing one or more dataset heatmaps.
    """
    # Normalize input
    # Convert a single DataFrame into a list for consistent handling
    if isinstance(dataframes, pd.DataFrame):
        dataframes = [dataframes]
        if dataset_names is None:
            dataset_names = ["Dataset 1"]
    else:
        # Auto-generate names if not provided
        if dataset_names is None:
            dataset_names = [f"Dataset {i + 1}" for i in range(len(dataframes))]

    # Compute Kruskal–Wallis statistics for each dataset
    results = []
    for name, df in zip(dataset_names, dataframes):
        res = compute_kruskal_wallis(df, continuous_var, categorical_vars)
        res["dataset"] = name
        results.append(res)

    # Combine all results into a single DataFrame for visualization
    all_results = pd.concat(results, ignore_index = True)

    # Prepare pivot tables for plotting
    # H-statistics for heatmap values, p-values for masking
    pivot_h = all_results.pivot(index = "categorical_var", columns = "dataset", values = "H_statistic")
    pivot_p = all_results.pivot(index = "categorical_var", columns = "dataset", values = "p_value")

    # Mask non-significant values (p >= significance_level)
    if show_only_significant:
        pivot_h = pivot_h.where(pivot_p < significance_level)

    # Configure subplot grid
    n_datasets = len(dataset_names)
    fig, axes = plt.subplots(
        n_datasets, 1,
        figsize = (figsize[0], figsize[1] * n_datasets),
        sharex = True  # Shared x-axis aligns categories across datasets
    )

    # Ensure axes is iterable even for one dataset
    if n_datasets == 1:
        axes = [axes]

    # Plot each dataset’s heatmap
    for ax, name in zip(axes, dataset_names):
        # Select H-statistics for this dataset only
        subset = pivot_h[[name]].T

        # Draw heatmap with annotated H-statistics
        sns.heatmap(
            subset,
            cmap = cmap,
            annot = True,
            fmt = ".2f",
            linewidths = 0.5,
            cbar = False,
            ax = ax
        )

        # Title and axis cleanup
        ax.set_title(name, fontsize = 12)
        ax.set_xlabel("")
        ax.set_ylabel("")

        # Label y-axis with the continuous variable
        ax.set_yticklabels([snake_to_title(continuous_var)], rotation = 0)

    # Format tick labels, only format x-axis for the last subplot due to sharex = True
    # Needs to be called outside of the loop since sns.heatmap overwrites x-ticks with sharex = True
    for i, ax in enumerate(axes):
        snake_to_title_ticks(ax, x = i == len(axes) - 1, y = True, rotation_x = tick_rotation, rotation_y = 0)
    
    # Final figure formatting
    if title is None:
        title = f"Kruskal–Wallis H-Statistics for '{snake_to_title(continuous_var)}'"
    fig.suptitle(title, fontsize = 14)
    plt.tight_layout()

    return fig


# ---- Custom Plotting Functions Using Seaborn ----

def custom_countplot(dataframe: pd.DataFrame, x: str, 
                     plot_order: Optional[List[str] | str] = "auto",
                     stat: str = "count", figsize: tuple = (10, 6),
                     title: Optional[str] = None, tick_rotation: Optional[int] = 0,
                     ticklabels: Optional[List[str]] = None,
                     xlabel: Optional[str] = None,
                     ax: Optional[Axes] = None, color: Optional[str] = None,
                     label: Optional[str] = None, alpha: float = 1.0) -> Axes:
    """
    Draw a customized countplot with optional axis reuse, automatic title generation,
    formatted tick labels, and control over bar transparency.
    
    Args:
        dataframe (pd.DataFrame):
            Input DataFrame containing the data for plotting.
        x (str):
            Column name to use for the categorical x-axis.
        plot_order (list or str, optional):
            Order of categories along the x-axis.
            'auto' (default) sorts categories by descending frequency.
            None keeps original order. A list specifies a custom order.
        stat (str, optional):
            Statistic to plot. Defaults to 'count'.
        figsize (tuple, optional):
            Figure size if a new axis is created. Defaults to (10, 6).
        title (str, optional):
            Plot title. If None, a title is generated automatically.
        tick_rotation (int, optional):
            Rotation angle for x-axis tick labels. Defaults to 0.
        ticklabels (list of str, optional):
            Custom labels for the x-axis ticks. Must match number of categories.
        xlabel (str, optional):
            Custom label for the x-axis. If None, auto-generates from `x`.
        ax (matplotlib.axes.Axes, optional):
            Axis to draw the plot on. If None, a new axis is created.
        color (str, optional):
            Bar color when no hue is provided. Defaults to None.
        label (str, optional):
            Label for the bars to include in a legend. Defaults to None.
        alpha (float, optional):
            Transparency of bars (0.0 = fully transparent, 1.0 = opaque). Defaults to 1.0.

    Returns:
        matplotlib.axes.Axes:
            The matplotlib axes object containing the plot.
    """

    # Create a new figure and axis if none provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Determine the plotting order for categorical x-axis
    if plot_order == "auto":
        plot_order = dataframe[x].value_counts().index

    # Draw the countplot
    sns.countplot(
        data = dataframe,
        x = x,
        order = plot_order,
        stat = stat,
        color = color,
        ax = ax,
        label = label,
        alpha = alpha
    )

    # Apply custom tick labels if provided
    if ticklabels is not None:
        if len(ticklabels) != len(ax.get_xticks()):
            raise ValueError("Length of `ticklabels` must match number of x-ticks.")
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(ticklabels, rotation = tick_rotation)
    else:
        snake_to_title_ticks(ax, y = False, rotation_x = tick_rotation)

    # Apply custom x-axis label
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    else:
        snake_to_title_axes(ax)

    # Set plot title
    if title is None:
        title = generate_title(x = x, hue = None, stat = stat)
    ax.set_title(title)

    return ax

def custom_histplot(dataframe: pd.DataFrame, x: str, stat: str = "proportion",
                    bins: Optional[int] = 30, binwidth: Optional[float] = None,
                    log1p: bool = False, kde: bool = False,
                    figsize: tuple = (10, 6), title: Optional[str] = None,
                    xlabel: Optional[str] = None, xlim: Optional[tuple] = None,
                    ax: Optional[Axes] = None, color: Optional[str] = None,
                    label: Optional[str] = None, alpha: float = 1.0) -> Axes:
    """
    Draw a customized histogram using seaborn with optional log1p transformation,
    KDE smoothing, and adjustable bin width.

    Args:
        dataframe (pd.DataFrame):
            Input dataframe containing the numeric variable to plot.
        x (str):
            Column name for the numeric variable to plot.
        stat (str, optional):
            Statistic for histogram ('count', 'frequency', 'proportion', etc.).
            Defaults to 'proportion'.
        bins (int, optional):
            Number of bins for the histogram. Ignored if binwidth is provided.
            Defaults to 30.
        binwidth (float, optional):
            Width of each bin. Overrides `bins` if provided. Defaults to None.
        log1p (bool, optional):
            If True, apply np.log1p transformation to x before plotting. Defaults to False.
        kde (bool, optional):
            If True, overlay a KDE curve on the histogram. Defaults to False.
        figsize (tuple, optional):
            Figure size if axis is created. Defaults to (10, 6).
        title (str, optional):
            Plot title. If None, a default title is generated. Defaults to None.
        xlabel (str, optional):
            Custom label for the x-axis. If None, the column name is used. Defaults to None.
        xlim (tuple, optional):
            Tuple specifying x-axis limits as (xmin, xmax). Defaults to None.
        ax (matplotlib.axes.Axes, optional):
            Axis to draw the plot on. If None, a new axis is created.
        color (str, optional):
            Bar color. Defaults to None.
        label (str, optional):
            Label for the histogram (for legend). Defaults to None.
        alpha (float, optional):
            Transparency of bars. Defaults to 1.0.

    Returns:
        matplotlib.axes.Axes:
            Axis containing the histogram.
    """
    # Ensure axis exists
    if ax is None:
        fig, ax = plt.subplots(figsize = figsize)

    # Prepare data
    data_to_plot = dataframe[x].copy()
    if log1p:
        data_to_plot = np.log1p(data_to_plot)

    # Draw the histogram
    sns.histplot(
        data = dataframe.assign(**{x: data_to_plot}),
        x = x,
        stat = stat,
        kde = kde,
        ax = ax,
        color = color,
        label = label,
        alpha = alpha,
        bins = None if binwidth else bins,  # use bins only if binwidth not provided
        binwidth = binwidth
    )

    # Set plot title
    if title is None:
        title_prefix = "Log1p " if log1p else ""
        title = f"{stat.title()} Histogram of {title_prefix}{snake_to_title(x)}"
    ax.set_title(title)

    # Set custom x-axis label (or use default prettified label)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    else:
        snake_to_title_axes(ax)

    # Apply x-axis limits if specified
    if xlim is not None:
        ax.set_xlim(xlim)

    # Prettify tick labels
    snake_to_title_ticks(ax, y = True)

    return ax

def custom_boxplot(dataframe: pd.DataFrame, x: Optional[str] = None, y: Optional[str] = None,
                   hue: Optional[str] = None, palette: Optional[str | list] = None,
                   orient: str = "v", log1p_x: bool = False, log1p_y: bool = False,
                   figsize: tuple = (10, 6), color: Optional[str] = None,
                   ax: Optional[Axes] = None, title: Optional[str] = None,
                   xlabel: Optional[str] = None, ylabel: Optional[str] = None,
                   xlim: Optional[tuple] = None, ylim: Optional[tuple] = None,
                   label: Optional[str] = None, xticklabels: Optional[List[str]] = None,
                   yticklabels: Optional[List[str]] = None, tick_rotation: int = 0) -> Axes:
    """
    Create a highly customizable boxplot with flexible axis control, color palettes,
    log-scale transformations, and optional tick label rotation.

    Args:
        dataframe (pd.DataFrame):
            Input dataframe containing variables for plotting.
        x (str, optional):
            Variable name for the x-axis.
        y (str, optional):
            Variable name for the y-axis.
        hue (str, optional):
            Variable defining groups for color differentiation.
        palette (str or list, optional):
            Color palette for hue levels or a predefined Seaborn palette name.
        orient (str, optional):
            Orientation of the plot — "v" for vertical or "h" for horizontal. Defaults to "v".
        log1p_x (bool, optional):
            Apply log(1 + x) transformation to x variable. Defaults to False.
        log1p_y (bool, optional):
            Apply log(1 + x) transformation to y variable. Defaults to False.
        figsize (tuple, optional):
            Figure size if a new axis is created. Defaults to (10, 6).
        color (str, optional):
            Single bar color used when `hue` is not specified.
        ax (matplotlib.axes.Axes, optional):
            Axis to draw the plot on. Creates a new one if None.
        title (str, optional):
            Plot title. If None, no title is added.
        xlabel (str, optional):
            Custom x-axis label. If None, auto-generated from `x`.
        ylabel (str, optional):
            Custom y-axis label. If None, auto-generated from `y`.
        xlim (tuple, optional):
            Limits for x-axis in the form (min, max).
        ylim (tuple, optional):
            Limits for y-axis in the form (min, max).
        label (str, optional):
            Optional legend label for the boxplot.
        xticklabels (list of str, optional):
            Custom labels for x-axis ticks.
        yticklabels (list of str, optional):
            Custom labels for y-axis ticks.
        tick_rotation (int, optional):
            Rotation angle (in degrees) for x-axis tick labels. Defaults to 0.

    Returns:
        matplotlib.axes.Axes:
            The matplotlib Axes object containing the customized boxplot.
    """

    # Ensure axis exists
    if ax is None:
        fig, ax = plt.subplots(figsize = figsize)

    # Prepare data
    df = dataframe.copy()
    if log1p_x and x is not None:
        df[x] = np.log1p(df[x])
    if log1p_y and y is not None:
        df[y] = np.log1p(df[y])

    # Draw the boxplot
    sns.boxplot(
        data = df,
        x = x,
        y = y,
        hue = hue,
        palette = palette,
        orient = orient,
        ax = ax,
        color = color if hue is None else None,
        showfliers = False
    )

    # Set title and labels
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    elif x is not None:
        ax.set_xlabel(snake_to_title(x))
    if ylabel:
        ax.set_ylabel(ylabel)
    elif y is not None:
        ax.set_ylabel(snake_to_title(y))

    # Set axis limits
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    # Set custom tick labels if provided
    if xticklabels is not None:
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(xticklabels, rotation = tick_rotation)
    if yticklabels is not None:
        ax.set_yticks(ax.get_yticks())
        ax.set_yticklabels(yticklabels)

    # Prettify tick labels
    snake_to_title_ticks(ax, x = xticklabels is None, y = yticklabels is None)

    return ax

def custom_regplot(dataframe: pd.DataFrame, x: str, y: str,
                   show_scatter: bool = True, show_regression: bool = True,
                   log1p_x: bool = False, log1p_y: bool = False,
                   subsample: Optional[float] = None, random_state: Optional[int] = None,
                   figsize: tuple = (10, 6), alpha: float = 1.0,
                   color: Optional[str] = None, label: Optional[str] = None,
                   ax: Optional[Axes] = None, title: Optional[str] = None,
                   xlabel: Optional[str] = None, ylabel: Optional[str] = None,
                   xlim: Optional[tuple] = None, ylim: Optional[tuple] = None) -> Axes:
    """
    Draw a customizable regression plot using seaborn's regplot with options for:
    log1p transformations, subsampling, and selective display of scatter and regression line.

    Args:
        dataframe (pd.DataFrame):
            Input dataframe containing the data for plotting.
        x (str):
            Column name for x-axis variable.
        y (str):
            Column name for y-axis variable.
        show_scatter (bool, optional):
            Whether to show scatter points. Defaults to True.
        show_regression (bool, optional):
            Whether to show regression line. Defaults to True.
        log1p_x (bool, optional):
            If True, apply np.log1p transformation to x. Defaults to False.
        log1p_y (bool, optional):
            If True, apply np.log1p transformation to y. Defaults to False.
        subsample (float, optional):
            Fraction of rows to randomly sample for plotting. Defaults to None.
        random_state (int, optional):
            Random seed for reproducibility. Defaults to None.
        figsize (tuple, optional):
            Figure size if new figure is created. Defaults to (10, 6).
        alpha (float, optional):
            Transparency level for plot elements (0.0–1.0). Defaults to 1.0.
        color (str, optional):
            Color of points and regression line. Defaults to None.
        label (str, optional):
            Label for legend. Defaults to None.
        ax (matplotlib.axes.Axes, optional):
            Axis to draw the plot on. If None, a new axis is created.
        title (str, optional):
            Custom plot title. If None, one is auto-generated.
        xlabel (str, optional):
            Custom x-axis label. Defaults to None.
        ylabel (str, optional):
            Custom y-axis label. Defaults to None.
        xlim (tuple, optional):
            Tuple specifying x-axis limits (xmin, xmax). Defaults to None.
        ylim (tuple, optional):
            Tuple specifying y-axis limits (ymin, ymax). Defaults to None.

    Returns:
        matplotlib.axes.Axes:
            The matplotlib axes object containing the plot.
    """

    # Create or reuse axis
    if ax is None:
        fig, ax = plt.subplots(figsize = figsize)

    # Copy and transform data
    df = dataframe.copy()
    if log1p_x:
        df[x] = np.log1p(df[x])
    if log1p_y:
        df[y] = np.log1p(df[y])

    # Subsample if requested
    if subsample is not None and 0 < subsample < 1:
        df = df.sample(frac = subsample, random_state = random_state)

    # Plot using sns.regplot
    sns.regplot(
        data = df,
        x = x,
        y = y,
        ax = ax,
        scatter = show_scatter,
        fit_reg = show_regression,
        ci = None, # Suppress confidence interval as it is computationally expensive to compute
        color = color,
        scatter_kws = {"alpha": alpha, "label": label if show_scatter else None},
        line_kws = {"alpha": alpha, "label": label if show_regression else None},
    )

    # Title generation
    if title is None:
        x_label = snake_to_title(x)
        y_label = snake_to_title(y)
        x_prefix = "Log1p " if log1p_x else ""
        y_prefix = "Log1p " if log1p_y else ""
        title = f"Regression Plot of {x_prefix}{x_label} vs {y_prefix}{y_label}"
    ax.set_title(title)

    # Axis labeling
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    # Apply axis limits if provided
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Prettify labels and ticks if not overridden
    if xlabel is None or ylabel is None:
        snake_to_title_axes(ax)
    snake_to_title_ticks(ax)

    return ax

def custom_hexbin(dataframe: pd.DataFrame, x: str, y: str, log1p_x: bool = False, log1p_y: bool = False,
                  bins: Optional[int] = None, binwidth_x: Optional[float] = None,
                  binwidth_y: Optional[float] = None, xlim: Optional[tuple[float, float]] = None,
                  ylim: Optional[tuple[float, float]] = None, figsize: int = 8,
                  color: Optional[str] = None, title: Optional[str] = None,
                  xlabel: Optional[str] = None, ylabel: Optional[str] = None) -> sns.JointGrid:
    """
    Create a hexbin plot with marginal histograms, supporting log1p transforms and
    per-axis binwidth control. Allows for axis label customization.

    Args:
        dataframe (pd.DataFrame):
            Input dataframe.
        x, y (str):
            Column names for x and y variables.
        log1p_x, log1p_y (bool, optional):
            Apply log1p transformation to x and/or y. Defaults to False.
        bins (int, optional):
            Number of bins for hexbin (passed to joint_kws).
        binwidth_x, binwidth_y (float, optional):
            Binwidth for marginal histograms along x and y.
        xlim, ylim (tuple, optional):
            Axis limits for x and y.
        figsize (int, optional):
            Figure size. Defaults to 8.
        color (str, optional):
            Color for hexbin and marginal histograms.
        title (str, optional):
            Title of the plot.
        xlabel, ylabel (str, optional):
            Custom axis labels. If None, defaults to prettified column names.

    Returns:
        sns.JointGrid: Seaborn JointGrid object.
    """

    # Prepare dataframe
    df = dataframe.copy()
    if log1p_x:
        df[x] = np.log1p(df[x])
    if log1p_y:
        df[y] = np.log1p(df[y])

    # Create jointplot
    g = sns.jointplot(
        data = df,
        x = x,
        y = y,
        kind = "hex",
        color = color,
        joint_kws = dict(
            bins = bins,
            color = color
        ),
        marginal_kws = dict(
            binwidth = binwidth_x if binwidth_x else None,
            fill = True,
            color = color
        ),
        height = figsize,
    )

    # If custom binwidth_y specified, redraw the Y marginal manually (and hide axes)
    if binwidth_y:
        g.ax_marg_y.clear()
        sns.histplot(
            df,
            y = y,
            binwidth = binwidth_y,
            fill = True,
            color = color,
            ax = g.ax_marg_y
        )
        # Turn off y marginal axes completely
        g.ax_marg_y.set_axis_off()


    # Apply limits if provided
    if xlim:
        g.ax_joint.set_xlim(xlim)
    if ylim:
        g.ax_joint.set_ylim(ylim)

    # Build title (include log info)
    if title is None:
        prefix_x = "Log1p " if log1p_x else ""
        prefix_y = "Log1p " if log1p_y else ""
        title = f"{prefix_y}{snake_to_title(y)} vs {prefix_x}{snake_to_title(x)} Hexbin Plot"
    g.fig.suptitle(title)
    g.fig.subplots_adjust(top = 0.93)

    # Axis labels (custom or prettified)
    g.ax_joint.set_xlabel(xlabel if xlabel is not None else snake_to_title(x))
    g.ax_joint.set_ylabel(ylabel if ylabel is not None else snake_to_title(y))

    return g


# ---- Meta-Plotting Functions to Compare Datasets ----
    
def overlay_plots(plot_func: Callable, dataframes: List[pd.DataFrame],
                  plot_kwargs: Optional[List[Dict[str, Any]]] = None,
                  labels: Optional[List[str]] = None,
                  palette: Optional[List[str]] = None,
                  alpha: float = 0.3, figsize: tuple = (10, 6),
                  title: Optional[str] = None) -> plt.Axes:
    """
    Overlay multiple plots from a plotting function on the same axis with transparency for overlapping.

    Args:
        plot_func (Callable):
            Plotting function that accepts a dataframe and optional kwargs.
            Must accept an 'ax' parameter.
        dataframes (List[pd.DataFrame]):
            List of DataFrames to plot.
        plot_kwargs (List[Dict[str, Any]], optional):
            List of kwargs dictionaries to pass to plot_func for each dataframe.
            If None, empty dicts are used for all dataframes.
        labels (List[str], optional):
            Labels for each dataset in the legend. Defaults to "Dataset 1", "Dataset 2", etc.
        palette (List[str], optional):
            Colors to use for each dataset. Defaults to Matplotlib "tab10" colors.
        alpha (float, optional):
            Transparency for each plot. Defaults to 0.3.
        figsize (tuple, optional):
            Figure size. Defaults to (10, 6).
        title (str, optional):
            Title for the plot.

    Returns:
        matplotlib.axes.Axes: Axis object with all plots overlaid.
    """

    n = len(dataframes)
    if n == 0:
        raise ValueError("At least one dataframe must be provided.")

    # Default kwargs per dataframe
    if plot_kwargs is None:
        plot_kwargs = [{} for _ in range(n)]
    elif isinstance(plot_kwargs, dict):
        plot_kwargs = [plot_kwargs.copy() for _ in range(n)]
    elif len(plot_kwargs) != n:
        raise ValueError("Length of plot_kwargs must match number of dataframes.")

    # Default labels
    if labels is None:
        labels = [f"Dataset {i + 1}" for i in range(n)]

    # Default color palette: use tab10
    if palette is None:
        tab10_colors = plt.get_cmap("tab10").colors
        palette = [tab10_colors[i % 10] for i in range(n)]

    # Create figure and shared axis
    fig, ax = plt.subplots(figsize = figsize)

    # Plot each dataframe on the same axis
    for df, color, label, kwargs in zip(dataframes, palette, labels, plot_kwargs):
        local_kwargs = kwargs.copy()
        local_kwargs["color"] = color
        local_kwargs["label"] = label
        local_kwargs["alpha"] = alpha
        local_kwargs["ax"] = ax  # enforce shared axis

        plot_func(df, **local_kwargs)

    # Add legend and title
    ax.legend()
    if title:
        ax.set_title(title)

    return ax

def hue_plots(plot_func: Callable, dataframes: List[pd.DataFrame],
              labels: Optional[List[str]] = None,
              hue_col: str = "Dataset",
              plot_kwargs: Optional[Dict[str, Any]] = None,
              palette: Optional[List[str]] = None,
              figsize: tuple = (10, 6),
              title: Optional[str] = None) -> plt.Axes:
    """
    Combine multiple dataframes and plot using a hue to distinguish datasets.

    Args:
        plot_func (Callable):
            Plotting function that accepts a dataframe and a `hue` argument.
            Must also accept 'ax', 'color', 'alpha', and 'label' kwargs.
        dataframes (List[pd.DataFrame]):
            List of DataFrames to combine.
        labels (List[str], optional):
            Labels corresponding to each dataframe. Defaults to "Dataset 1", "Dataset 2", etc.
        hue_col (str, optional):
            Column name to store dataset labels for hueing. Defaults to "Dataset".
        plot_kwargs (Dict[str, Any], optional):
            Keyword arguments to pass to the plotting function. Defaults to None.
        palette (List[str], optional):
            Colors to use for each dataset. Defaults to Matplotlib tab10 colors.
        figsize (tuple, optional):
            Figure size. Defaults to (10, 6).
        title (str, optional):
            Title for the plot. Defaults to None.

    Returns:
        matplotlib.axes.Axes: Axis object with the hue plot.
    """

    n = len(dataframes)
    if n == 0:
        raise ValueError("At least one dataframe must be provided.")

    # Default labels
    if labels is None:
        labels = [f"Dataset {i+1}" for i in range(n)]
    if len(labels) != n:
        raise ValueError("Length of labels must match number of dataframes.")

    # Assign dataset label column and combine dataframes
    combined_df = pd.concat(
        [df.assign(**{hue_col: label}) for df, label in zip(dataframes, labels)],
        ignore_index = True
    )

    # Default color palette
    if palette is None:
        tab10_colors = plt.get_cmap("tab10").colors
        palette = [tab10_colors[i % 10] for i in range(n)]

    # Map labels to colors
    color_dict = dict(zip(labels, palette))

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Prepare kwargs
    if plot_kwargs is None:
        plot_kwargs = {}
    plot_kwargs = plot_kwargs.copy()
    plot_kwargs.update(dict(ax = ax, hue = hue_col, palette = color_dict))

    # Call plotting function once on combined dataframe
    plot_func(combined_df, **plot_kwargs)

    # Set title if provided
    if title:
        ax.set_title(title)

    return ax

# ---- Plotting Functions for Visualizing Modeling Results ----

def epoch_plots(epoch_data: pd.DataFrame, model_name: str, optimizer: Optional[str] = None,
                title_fontsize: int = 20, label_fontsize: int = 12,
                tick_fontsize: int = 9, title_pos: Tuple[float, float] = (0.33, 0.98)) -> sns.FacetGrid:
    """
    Generate faceted line plots of training and validation MSE across epochs.

    This function visualizes training (`mse`) and validation (`val_mse`) mean squared errors 
    across different learning rates and optimizers. If an `optimizer` is provided, 
    plots are restricted to that optimizer only.

    Args:
        epoch_data (pd.DataFrame):
            Dataframe containing epoch data (`epoch`, `optimizer`, `learning_rate`, `rmse`, `val_rmse`).
        model_name (str):
            Descriptive name of the model (used in plot titles).
        optimizer (Optional[str], default = None):
            If provided, filter results to only include the specified optimizer
            (e.g., "Adam", "SGD", or "RMSprop").
        title_fontsize (int, default = 20):
            Font size of the main figure title.
        label_fontsize (int, default = 12):
            Font size for axis labels and facet titles.
        tick_fontsize (int, default = 9):
            Font size for axis tick labels.
        title_pos (Tuple[float, float], default = (0.33, 0.98)):
            (x, y) coordinates for positioning the figure’s suptitle.

    Returns:
        sns.FacetGrid:
            A Seaborn FacetGrid object containing the plotted MSE curves.
    """
    # Optionally filter to a specific optimizer
    if optimizer is not None:
        epoch_data = epoch_data[epoch_data["optimizer"] == optimizer]

    # Create a FacetGrid comparing learning rates and optimizers
    g = sns.FacetGrid(
        data = epoch_data,
        row = "optimizer" if optimizer is None else None,  # Only facet rows if not filtered
        col = "learning_rate",
        col_order = [0.01, 0.001, 0.0001],
        margin_titles = True,
        height = 3,
        aspect = 1.5,
        sharey = False,
        sharex = False,
        legend_out = True
    )

    # Plot training and validation MSE lines
    g.map_dataframe(sns.lineplot, x = "epoch", y = "rmse", color = "red", label = "Train RMSE")
    g.map_dataframe(sns.lineplot, x = "epoch", y = "val_rmse", color = "green", label = "Validation RMSE")

    # Add a reference line at the minimum validation MSE
    min_val_rmse = epoch_data["val_rmse"].min()
    g.refline(y = min_val_rmse, label = f"Best Validation Minimum ({min_val_rmse:.4f})", color = "0.35")

    # Add a global title, position, and size adjustable via parameters
    optimizer_label = optimizer if optimizer else "All Optimizers"
    g.figure.suptitle(
        f"Train vs. Validation RMSE for {model_name} ({optimizer_label})",
        x = title_pos[0],
        y = title_pos[1],
        fontsize = title_fontsize,
        horizontalalignment = "center"
    )

    # Configure axes and legend
    g.set_titles(
        row_template = "{row_name}",
        col_template = "Initial Learning Rate: {col_name}",
        size = label_fontsize
    )
    g.set_xlabels("Epoch", fontsize = label_fontsize)
    g.set_ylabels("RMSE", fontsize = label_fontsize)
    g.tick_params(labelsize = tick_fontsize)
    g.add_legend(bbox_to_anchor = (0.5, 0.0), fontsize = label_fontsize, ncols = 3, frameon = False)

    # Ensure x-axis tick labels are integers (epochs)
    for ax in g.axes.flatten():
        ax.xaxis.set_major_locator(MaxNLocator(integer = True))

    # Adjust subplot layout padding
    g.tight_layout(pad = 1.8)

    return g

def compare_model_curves(model_data: List[pd.DataFrame], model_names: List[str],
                         metric: str = 'rmse', epochs: int = 12, figsize: tuple = (12, 6)) -> plt.Figure:
    """
    Plot training and validation curves for multiple models on the same axes.

    Args:
        model_data (List[pd.DataFrame]): List of DataFrames containing training history.
            Each DataFrame must include columns 'epoch', metric, and 'val_' + metric.
        model_names (List[str]): List of model names corresponding to `model_data`.
        metric (str, optional): Metric to plot (default is 'rmse'). Validation column
            should be 'val_<metric>'.
        epochs (int, optional): Maximum number of epochs to display on x-axis.
        figsize (tuple, optional): Figure size for the plot.

    Returns:
        matplotlib.figure.Figure: The figure object containing the plot.
    """

    # Create the figure and axes
    fig, ax = plt.subplots(figsize = figsize)

    # Define colors
    colors = sns.color_palette('bright', n_colors = len(model_names))

    # Plot training and validation curves for each model
    for i, (name, data) in enumerate(zip(model_names, model_data)):
        # Training curve
        sns.lineplot(
            data = data,
            x = 'epoch',
            y = metric,
            color = colors[i],
            label = f'{name} Train {metric.upper()}',
            ax = ax,
            alpha = 0.7
        )
        # Validation curve
        sns.lineplot(
            data = data,
            x = 'epoch',
            y = f'val_{metric}',
            color = colors[i],
            label = f'{name} Validation {metric.upper()}',
            ax = ax,
            linestyle = '--',
            alpha = 0.7
        )

    # Set axis limits, labels, and title
    ax.set_xlim([0, epochs])
    ax.set_ylabel(metric.upper(), fontsize = 14)
    ax.set_xlabel('Epoch', fontsize = 14)
    ax.tick_params(axis = 'both', labelsize = 12)
    ax.legend(fontsize = 12)
    ax.set_title(f'Training and Validation Curves Across Model Types', fontsize = 18)
    
    return fig

def prediction_and_residuals_plot(y: np.ndarray, y_pred: np.ndarray,
                                  model_title: str, figsize: Optional[Tuple[int]] = (12, 6)) -> plt.Figure:
    """
    Generate side-by-side plots showing model predictions vs. actual values
    and residuals for a regression model.

    Args:
        y (np.ndarray): Array of true target values.
        y_pred (np.ndarray): Array of predicted values.
        model_title (str): Descriptive title for the model used in plot titles.
        figsize (tuple, optional): Figure size for the two subplots. Default is (12, 6).

    Returns:
        plt.Figure: Matplotlib Figure object containing the two plots.
    """
    # Create two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize = figsize)

    # Extract predictions for the selected model
    y = np.array(y)

    # Left plot: Actual vs. Predicted values
    sns.scatterplot(
        x = y,
        y = y_pred,
        alpha = 0.6,
        ax = axes[0],
        label = 'Model Predictions'
    )
    axes[0].axline([0, 0], [1, 1], color = 'red', linestyle = '--', label = 'Perfect Prediction Line')

    axes[0].set_title(f'Actual vs. Predicted Values for {model_title}', fontsize = 14)
    axes[0].set_xlabel('Actual Values', fontsize = 12)
    axes[0].set_ylabel('Predicted Values', fontsize = 12)
    axes[0].legend(fontsize = 11)

    # Set axis limits dynamically based on min/max values
    min_bound = min(y_pred.min(), y.min())
    max_bound = max(y_pred.max(), y.max())
    axes[0].set_xlim([min_bound - 0.1, max_bound + 0.1])
    axes[0].set_ylim([min_bound - 0.1, max_bound + 0.1])
    axes[0].tick_params(axis = 'both', labelsize = 10)

    # Right plot: Residuals
    resid = y_pred - y # Zero-centered residuals
    sns.scatterplot(
        x = y,
        y = resid,
        alpha = 0.6,
        ax = axes[1],
        label = 'Residuals'
    )
    axes[1].axhline(y = 0, color = 'red', linestyle = '--', label = 'Zero Residual Line')

    axes[1].set_title(f'Residuals for {model_title}', fontsize = 14)
    axes[1].set_xlabel('Actual Values', fontsize = 12)
    axes[1].set_ylabel('Standardized Residuals', fontsize = 12)
    axes[1].legend(fontsize = 11)
 
    # Set y-axis symmetric around 0 for better visualization
    y_bound = max(np.abs(resid.min()), np.abs(resid.max()))
    axes[1].set_xlim([min_bound - 0.1, max_bound + 0.1])
    axes[1].set_ylim([-y_bound - 0.1, y_bound + 0.1])
    axes[1].tick_params(axis = 'both', labelsize = 10)

    # Adjust layout to prevent overlap
    plt.tight_layout(pad = 2)

    return fig