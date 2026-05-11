from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import norm, kruskal
from scipy.stats.contingency import association

def calculate_data_proportion(data: pd.DataFrame, slice_argument: pd.Series) -> float:
    """
    Calculate the proportion of rows in a DataFrame that satisfy a given condition.

    This function takes a boolean mask (slice_argument) corresponding to the input DataFrame
    and computes what fraction of the total rows meet that condition.

    Args:
        data (pd.DataFrame):
            The full dataset from which the subset proportion is being calculated.
        slice_argument (pd.Series):
            A boolean Series of the same length as `data` where `True` indicates
            rows that meet the desired condition.

    Returns:
        float:
            The proportion of rows in `data` where `slice_argument` is True,
            expressed as a value between 0 and 1.
    """

    # Compute the ratio of the subset (rows satisfying the condition)
    # to the total number of rows in the dataset.
    return data[slice_argument].shape[0] / data.shape[0]

def display_quantiles(data: pd.DataFrame, column: str,
                      quantiles: Union[List[float], float], print_max: bool = True) -> None:
    """
    Display specified quantiles (percentiles) and optionally the maximum value for a column.

    This function prints out the values of given quantiles for a specified numeric column
    in a DataFrame. It optionally displays the column's maximum value for reference.

    Args:
        data (pd.DataFrame):
            The DataFrame containing the column to analyze.
        column (str):
            The name of the numeric column for which to display quantile information.
        quantiles (Union[List[float], float]):
            A single quantile (e.g., 0.5 for the median) or a list of quantiles (e.g., [0.25, 0.5, 0.75]).
        print_max (bool, optional):
            Whether to print the maximum value of the column. Defaults to True.
    """

    # Ensure quantiles is always a list for consistent iteration
    quantiles = quantiles if isinstance(quantiles, list) else [quantiles]

    # Loop through the list of quantiles and print their corresponding values
    for quantile in quantiles:
        quantile_precision = len(str(quantile)) - 4
        print(f'{quantile * 100:.{quantile_precision}f} percentile of {column}: {data[column].quantile(quantile)}')

    # Optionally print the maximum value for additional reference
    if print_max:
        print(f'Max of {column}: {data[column].max()}')

    return

def compute_pairwise_associations(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pairwise association metrics (Cramer's V and Cohen's Omega) 
    between all categorical columns in a DataFrame.

    Args:
        dataframe (pd.DataFrame): 
            Input DataFrame containing categorical variables.

    Returns:
        pd.DataFrame:
            A DataFrame with one row per unique column pair and four columns:
            - 'column1': The first variable in the pair.
            - 'column2': The second variable in the pair.
            - 'cramers_v': Cramer's V statistic for association strength.
            - 'cohens_omega': Cohen's Omega statistic derived from Cramer's V.
    """
    
    associations = []
    
    # Loop through all unique pairs of columns
    for i, col1 in enumerate(dataframe):
        for col2 in dataframe[i + 1:]:
            # Create contingency table for the two categorical variables
            crosstab = pd.crosstab(dataframe[col1], dataframe[col2])
            
            # Compute Cramer's V
            # Use correction only if the table is exactly 2x2
            cramers_v = association(crosstab, correction = crosstab.shape == (2, 2))
            
            # Compute Cohen's Omega using Cramer's V
            cohens_omega = cramers_v * np.sqrt(np.min(crosstab.shape) - 1)
            
            # Store results for this column pair
            associations.append((col1, col2, cramers_v, cohens_omega))

    # Convert results into a DataFrame
    associations_df = pd.DataFrame(
        associations,
        columns = ['column1', 'column2', 'cramers_v', 'cohens_omega']
    )

    return associations_df

def fisher_z_test_correlations(datasets: List[pd.DataFrame], method: str = "pearson",
                               reference_index: int = 0, return_significant_only: bool = True,
                               alpha: float = 0.05, dataset_names: Optional[List[str]] = None,
                               corr_magnitude_thresh: float = 0.5) -> Dict[str, Dict[str, pd.DataFrame] | pd.DataFrame]:
    """
    Compare correlation matrices of multiple datasets against a reference
    using Fisher's Z-transform to test for significant differences.

    Args:
        datasets (List[pd.DataFrame]):
            List of DataFrames with identical columns.
        method (str, optional):
            Correlation method: 'pearson' or 'spearman'. Defaults to 'pearson'.
        reference_index (int, optional):
            Index of the reference dataset for comparison. Defaults to 0.
        return_significant_only (bool, optional):
            If True, only return correlations with significant differences (p < alpha).
            Non-significant entries are replaced with NaN. Defaults to True.
        alpha (float, optional):
            Significance threshold for filtering when return_significant_only=True.
            Defaults to 0.05.
        dataset_names (List[str], optional):
            Custom names for datasets. Must match length of datasets list.
            Defaults to auto-generated names ("dataset_0", "dataset_1", etc.).
        corr_magnitude_thresh (float, optional):
            Minimum absolute correlation magnitude (in either dataset) required
            for inclusion in the summary. Defaults to 0.5.

    Returns:
        Dict[str, Dict[str, pd.DataFrame] | pd.DataFrame]:
            Dictionary containing:
                - "correlations": Correlation matrices for all datasets.
                - "z_scores": Fisher Z-test statistic matrices (vs reference).
                - "p_values": Two-tailed p-value matrices (vs reference).
                - "summary": Combined summary DataFrame of all variable pairs and results.
    """
    # Validate dataset names
    if dataset_names is None:
        dataset_names = [f"dataset_{i}" for i in range(len(datasets))]
    elif len(dataset_names) != len(datasets):
        raise ValueError("Length of dataset_names must match number of datasets.")

    # Compute correlation matrices and sample sizes
    correlations = {
        dataset_names[i]: df.corr(method = method, numeric_only = True) for i, df in enumerate(datasets)
    }
    ns = [len(df) for df in datasets] # number of samples for each dataset

    # Fisher's Z-transform helper
    def fisher_z(r: np.ndarray) -> np.ndarray:
        # Clip to avoid log 0 and divide by 0 errors
        r = np.clip(r, -0.999999, 0.999999)
        return 0.5 * np.log((1 + r) / (1 - r))

    ref_name = dataset_names[reference_index]
    ref_corr = correlations[ref_name]
    ref_n = ns[reference_index]
    ref_z = fisher_z(ref_corr)

    z_scores = {}
    p_values = {}
    summary_records = []

    for i, df in enumerate(datasets):
        # Only compare non-reference datasets to the reference dataset
        if i == reference_index:
            continue

        # Extract names, correlations, number of samples, and Fisher's Z
        name = dataset_names[i]
        r_corr = correlations[name]
        n_i = ns[i]
        r_z = fisher_z(r_corr)

        # Standard error for independent samples
        se = np.sqrt(1 / (ref_n - 3) + 1 / (n_i - 3))

        # Z-statistic and p-value
        z_diff = (ref_z - r_z) / se
        p_matrix = 2 * (1 - norm.cdf(np.abs(z_diff)))

        # Convert to DataFrame
        z_df = pd.DataFrame(z_diff, index = ref_corr.index, columns = ref_corr.columns)
        p_df = pd.DataFrame(p_matrix, index = ref_corr.index, columns = ref_corr.columns)

        # Filter if requested
        if return_significant_only:
            z_df = z_df.where(p_df < alpha)
            p_df = p_df.where(p_df < alpha)

        # Record Z-scores and p-values
        comparison_name = f"{name}_vs_{ref_name}"
        z_scores[comparison_name] = z_df
        p_values[comparison_name] = p_df

        # Build summary records pairwise
        for col1 in ref_corr.columns:
            for col2 in ref_corr.columns:
                if col1 >= col2:
                    continue

                r_ref = ref_corr.loc[col1, col2]
                r_cmp = r_corr.loc[col1, col2]
                z_val = z_df.loc[col1, col2]
                p_val = p_df.loc[col1, col2]

                if np.isnan(p_val):
                    continue  # skip NaNs (filtered-out cases)

                # Only include if either correlation exceeds magnitude threshold
                if (abs(r_ref) < corr_magnitude_thresh) and (abs(r_cmp) < corr_magnitude_thresh):
                    continue

                summary_records.append({
                    "variable_1": col1,
                    "variable_2": col2,
                    "reference_dataset": ref_name,
                    "comparison_dataset": name,
                    "r_reference": r_ref,
                    "r_comparison": r_cmp,
                    "z_score": z_val,
                    "p_value": p_val,
                    "significant": p_val < alpha
                })

    # Convert summary records to a dataframe
    summary_df = pd.DataFrame(summary_records)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["p_value", "comparison_dataset"]).reset_index(drop = True)

    return {
        "correlations": correlations,
        "z_scores": z_scores,
        "p_values": p_values,
        "summary": summary_df
    }

def compute_kruskal_wallis(dataframe: pd.DataFrame, continuous_var: str,
                           categorical_vars: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Compute Kruskal-Wallis tests for a single continuous variable
    across multiple categorical variables in a DataFrame.

    Args:
        dataframe (pd.DataFrame): Input DataFrame containing continuous and categorical variables.
        continuous_var (str): Name of the continuous variable.
        categorical_vars (List[str], optional): List of categorical variables to test. 
            If None, all object or category dtype columns are used.

    Returns:
        pd.DataFrame: DataFrame with one row per categorical variable containing:
            - 'categorical_var': Name of the categorical variable
            - 'H_statistic': Kruskal-Wallis H-statistic
            - 'p_value': p-value from the test
            - 'n_groups': Number of groups
            - 'n_total': Total number of observations used
    """
    # Select categorical variables if not passed in
    if categorical_vars is None:
        categorical_vars = dataframe.select_dtypes(include = ['object', 'category']).columns.tolist()
    
    results = []

    for cat_var in categorical_vars:
        # Extract values of continuous variable grouped into arrays by category group
        groups = [group[continuous_var].dropna().values 
                  for name, group in dataframe.groupby(cat_var)]

        # If there are not at least 2 groups, category is univalent
        if len(groups) < 2:
            continue

        # Calculate Kruskal-Wallis H-stat and p-value
        H_stat, p_val = kruskal(*groups)
        
        results.append({
            'categorical_var': cat_var,
            'H_statistic': H_stat,
            'p_value': p_val,
            'n_groups': len(groups),
            'n_total': sum(len(g) for g in groups)
        })
    
    return pd.DataFrame(results)

def calculate_VIF(dataframe: pd.DataFrame, columns: Optional[List[str]] = None,
                  log1p_columns: Optional[List[str]] = None, ridge: float = 1e-8,
                  verbose: bool = False) -> pd.Series:
    """
    Calculate Variance Inflation Factors (VIF) for numeric columns in a DataFrame.

    VIF is calculated as the diagonal of the inverse of the correlation matrix:
        VIF_j = diag(inv(corr_matrix))

    This version handles singular matrices by adding a small ridge term.

    Args:
        dataframe (pd.DataFrame): Input DataFrame.
        columns (List[str], optional): Subset of columns to compute VIF for.
            Defaults to all numeric columns.
        log1p_columns (List[str], optional): List of columns to apply np.log1p transformation
            before computing correlation. Defaults to None.
        ridge (float, optional): Small value to add to diagonal to handle singular matrices. Defaults to 1e-8.
        verbose (bool, optional): If True, print warnings for high VIF (>10). Defaults to False.

    Returns:
        pd.Series: VIF values indexed by column names, series name "VIF".
    """
    # Select numeric columns
    numeric_cols = dataframe.select_dtypes(include = np.number).columns.tolist()
    
    if columns is not None:
        numeric_cols = [col for col in columns if col in numeric_cols]

    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns available for VIF calculation.")

    # Copy and transform log1p columns if specified
    df_vif = dataframe[numeric_cols].copy()
    if log1p_columns:
        for col in log1p_columns:
            if col in df_vif.columns:
                df_vif[col] = np.log1p(df_vif[col])
    
    # Compute correlation matrix
    corr_matrix = df_vif.corr(numeric_only = True)
    
    # Regularize diagonal to handle singular matrices
    corr_matrix += np.eye(len(corr_matrix)) * ridge

    # Invert correlation matrix
    inv_corr_matrix = np.linalg.inv(corr_matrix.values)
    
    # VIF is the diagonal of the inverse correlation matrix
    vif_values = pd.Series(np.diag(inv_corr_matrix), index = corr_matrix.columns, name = "VIF")
    
    # Verbose warnings for high VIF
    if verbose:
        high_vif = vif_values[vif_values > 10]
        if not high_vif.empty:
            print("Warning: High multicollinearity detected. Variables with VIF > 10:")
            for col, val in high_vif.items():
                print(f"  - {col}: {val:.2f}")
    
    return vif_values

def calculate_multi_VIF(dataframes: List[pd.DataFrame],
                        dataset_names: Optional[List[str]] = None,
                        columns: Optional[List[str]] = None,
                        log1p_columns: Optional[List[str]] = None,
                        ridge: float = 1e-8, verbose: bool = False) -> pd.DataFrame:
    """
    Calculate VIFs for multiple datasets and combine results into a single DataFrame.

    Args:
        dataframes (List[pd.DataFrame]): List of DataFrames to compute VIF for.
        dataset_names (List[str], optional): Names for datasets. Defaults to "Dataset 1", "Dataset 2", etc.
        columns (List[str], optional): Subset of columns to compute VIF for. Defaults to all numeric columns.
        log1p_columns (List[str], optional): Columns to apply np.log1p transformation. Defaults to None.
        ridge (float, optional): Small value to add to diagonal to handle singular matrices. Defaults to 1e-8.
        verbose (bool, optional): If True, print warnings for high VIF (>10). Defaults to False.

    Returns:
        pd.DataFrame: DataFrame with columns = dataset names, rows = variable names, values = VIFs.
    """
    if dataset_names is None:
        dataset_names = [f"Dataset {i + 1}" for i in range(len(dataframes))]

    if len(dataset_names) != len(dataframes):
        raise ValueError("Length of dataset_names must match number of dataframes.")

    # Compute VIF for each dataset and collect results
    vif_dict = {}
    for name, df in zip(dataset_names, dataframes):
        vif_series = calculate_VIF(
            dataframe = df,
            columns = columns,
            log1p_columns = log1p_columns,
            ridge = ridge,
            verbose = verbose
        )
        vif_dict[name] = vif_series

    # Combine into a single DataFrame
    vif_df = pd.DataFrame(vif_dict)

    return vif_df