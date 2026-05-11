import re
from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

# ---- Weather Data Preprocessing Functions ----

def preprocess_weather_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess raw weather data for use in trip-level modeling.

    This function standardizes column names, converts date values to datetime,
    handles missing values through linear interpolation, and rounds all 
    numerical features to two decimal places for consistency.

    Args:
        data (pd.DataFrame):
            Raw weather data containing measurements such as temperature,
            precipitation, wind, and air pressure. The input is assumed to
            follow the original column order of the weather source file.

    Returns:
        pd.DataFrame:
            A cleaned and preprocessed weather dataset with:
              - Standardized column names.
              - Datetime-formatted 'date' column.
              - Missing values imputed via linear interpolation.
              - Numeric features rounded to two decimal places.
    """

    # Step 1: Assign standardized column names for clarity and consistency
    data.columns = [
        'date', 'average_temperature', 'min_temperature', 'max_temperature',
        'precipitation', 'snow', 'wind_direction', 'wind_speed', 'wind_peak_gust',
        'air_pressure', 'sun_duration'
    ]

    # Step 2: Convert the 'date' column to datetime format for time-based operations
    data.date = pd.to_datetime(data.date)

    # Step 3: Drop columns that are entirely missing and linearly interpolate remaining gaps
    # Interpolation helps maintain temporal continuity in weather trends
    data = data.dropna(axis = 1, how = 'all').interpolate(method = 'linear')

    # Step 4: Round all numeric columns to two decimal places for consistency
    data = np.round(data, 2)

    # Step 5: Return the cleaned and preprocessed DataFrame
    return data

def join_taxi_weather_data(taxi_data: pd.DataFrame, weather_data: pd.DataFrame) -> pd.DataFrame:
    """
    Merge taxi trip data with corresponding daily weather observations.

    This function aligns taxi trip records with weather data by:
      - Extracting the pickup date from each trip's timestamp.
      - Setting both datasets to use 'date' as their index.
      - Performing an inner join to retain only overlapping dates.
      - Resetting the index after merging for a clean tabular format.

    Args:
        taxi_data (pd.DataFrame):
            DataFrame containing taxi trip records. Must include a
            'tpep_pickup_datetime' column representing trip start times.

        weather_data (pd.DataFrame):
            DataFrame containing preprocessed daily weather information
            with a 'date' column as produced by `preprocess_weather_data()`.

    Returns:
        pd.DataFrame:
            Merged DataFrame where each taxi trip record is augmented with
            weather features corresponding to its pickup date.
            The resulting data includes only dates present in both datasets.
    """

    # Step 1: Extract pickup date (without time) and convert to datetime
    taxi_data['date'] = pd.to_datetime(taxi_data.tpep_pickup_datetime.dt.date)

    # Step 2: Set 'date' as the index for both datasets to prepare for merging
    taxi_data = taxi_data.set_index('date')
    weather_data = weather_data.set_index('date')

    # Step 3: Perform an inner join on 'date' to combine only overlapping days
    merged_data = pd.merge(
        taxi_data,
        weather_data,
        on = 'date',
        how = 'inner'
    ).reset_index(drop = True)

    # Step 4: Return the merged DataFrame containing both trip and weather data
    return merged_data

# ---- Taxi Data Preprocessing Functions ----

def normalize_column_names(data: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names in a DataFrame to a consistent snake_case format.

    This function standardizes column names by:
      - Replacing spaces, dashes, and other non-alphanumeric separators with underscores.
      - Converting camelCase or PascalCase to snake_case.
      - Handling uppercase acronyms at the start of a column name (e.g., 'IDNumber' -> 'id_number').
      - Ensuring all column names are lowercase.

    Args:
        data (pd.DataFrame):
            Input DataFrame whose column names will be normalized.

    Returns:
        pd.DataFrame:
            A new DataFrame with column names converted to snake_case format.
    """

    normalized_columns = []
    for col in data.columns:
        # Replace any sequence of non-alphanumeric characters with underscores
        col = re.sub(r'[^0-9a-zA-Z]+', '_', col)

        # Handle standard camelCase or PascalCase (e.g., 'VendorID' -> 'vendor_ID')
        if not col[:3].isupper():
            col = re.sub(r'([a-z]+)([A-Z])', r'\1_\2', col)
        else:
            # Handle leading uppercase sequences of length 2 and middle lowercase sequences
            # (e.g., 'PULocationID -> PU_Location_ID')
            col = re.sub(r'^([A-Z]{2})([A-Z][a-z]+)([A-Z])', r'\1_\2_\3', col)

        # Convert to lowercase and remove trailing underscores
        col = col.lower().strip('_')

        normalized_columns.append(col)

    data.columns = normalized_columns
    
    return data
    
def feature_extraction(data: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature extraction on a taxi trip dataset by engineering new time-based 
    and speed-related features.

    This function creates derived columns for trip duration and average speed, 
    as well as extracting the pickup hour from datetime columns. It assumes that 
    the input DataFrame contains the columns:
      - 'tpep_pickup_datetime'
      - 'tpep_dropoff_datetime'
      - 'trip_distance'

    Args:
        data (pd.DataFrame):
            Input DataFrame containing trip-level data with pickup/dropoff timestamps 
            and trip distance.

    Returns:
        pd.DataFrame:
            The modified DataFrame including the new columns:
              - 'pickup_hour': Hour of the day the trip started (0–23)
              - 'trip_time': Duration of the trip in hours
              - 'avg_speed': Average trip speed in miles per hour (mph)
            and with datetime columns removed.
    """

    # Calculate total trip time in hours and insert as the 4th column (index 3)
    # Using np.timedelta64 ensures consistent unit conversion to hours.
    data.insert(
        3,
        'trip_time',
        (data.tpep_dropoff_datetime - data.tpep_pickup_datetime) / np.timedelta64(1, 'h')
    )

    # Compute average speed in miles per hour and insert as the 7th column (index 6)
    # Avoid division by zero — infinite or NaN speeds can be handled later if needed.
    data.insert(6, 'avg_speed', data.trip_distance / data.trip_time)

    # Extract hour of pickup from datetime, capturing daily temporal patterns
    data.insert(1, 'pickup_hour', data.tpep_pickup_datetime.dt.hour)

    # Drop original datetime columns since derived features capture their useful information
    data = data.drop(['tpep_pickup_datetime', 'tpep_dropoff_datetime'], axis = 1)

    return data
    
def map_via_data_dictionary(data: pd.DataFrame) -> pd.DataFrame:
    """
    Map coded categorical values in NYC Yellow Taxi trip data to their descriptive labels
    using the official NYC Taxi and Limousine Commission (TLC) data dictionary.

    The mappings are based on the TLC Yellow Taxi Trip Records Data Dictionary:
    https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf

    Specifically, this function decodes:
      - `vendor_id`: Identifies the technology vendor that provided the trip record.
      - `ratecode_id`: Indicates the rate code used to calculate the fare.
      - `payment_type`: Describes the payment method used by the passenger.

    Args:
        data (pd.DataFrame):
            Input DataFrame containing NYC Yellow Taxi trip record columns,
            including 'vendor_id', 'ratecode_id', and 'payment_type'.

    Returns:
        pd.DataFrame:
            A DataFrame where coded identifiers have been replaced with human-readable
            category names. Columns are cast to object dtype.

    Notes:
        - Unrecognized or missing codes are mapped to NaN.
        - The function assumes the specified columns exist in the input DataFrame.
    """

    # Vendor Mapping - Maps integer vendor IDs to their respective company names
    vendor_map = defaultdict(lambda: np.nan)
    vendor_map.update({
        1: 'Creative Mobile Technologies, LLC',
        2: 'Curb Mobility, LLC',
        6: 'Myle Technologies Inc',
        7: 'Helix'
    })
    data['vendor_id'] = data['vendor_id'].map(lambda x: vendor_map[x]).astype('object')

    # Rate Code Mapping - Maps rate code IDs to fare structure categories
    ratecode_map = defaultdict(lambda: np.nan)
    ratecode_map.update({
        1: 'Standard',
        2: 'JFK',
        3: 'Newark',
        4: 'Nassau/Westchester',
        5: 'Negotiated',
        6: 'Group',
        99: np.nan  # Explicitly mark invalid or unknown codes
    })
    data['ratecode_id'] = data['ratecode_id'].map(lambda x: ratecode_map[x]).astype('object')

    # Payment Type Mapping - Maps numeric payment codes to descriptive payment methods
    payment_map = defaultdict(lambda: np.nan)
    payment_map.update({
        0: 'Flex',
        1: 'Credit',
        2: 'Cash',
        3: 'No Charge',
        4: 'Dispute',
        5: np.nan,  # Unknown or invalid payment
        6: 'Voided'
    })
    data['payment_type'] = data['payment_type'].map(lambda x: payment_map[x]).astype('object')

    return data

def impute_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing or invalid values in key NYC Yellow Taxi trip dataset columns.

    This function applies a series of logic-based imputations and corrections:
      - Fills missing categorical vendor identifiers using the most frequent value.
      - Cleans and imputes `congestion_surcharge` values based on TLC fare rules.
      - Replaces invalid or missing passenger counts with reasonable defaults.
      - Infers missing rate codes using location-based heuristics.

    Args:
        data (pd.DataFrame):
            Input DataFrame containing at least the following columns:
            ['vendor_id', 'congestion_surcharge', 'passenger_count',
             'ratecode_id', 'do_location_id', 'pu_location_id'].

    Returns:
        pd.DataFrame:
            DataFrame with imputed and corrected values for vendor, surcharge,
            passenger count, and rate codes.

    Notes:
        - The logic is based on NYC Taxi and Limousine Commission (TLC) trip data standards.
        - Assumes that surcharge values should be one of {0, 0.75, 2.5}.
        - Caps `passenger_count` at 5 and replaces 0 with 1 (minimum valid count).
        - Rate codes are inferred contextually using pickup and drop-off locations.
    """

    # Vendor ID Imputation - Replace missing vendor IDs with the most common vendor (mode).
    data['vendor_id'] = data['vendor_id'].fillna(data['vendor_id'].mode()[0])

    # Congestion Surcharge Cleaning & Imputation
    # Ensure valid surcharge values are only $0, $0.75, or $2.50.
    # Any other positive values are treated as invalid and replaced with NaN.
    data['congestion_surcharge'] = data['congestion_surcharge']\
                                    .abs().map(lambda x: x if x in [0, 0.75, 2.5] else np.nan)

    # For 'Group' rate code trips with missing surcharge, set to $0.75.
    data.loc[
        (data['congestion_surcharge'].isnull()) & (data['ratecode_id'] == 'Group'),
        'congestion_surcharge'
    ] = 0.75

    # For all remaining missing surcharge values, assume the standard $2.50 rate.
    data['congestion_surcharge'] = data['congestion_surcharge'].fillna(2.5)

    # Passenger Count Imputation
    # Replace invalid zero passengers with NaN, cap excessive counts at 5.
    # Fill missing counts with 1 passenger (most common assumption for taxi trips).
    data['passenger_count'] = data['passenger_count'].replace({0: np.nan})\
                                .clip(upper = 5).fillna(1).astype(int)

    # Rate Code Inference
    # Fill missing rate codes using contextual location-based logic:
    # - Dropoff at location 132 -> JFK airport
    data.loc[
        (data['ratecode_id'].isnull()) & (data['do_location_id'] == 132),
        'ratecode_id'
    ] = 'JFK'

    # - Dropoff at location 1 -> Newark airport
    data.loc[
        (data['ratecode_id'].isnull()) & (data['do_location_id'] == 1),
        'ratecode_id'
    ] = 'Newark'

    # - Pickup or dropoff at 265 -> use the most common rate code for that zone
    data.loc[
        (data['ratecode_id'].isnull()) & ((data['do_location_id'] == 265) | (data['pu_location_id'] == 265)),
        'ratecode_id'
    ] = data.loc[
        (data['do_location_id'] == 265) | (data['pu_location_id'] == 265),
        'ratecode_id'
    ].mode()[0]

    # - Trips with group surcharge -> assign 'Group' rate code
    data.loc[
        (data['ratecode_id'].isnull()) & (data['congestion_surcharge'] == 0.75),
        'ratecode_id'
    ] = 'Group'

    # Fill remaining missing rate codes as 'Standard'
    data['ratecode_id'] = data['ratecode_id'].fillna('Standard')

    return data

def remove_domain_outliers(data: pd.DataFrame) -> pd.DataFrame:
    """
    Remove domain-specific outliers from a taxi trip dataset.

    This function filters out records that violate reasonable physical
    or operational constraints derived from transportation domain knowledge.
    It ensures that average speed, trip time, and trip distance fall
    within plausible real-world ranges.

    Args:
        data (pd.DataFrame):
            Input DataFrame containing at least the following numeric columns:
            ['avg_speed', 'trip_time', 'trip_distance'].

    Returns:
        pd.DataFrame:
            A filtered DataFrame excluding rows where:
              - avg_speed <= 0 or avg_speed > 70 mph,
              - trip_time <= 0 or trip_time > 12 hours,
              - trip_distance <= 0 miles.

    Notes:
        - The thresholds are chosen based on practical taxi operation limits:
            * Speeds above 70 mph are unrealistic for urban trips.
            * Trip times longer than 12 hours are assumed erroneous.
            * Distances <= 0 are invalid measurements.
        - This is a domain-based outlier filter, not statistical.
          It enforces physical plausibility, not probabilistic deviation.
    """

    # Keep only trips that satisfy all logical constraints:
    # 1. Average speed between (0, 70] mph
    # 2. Trip time between (0, 12] hours
    # 3. Trip distance > 0 miles
    data = data[
        ((data['avg_speed'] > 0) & (data['avg_speed'] <= 70)) &
        ((data['trip_time'] > 0) & (data['trip_time'] <= 12)) &
        (data['trip_distance'] > 0)
    ]

    return data

def impute_and_remove_manual_outliers(data: pd.DataFrame) -> pd.DataFrame:
    """
    Impute minor inconsistencies and remove manual outliers from a taxi trip dataset.

    This function performs two main cleaning steps:
      1. Imputation of known categorical or fixed-value inconsistencies
         - Ensures that `mta_tax` and `congestion_surcharge` contain only valid domain values.
         - Synchronizes `ratecode_id` and `congestion_surcharge` consistency (e.g., 'Group' -> 0.75).
      2. Removal of extreme or logically invalid values
         - Filters out extreme outliers in continuous variables such as
           `trip_time`, `trip_distance`, `fare_amount`, and payment-related columns.

    Args:
        data (pd.DataFrame):
            Input DataFrame containing at least the following columns:
            ['mta_tax', 'congestion_surcharge', 'ratecode_id', 'trip_time',
             'trip_distance', 'fare_amount', 'extra', 'tolls_amount', 'tip_amount',
             'pu_location_id', 'do_location_id'].

    Returns:
        pd.DataFrame:
            Cleaned DataFrame where invalid, extreme, or inconsistent entries have been corrected or removed.

    Notes:
        - `mta_tax` is expected to be either 0 or 0.5; other values are coerced to 0.5.
        - The relationship between `ratecode_id` and `congestion_surcharge` is enforced:
            * If congestion surcharge = 0.75 -> ratecode = 'Group'
            * If ratecode = 'Group' -> congestion surcharge = 0.75
        - Outlier thresholds are set manually:
            * trip_time > 0 and below 99.99th percentile
            * trip_distance >= 0.25 and below 99.99th percentile,
              except if pickup or dropoff is at JFK (zone 265)
            * fare_amount in (0, 1000]
            * extra >= 0
            * tolls_amount in [0, 50]
            * tip_amount in [0, 120]
        - Designed for NYC Taxi & Limousine Commission (TLC) trip record data.
    """

    # Step 1: Impute fixed or domain-constrained values

    # Ensure MTA tax values are only 0 or 0.5 (correct invalid values to 0.5)
    data['mta_tax'] = data['mta_tax'].abs().map(lambda x: x if x in [0, 0.5] else 0.5)

    # Enforce logical consistency between ratecode and congestion surcharge
    data.loc[data['congestion_surcharge'] == 0.75, 'ratecode_id'] = 'Group'
    data.loc[data['ratecode_id'] == 'Group', 'congestion_surcharge'] = 0.75

    # Step 2: Remove manually defined outliers based on domain knowledge
    data = data[
        # Trip time should be positive and below 99.99th percentile (remove abnormally long trips)
        ((data['trip_time'] > 0) & (data['trip_time'] <= data['trip_time'].quantile(0.9999))) &

        # Trip distance must be >= 0.25 miles and below 99.99th percentile,
        # unless pickup/dropoff is at JFK (zone 265)
        ((data['trip_distance'] >= 0.25) &
         ((data['trip_distance'] <= data['trip_distance'].quantile(0.9999)) |
          (data['pu_location_id'] == 265) | (data['do_location_id'] == 265))) &

        # Fare amount must be positive and below $1,000
        ((data['fare_amount'] > 0) & (data['fare_amount'] <= 1000)) &

        # Extra charges cannot be negative
        (data['extra'] >= 0) &

        # Tolls and tips within realistic urban taxi ranges
        ((data['tolls_amount'] >= 0) & (data['tolls_amount'] <= 50)) &
        ((data['tip_amount'] >= 0) & (data['tip_amount'] <= 120))
    ]

    return data

def drop_unneeded_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Drop non-essential or redundant columns from the NYC Taxi dataset.

    Args:
        data (pd.DataFrame):
            Input DataFrame expected to include the columns:
            ['airport_fee', 'store_and_fwd_flag', 'improvement_surcharge'].

    Returns:
        pd.DataFrame:
            DataFrame with the specified unnecessary columns removed.
    """

    # Drop columns that add little or no analytical value to the dataset
    data = data.drop(['airport_fee', 'store_and_fwd_flag', 'improvement_surcharge'], axis = 1)

    return data

def convert_to_categorical(data: pd.DataFrame) -> pd.DataFrame:
    """
    Converts columns in a DataFrame to categorical dtype.
    Ensures that 'pickup_hour' (0–23) and 'passenger_count' (1–5) are 
    converted to ordered categorical types.

    Args:
        df (pd.DataFrame): 
            The input DataFrame.

    Returns:
        pd.DataFrame: 
            A copy of the DataFrame with updated categorical columns.
    """
    data_converted = data.copy()

    categorical_columns = [
        'vendor_id', 'pickup_hour', 'passenger_count', 'ratecode_id', 'pu_location_id',
        'do_location_id', 'payment_type', 'mta_tax', 'congestion_surcharge'
    ]

    # Define ordered category ranges
    hour_categories = pd.CategoricalDtype(categories = list(range(24)), ordered = True)
    passenger_categories = pd.CategoricalDtype(categories = list(range(1, 6)), ordered = True)

    for col in categorical_columns:
        if col == "pickup_hour":
            data_converted[col] = data_converted[col].astype(hour_categories)
        elif col == "passenger_count":
            data_converted[col] = data_converted[col].astype(passenger_categories)
        else:
            data_converted[col] = data_converted[col].astype("category")

    return data_converted

def transform_and_select_taxi_features(data: pd.DataFrame) -> pd.DataFrame:    
    """
    Apply log transformations and feature selection for taxi trip data.

    This function prepares raw taxi trip features for modeling by:
      - Renaming key variables for consistency.
      - Applying log(1 + x) scaling to fare and trip duration variables.
      - Filtering out trips with unrealistic fare or duration values
        after log transformation.
      - Dropping redundant or non-predictive columns.

    Args:
        data (pd.DataFrame):
            DataFrame containing taxi trip-level features. Expected to include:
            [
                'trip_time', 'fare_amount', 'vendor_id', 'pickup_hour',
                'passenger_count', 'trip_distance', 'avg_speed', 'extra',
                'mta_tax', 'total_amount', 'congestion_surcharge'
            ]

    Returns:
        pd.DataFrame:
            A cleaned and feature-reduced DataFrame containing only
            relevant predictor variables with log-transformed target
            ('log_trip_time') and fare ('log_fare_amount') columns.
    """

    # Step 1: Rename columns
    data = data.rename(columns = {
        'trip_time': 'log_trip_time', 'fare_amount': 'log_fare_amount',
        'ratecode_id': 'rate_code', 'pu_location_id': 'pu_location',
        'do_location_id': 'do_location'
    })

    # Step 2: Apply log(1 + x) transformation to reduce skewness and stabilize variance
    data['log_fare_amount'] = np.log1p(data['log_fare_amount'])
    data['log_trip_time'] = np.log1p(data['log_trip_time'])
    
    # Step 3: Filter out unrealistic trip durations and fares
    # Keeps only trips with log_trip_time ≤ 0.75 and log_fare_amount in [1, 5]
    data = data[
        (data['log_trip_time'] <= 0.75) &
        ((data['log_fare_amount'] >= 1) & (data['log_fare_amount'] <= 5))
    ]

    # Step 4: Remove unused or redundant features not relevant for modeling
    columns_to_drop = [
        'vendor_id', 'pickup_hour', 'passenger_count', 'trip_distance',
        'avg_speed', 'extra', 'mta_tax', 'total_amount', 'congestion_surcharge'
    ]
    data = data.drop(columns_to_drop, axis = 1)

    # Step 5: Return the cleaned and feature-selected dataset
    return data

def preprocess_taxi_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Execute the full preprocessing pipeline for NYC Yellow Taxi trip data.

    This function performs a sequence of transformations to clean, standardize,
    and prepare raw taxi trip records for modeling. It integrates several
    domain-specific preprocessing steps — including normalization, feature
    engineering, mapping, imputation, outlier removal, and feature transformation.

    The pipeline applies the following stages in order:

      1. normalize_column_names()
         - Standardizes column names to consistent snake_case formatting.

      2. feature_extraction()
         - Engineers trip-level temporal and speed-related features such as
           `pickup_hour`, `trip_time`, and `avg_speed` from pickup/dropoff timestamps.

      3. map_via_data_dictionary()
         - Maps coded identifiers (`vendor_id`, `ratecode_id`, `payment_type`) 
           to descriptive labels using the official TLC data dictionary.

      4. impute_missing_values()
         - Fills missing or invalid categorical and numeric values with domain-consistent
           imputations (e.g., vendor, rate code, surcharge, passenger count).

      5. remove_domain_outliers()
         - Filters out records that violate realistic operational constraints 
           (e.g., extreme speeds, implausible durations, or non-positive distances).

      6. impute_and_remove_manual_outliers()
         - Applies targeted imputation and removes manually defined outliers based on
           domain thresholds for fares, distances, and related trip attributes.

      7. drop_unneeded_columns()
         - Removes non-predictive or redundant fields (e.g., `airport_fee`, `store_and_fwd_flag`).

      8. convert_to_categorical()
         - Converts discrete numerical columns (`pickup_hour`, `passenger_count`) 
           to categorical data types with defined ordering.

      9. transform_and_select_taxi_features()
         - Applies log transformations, filters unrealistic values, and 
           drops irrelevant variables to produce the final modeling-ready dataset.

    Args:
        data (pd.DataFrame):
            Raw NYC Yellow Taxi trip record DataFrame, typically sourced
            from the NYC TLC trip record dataset.

    Returns:
        pd.DataFrame:
            A fully preprocessed, feature-engineered DataFrame ready for 
            integration with external datasets (e.g., weather data) or 
            downstream modeling pipelines.

    Notes:
        - This function assumes the input DataFrame includes all standard NYC
          Yellow Taxi fields (pickup/dropoff times, locations, fare details, etc.).
        - The transformations are cumulative and ordered; changing the sequence 
          may alter the final feature set.
    """

    # Step 1: Standardize all column names to snake_case
    data = normalize_column_names(data)

    # Step 2: Engineer time-based and speed-related features
    data = feature_extraction(data)

    # Step 3: Decode coded categorical variables using the TLC data dictionary
    data = map_via_data_dictionary(data)

    # Step 4: Impute missing and invalid values using domain-informed rules
    data = impute_missing_values(data)

    # Step 5: Remove records violating physical or operational constraints
    data = remove_domain_outliers(data)

    # Step 6: Apply targeted imputations and manually remove extreme outliers
    data = impute_and_remove_manual_outliers(data)

    # Step 7: Drop redundant or non-essential fields
    data = drop_unneeded_columns(data)

    # Step 8: Convert discrete numeric fields to categorical data types
    data = convert_to_categorical(data)

    # Step 9: Apply log transformations and select final modeling features
    data = transform_and_select_taxi_features(data)

    # Step 10: Return the fully cleaned and preprocessed dataset
    return data

# ---- Final Preprocessing Functions ----

def feature_select_and_encode_eda(data: pd.DataFrame) -> pd.DataFrame:
    """
    Selects relevant taxi trip features, applies log and scaling transformations, 
    and encodes categorical variables using a mixed preprocessing pipeline.

    Used for the EDA portion of the project.

    Args:
        data (pd.DataFrame): 
            Input dataframe containing raw trip-level features, including:
            ['trip_time', 'fare_amount', 'tip_amount', 'tolls_amount',
             'ratecode_id', 'payment_type', 'pu_location_id', 'do_location_id']

    Returns:
        pd.DataFrame:
            Transformed dataframe containing:
            - Scaled and encoded predictor features.
            - Standardized target column ('log_trip_time').
    """

    # Step 1: Keep only relevant columns used for modeling
    data = data[
        [
            'trip_time', 'fare_amount', 'tip_amount', 'tolls_amount',
            'ratecode_id', 'payment_type', 'pu_location_id', 'do_location_id'
        ]
    ]

    # Step 2: Filter out outliers based on log-transformed trip time and fare amount
    # Keeps trips with reasonable durations and fares after log scaling
    data = data[
        (np.log1p(data['trip_time']) <= 0.75) &
        ((np.log1p(data['fare_amount']) >= 1) & (np.log1p(data['fare_amount']) <= 5))
    ]
    
    # Step 3: Rename columns to standardized names for downstream processing
    data.columns = [
        'log_trip_time', 'log_fare_amount', 'tip_amount', 'tolls_amount',
        'rate_code', 'payment_type', 'pu_location', 'do_location'
    ]

    # Step 4: Apply log(1 + x) transformation to continuous variables
    # Reduces right skew and helps stabilize relationships for linear modeling
    data['log_fare_amount'] = np.log1p(data['log_fare_amount'])
    data['log_trip_time'] = np.log1p(data['log_trip_time'])

    # Step 5: Split into predictors (X) and target (y)
    X = data.drop(columns = ['log_trip_time'])
    y = data['log_trip_time'].to_frame()

    # Step 6: Standardize the target variable using z-score normalization
    ssc = StandardScaler()
    y_scaled = ssc.fit_transform(y)
    y = pd.Series(
        y_scaled.flatten(), 
        index = y.index, 
        name = ssc.get_feature_names_out()[0]
    )

    # Step 7: Define preprocessing transformations for predictors
    preprocessor = ColumnTransformer(
        transformers = [
            ('ssc', StandardScaler(), ['log_fare_amount', 'tip_amount', 'tolls_amount']), # scale continuous features
            ('ohe', OneHotEncoder(), ['rate_code', 'payment_type']), # one-hot encode low-cardinality categories
            ('te', TargetEncoder(), ['pu_location', 'do_location']) # target encode high-cardinality features
        ],
        sparse_threshold = 0, # force dense output for easier downstream use
        n_jobs = -1, # parallelize transformations
        verbose_feature_names_out = False # cleaner feature names in output
    )

    # Step 8: Fit and transform predictors using the preprocessing pipeline
    X_transformed = preprocessor.fit_transform(X, y)
    X = pd.DataFrame(
        X_transformed, 
        index = X.index, 
        columns = preprocessor.get_feature_names_out()
    )

    # Step 9: Combine encoded predictors with the standardized target variable
    data_transformed = pd.concat([X, y], axis = 1)

    return data_transformed

def feature_select_and_encode(data: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature selection, scaling, and encoding for taxi trip data.

    This function prepares trip-level data for modeling by:
      - Dropping unused or redundant features.
      - Scaling continuous predictors using standardization.
      - Encoding categorical features using a combination of
        one-hot and target encoding.

    Used for the modeling portion of the project.

    Args:
        data (pd.DataFrame): 
            Input DataFrame containing trip-level features. 
            Expected columns include:
            [
                'trip_time', 'fare_amount', 'tip_amount', 'tolls_amount',
                'rate_code', 'payment_type', 'pu_location', 'do_location',
                'log_trip_time', 'min_temperature', 'max_temperature',
                'wind_speed', 'air_pressure', 'average_temperature'
            ]

    Returns:
        pd.DataFrame:
            A transformed DataFrame containing:
              - Scaled continuous features.
              - One-hot encoded categorical features with low cardinality.
              - Target-encoded categorical features with high cardinality.
    """

    # Step 1: Remove unused feature(s) not required for modeling
    data = data.drop(['average_temperature'], axis = 1)

    # Step 2: Split into predictors (X) and target (y)
    X = data.drop(columns = ['log_trip_time'])
    y = data['log_trip_time']

    # Step 3: Define preprocessing pipeline for predictor variables
    preprocessor = ColumnTransformer(
        transformers = [
            # Scale continuous variables to zero mean and unit variance
            ('ssc', StandardScaler(), [
                'log_fare_amount', 'tip_amount', 'tolls_amount',
                'min_temperature', 'max_temperature', 'wind_speed', 'air_pressure'
            ]),
            # Apply one-hot encoding to low-cardinality categorical features
            ('ohe', OneHotEncoder(), ['rate_code', 'payment_type']),
            # Apply target encoding to high-cardinality location identifiers
            ('te', TargetEncoder(), ['pu_location', 'do_location'])
        ],
        sparse_threshold = 0, # Ensure dense output for compatibility
        n_jobs = -1, # Enable parallel processing
        verbose_feature_names_out = False # Simplify feature names in output
    )

    # Step 4: Fit and transform predictor data
    X_transformed = preprocessor.fit_transform(X, y)
    X = pd.DataFrame(
        X_transformed,
        index = X.index,
        columns = preprocessor.get_feature_names_out()
    )

    # Step 6: Combine preprocessed predictors with the standardized target
    data_transformed = pd.concat([X, y], axis = 1)

    return data_transformed