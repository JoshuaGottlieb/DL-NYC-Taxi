from typing import List
from tensorflow_metadata.proto.v0.schema_pb2 import Schema
import tensorflow_data_validation as tfdv

def set_drift_comparators(schema: Schema, categorical_columns: List[str],
                          numeric_columns: List[str], infinity_norm_thresh: float = 0.05,
                          jensen_shannon_thresh: float = 0.05) -> Schema:
    """
    Configure drift detection thresholds for categorical and numeric features
    in a TensorFlow Data Validation (TFDV) schema.

    This function creates a copy of an existing schema and sets the thresholds
    used by drift comparators:
      - For categorical features: Infinity Norm threshold
      - For numeric features: Jensen-Shannon divergence threshold

    Args:
        schema (Schema):
            Original TFDV Schema object.
        categorical_columns (List[str]):
            List of categorical feature names to configure with infinity norm drift comparator.
        numeric_columns (List[str]):
            List of numeric feature names to configure with Jensen-Shannon divergence drift comparator.
        infinity_norm_thresh (float, optional):
            Threshold for detecting drift in categorical features. Defaults to 0.05.
        jensen_shannon_thresh (float, optional):
            Threshold for detecting drift in numeric features. Defaults to 0.05.

    Returns:
        Schema:
            A new Schema object with drift comparator thresholds set for specified features.

    Notes:
        - The function does not modify the original schema; it returns a copied version.
        - The thresholds represent the sensitivity for detecting drift: lower values make the detector more sensitive.
    """

    # Create a deep copy of the original schema to avoid modifying it in place
    schema_with_drift_config = Schema()
    schema_with_drift_config.CopyFrom(schema)

    # Set Infinity Norm threshold for categorical features
    for column in categorical_columns:
        feature = tfdv.get_feature(schema_with_drift_config, column)
        feature.drift_comparator.infinity_norm.threshold = infinity_norm_thresh

    # Set Jensen-Shannon divergence threshold for numeric features
    for column in numeric_columns:
        feature = tfdv.get_feature(schema_with_drift_config, column)
        feature.drift_comparator.jensen_shannon_divergence.threshold = jensen_shannon_thresh

    return schema_with_drift_config