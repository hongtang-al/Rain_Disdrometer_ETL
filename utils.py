import io
import json
import boto3
import pandas as pd
import numpy as np


def df_from_s3(key, bucket, **kwargs):
    """read csv from S3 as pandas df
    Arguments:
        key - key of file on S3
        bucket - bucket of file on S3
        **kwargs - additional keyword arguments to pass pd.read_ methods
    Returns:
        df - pandas df
    """

    format = key.split('/')[-1].split('.')[-1]
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)
    body = obj["Body"]
    if format == "csv":
        csv_string = body.read().decode("utf-8")
        df = pd.read_csv(io.StringIO(csv_string), **kwargs)
    elif format == "parquet":
        bytes_obj = body.read()
        df = pd.read_parquet(io.BytesIO(bytes_obj), **kwargs)
    else:
        raise Exception(f"format '{format}' not recognized, expected csv or parquet file.")
    return df


def df_to_s3(df, key, bucket, verbose=True, format="csv"):
    if format == "csv":
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
    elif format == "parquet":
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
    else:
        raise Exception(f"format '{format}' not recognized")
    # write stream to S3
    s3 = boto3.client("s3")
    s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
    if verbose:
        print(f"Uploaded file to s3://{bucket}/{key}")


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def perform_data_validation(
    df: pd.DataFrame,
    feature_boundaries: dict,
    check_for_nas=True,
    check_for_ranges=True,
):
    """
    perform data validation on sagemaker endpoint incoming data: check for null values and data ranges
    Arguments:
        df - dataframe of input data to validate
        feature_boundaries - dictionary with data ranges of the format: {col_name:[min,max],...}
        check_for_na - if False, data will not be checked for NAs during validation
        check for ranges - if False, data will not be checked for approriate range during validation
    Returns:
        df - pandas df with errors column appended, describing data validation errors
    """
    # iterate through each row looking for errors
    all_errors = pd.Series([np.nan] * len(df))
    for i, row in df.iterrows():
        row_errors = []
        if check_for_nas:
            # check for missing data
            for col in df.columns:
                if pd.isnull(row[col]):
                    row_errors.append(
                        {
                            "error_type": "missing_data",
                            "error_message": f"missing data in column: {col}",
                        }
                    )
        if check_for_ranges:
            # check for out of range errors
            unfamiliar_columns = [
                c
                for c in df.columns
                if c not in feature_boundaries.keys() and c != "time"
            ]
            if unfamiliar_columns:
                print(
                    f"WARNING: The following columns were found in the input data but do not have range checks assigned:{unfamiliar_columns}"
                )
            for col in intersection(df.columns, feature_boundaries.keys()):
                lower_boundary = feature_boundaries[col][0]
                upper_boundary = feature_boundaries[col][1]
                if row[col] < lower_boundary or row[col] > upper_boundary:
                    row_errors.append(
                        {
                            "error_type": "out_of_range",
                            "error_message": f"{col} value ({row[col]}) is out of bounds for acceptable input range for {col}: [{lower_boundary}, {upper_boundary}]",
                        }
                    )

        # store errors in df
        all_errors.loc[i] = json.dumps(row_errors) if row_errors else np.nan
    df["errors"] = all_errors

    return df


def check_data_shape(df: pd.DataFrame, expected_cols: list, table_name: str):
    """
    ensure all input columns expected from backend are present
    """
    actual_columns = df.columns
    extra_columns = [c for c in actual_columns if c not in expected_cols]
    missing_columns = [c for c in expected_cols if c not in actual_columns]
    if extra_columns:
        print(
            f"Warning: endpoint received unnecessary extra columns from the {table_name} table:{extra_columns}"
        )
    if missing_columns:
        raise ValueError(
            f"This endpoint did not recieve expected columns from the {table_name} table:{missing_columns}"
        )


# Model serving


def npy_loads(data):
    """
    Deserializes npy-formatted bytes into a numpy array
    """
    stream = io.BytesIO(data)
    return np.load(stream, allow_pickle=True)


def npy_dumps(data):
    """
    Serialized a numpy array into a stream of npy-formatted bytes.
    """
    buffer = io.BytesIO()
    np.save(buffer, data)
    return buffer.getvalue()


def unpack_dataframes(
    input_data,
    metadata_col_title="feature_name",
    metadata_col_key="feature_metadata",
):
    """
    Unpack data frames sent from backend to endpoint according to interface spec defined by backend
     see https://docs.google.com/document/d/1wI5GEPUzGJPOXufGfLunO-G6k-Tx3I2NbtNJdXk1X0Q/edit
    """
    np_data = npy_loads(input_data)
    metadata = json.loads(np_data[0])
    dfs_as_numpy = np_data[1:]
    dfs_metadata = metadata["frame_metadata"]

    all_dfs = {}
    for df_metadata, df_as_numpy in zip(dfs_metadata, dfs_as_numpy):
        table_name = df_metadata["table_name"]
        features = [
            x.get(metadata_col_title) for x in df_metadata.get(metadata_col_key)
        ]
        df = pd.DataFrame(df_as_numpy, columns=features)
        all_dfs[table_name] = df
    return all_dfs


def reconstruct_dataframe(input_data):
    """
    reconstruct df from endpoint response according to interface spec defined by backend
     see https://docs.google.com/document/d/1wI5GEPUzGJPOXufGfLunO-G6k-Tx3I2NbtNJdXk1X0Q/edit
    """
    np_data = npy_loads(input_data)
    metadata = json.loads(np_data[0])
    features = [x.get("result_name") for x in metadata.get("results_metadata")]
    return pd.DataFrame(np_data[1], columns=features)