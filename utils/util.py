import boto3
import pandas as pd
from io import BytesIO

import shutil
import os


def clear_folder(folder_path):
        # Check if the folder exists
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            # Delete the folder and all its contents
            shutil.rmtree(folder_path)
            print(f"Folder '{folder_path}' and all its contents have been deleted.")
        else:
            print("The folder does not exist or is not a directory.")


def partition_large_dataset(file_path, output_folder_path):
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='latin1')  # trying with 'latin1' encoding
        df['Index'] = range(1, len(df) + 1)
        df = df[['Index', 'text', 'link', 'headline', 'short_description', 'date']]

        # Calculate the partition size
        partition_size = len(df) // 10  # Integer division to get the size of each partition

        # Create a new folder for partitioned files
        os.makedirs(output_folder_path, exist_ok=True)

        # Partition and save
        for i in range(10):
            start = i * partition_size
            end = (i + 1) * partition_size if i < 9 else len(df)  # Adjust the end for the last partition
            partition = df.iloc[start:end]
            partition_file_path = os.path.join(output_folder_path, f'NYTimes_part_{i+1}.csv')
            partition.to_csv(partition_file_path, index=False)

        print("Partitioning completed.")


def upload_files_to_s3(bucket_name, folder_path, aws_access_key_id, aws_secret_access_key):
        """
        Upload all files from a local folder to an Amazon S3 bucket

        Parameters:
        - bucket_name (str): the name of the S3 bucket
        - folder_path (str): the local path of the folder containing files to be uploaded
        - aws_access_key_id (str): AWS access key ID with S3 written permissions
        - aws_secret_access_key (str): AWS secret access key corresponding to the provided access key

        Returns:
        None
        """

        # Create an S3 client
        s3 = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )

        # List files in the folder
        files = os.listdir(folder_path)

        # Upload each file to the bucket
        for file in files:
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                s3.upload_file(file_path, bucket_name, file)
                print(f"Uploaded {file} to S3 bucket {bucket_name}")


def read_from_partitions(bucket_name, base_file_path, target_indices, partition_size, aws_access_key_id, aws_secret_access_key):
        """
        Read specific rows from partitioned CSV files stored in an Amazon S3 bucket and combine them into a DataFrame

        Parameters:
        - bucket_name (str): the name of the S3 bucket
        - base_file_path (str): the base path of partitioned CSV files without the partition suffix
        - target_indices (list): a list of row indices to retrieve from the partitions
        - partition_size (int): the size of each partition
        - aws_access_key_id (str): AWS access key ID with S3 written permissions
        - aws_secret_access_key (str): AWS secret access key corresponding to the provided access key

        Returns:
        pandas.DataFrame: a DataFrame containing the rows specified by the target indices
        """

        # Create S3 client
        s3 = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )

        # Dictionary to hold dataframes for each partition
        partition_dfs = {}

        # List to hold the rows corresponding to target indices
        rows = []

        for index in target_indices:
            # Calculate the partition suffix
            partition_number = index // partition_size + 1

            if partition_number not in partition_dfs:
                # Obtain the partition file name
                partition_file_name = f"{base_file_path}_part_{partition_number}.csv"

                # Get the object from S3
                response = s3.get_object(Bucket=bucket_name, Key=partition_file_name)
                file_content = response['Body'].read()

                # Read the file content into a pandas DataFrame
                df = pd.read_csv(BytesIO(file_content))
                partition_dfs[partition_number] = df

            else:
                df = partition_dfs[partition_number]

            # Retrieve the row by index if it exists in the DataFrame
            if index in list(df.Index):
                rows.append(df[df.Index == index])

        # Combine all rows into a single DataFrame
        return pd.concat(rows, axis=0)

def read_from_local_partitions(target_indices, partition_size):
        """
        Read specific rows from partitioned CSV files stored locally and combine them into a DataFrame

        Parameters:
        - target_indices (list): a list of row indices to retrieve from the partitions
        - partition_size (int): the size of each partition

        Returns:
        pandas.DataFrame: a DataFrame containing the rows specified by the target indices
        """
        base_file_path = "NYTimes"
        # Dictionary to hold dataframes for each partition
        partition_dfs = {}

        # List to hold the rows corresponding to target indices
        rows = []

        for index in target_indices:

            partition_number = index // partition_size + 1
            if partition_number not in partition_dfs:

                # Calculate the partition file name
                partition_file_name = f"dataset/partitioned_nyt/{base_file_path}_part_{partition_number}.csv"

                df = pd.read_csv(partition_file_name)
                partition_dfs[partition_number] = df

            else:
                df = partition_dfs[partition_number]

            # Retrieve the row by index if it exists in the DataFrame
            if index in list(df.Index):

                rows.append(df[df.Index == index])

        # Combine all rows into a single DataFrame
        return pd.concat(rows, axis=0)
