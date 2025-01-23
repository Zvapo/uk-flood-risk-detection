import numpy as np
import pandas as pd
import re
import os
from colorama import Fore, Style
from typing import List, Tuple

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# from .geo import *
from data.file_mapping import *

class FileToAppend(object):
    def __init__(self, predictions: pd.Series, column_name: str):
        self.predictions = predictions
        self.column_name = column_name

    def __iter__(self):
        return iter([self.predictions, self.column_name])

class Tr(object):
    """
        A transformer object for handling .csv files, composing training, testing and validation datasets.
        If no custom data or custom data pipeline is provided, will process to use the default settings. 
    """

    def __init__(self, target_label: str, data_path: str, columns_2_exclude: list[str] = None, pipeline: Pipeline = None):
        '''
        Initialize the transformer object.
        Reads the data to a pandas DataFrame and composes the training, testing and validation datasets.
        If pipeline is not provided, a default pipeline is created.

        Parameters:
            - target_label (Required): str, the name of the target column in the dataset.
            - data_path (Required): str, the path to the dataset. File format is .csv or .xlsx.
            - columns_2_exclude (Optional): list, the columns to exclude from the dataset.
            - pipeline (Optional): sklearn.pipeline.Pipeline, the pipeline to use for data transformation.
        '''
        self.data_path = data_path
        self.data = Tr.read_to_df(self.data_path)
        self.target_label = target_label
        self.columns_2_exclude = columns_2_exclude

        if self.columns_2_exclude:
            assert all(col in self.data.columns for col in columns_2_exclude), Fore.RED + f"Columns to exclude are not a part of passed dataset." + Style.RESET_ALL
            self.data = self.data.drop(columns=self.columns_2_exclude)

        self.compose_X_y()
        self.pipeline = pipeline if pipeline is not None else self.make_pipeline()

        assert self.target_label in self.data.columns, Fore.RED + f"Target label needs to be a column in passed data." + Style.RESET_ALL

    @staticmethod
    def read_to_df(file_path: str) -> pd.DataFrame:
        '''
        Read a .csv or .xlsx file into a pandas DataFrame.

        Parameters:
            - file_path (Required): str, the path to the file. File format is .csv or .xlsx.

        Returns:
            - pd.DataFrame: The DataFrame containing the data from the file.
        '''
        _, extension = os.path.splitext(file_path)
        match extension:
            case ".csv":
                return pd.read_csv(file_path)
            case ".xlsx":
                return pd.read_excel(file_path)
            case _:
                raise Exception(Fore.RED + "File format not supported. Try '.csv' or '.xlsx' files." + Style.RESET_ALL)
            
    def compose_X_y(self, conversion_dict: dict = None):
        """
        Create training, testing and validation datasets.

        Parameters:
            - conversion_dict (Optional): dict, a dictionary to convert the target classes.
        """

        train_data = self.data.copy()
        
        self.X_data = train_data.drop(columns= [self.target_label] if isinstance(self.target_label, str) else self.target_label)
        self.y_data = train_data[self.target_label]
        
        #convert risk classes into risk probabilites
        if conversion_dict:
            self.y_data=self.convert_target_class(self.y_data, conversion_dict)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_data, self.y_data, test_size=0.2)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=0.1)

    def make_pipeline(self):
        '''
        Create a pipeline for data transformation.
        Applies OneHotEncoding to categorical columns and MinMaxScaling to numerical columns.

        Returns:
            - sklearn.compose.ColumnTransformer: The pipeline for data transformation.
        '''
        num_cols = self.X_data.select_dtypes(include=np.number).columns
        cat_cols = self.X_data.select_dtypes(exclude=np.number).columns

        categorical_transformer = Pipeline([
                ('onehot', OneHotEncoder(handle_unknown = 'ignore'))
            ])
        numeric_transformer = Pipeline([
                ('scaler', MinMaxScaler())
            ])

        preproc = ColumnTransformer([
                ('categoricals', categorical_transformer, cat_cols),
                ('numericals', numeric_transformer, num_cols)
            ],remainder = 'drop')
        
        return preproc
    

    def convert_target_class(self, y: pd.Series, conversion_dict: dict, to_float: bool = False) -> pd.Series:
        """
        Converts the target class using a given conversion dictionary.

        Parameters:
        - y: pd.Series, the target series to be transformed.
        - conversion_dict: dict, mapping either:
            - int: (float, float), or
            - (float, float): int.

        Returns:
        - pd.Series: Transformed series.
        """
        # Check if keys are tuples (probability to label conversion)
        if to_float:
            trans_y = []
            for prob in y:
                found = False
                for label, prob_range in conversion_dict.items():
                    if prob < 0:
                        prob = 0

                    if prob_range[1] <= prob <= prob_range[0]:  # check if probability falls within range
                        trans_y.append(label)
                        found = True
                        break
                if not found:
                    raise ValueError(Fore.RED + f"Probability {prob} does not fit into any range in conversion_dict." + Style.RESET_ALL)
            return pd.Series(trans_y, index=y.index)
        
        else:
            trans_y = []
            for label in y:
                if label in conversion_dict:
                    trans_y.append(conversion_dict[label][0])  # use lower bound for probability
                else:
                    raise ValueError(f"Label {label} not found in conversion_dict.")
            return pd.Series(trans_y, index=y.index)
            
    @staticmethod
    def postcode_normalize(postcodes: list[str]) -> np.array:
        '''
        Normalize input postcodes. Example: ' sW7 2az  ' -> 'SW7 2AZ'
        Postcode consists of an inward and outward part. 
        The inward part in 'SW7 2AZ' is '2AZ', and the outward part is 'SW7'.

        Parameters:
            - postcode: sequence of str, the input postcodes to normalize
            
        Returns:
            - array of str, the normalized postcodes
        '''
        inward_regex = r"^\d[A-Z]{2}$"
        outward_regex = r"^[A-Z]{2}\d$"

        normalized_postcodes = []
        for postcode in np.array(postcodes, ndmin=1):
            postcode = str(postcode).upper().strip()

            inward = postcode[-3:]
            outward = postcode[:-3]

            outward = outward.upper().strip()
            inward = inward.upper().strip()

            if not re.match(inward_regex, inward) or not re.match(outward_regex, outward):
                print(Fore.YELLOW + 'Postcode '+postcode+' is invalid.' + Style.RESET_ALL)
                normalized_postcodes.append(np.nan)
                continue

            normalized_postcodes.append(outward+' '+inward)
        return np.array(normalized_postcodes)

    @staticmethod
    def append_data_to_df(unlabelled_file_path : str,
                        predictions_2_append : List[Tuple[pd.Series, str]],
                        csv_db_file : str,
                        labelled_file_path : str = None):
        
        '''
        Append data to a csv file. If the databse file does not exist, the unlabelled file and the labelled file are merged to create it.
        Otherwise, if the database file exists, the predictions are appended to it.

        Parameters:
            - unlabelled_file_path : str, the path to the unlabelled file.
            - predictions_2_append : List[Tuple[pd.Series, str]], the predictions to append to the file.
            - csv_db_file : str, the path to the csv file to append the data to.
            - labelled_file_path (Optional): str, the path to the labelled file.
        '''
        assert os.path.exists(unlabelled_file_path), Fore.RED + "Invalid unlabelled file path passed to append_data_from_csv." + Style.RESET_ALL
        unlabelled_df = pd.read_csv(unlabelled_file_path)
        for prediction_tuple in predictions_2_append:
            unlabelled_df[prediction_tuple[1]] = prediction_tuple[0]
            
        if os.path.exists(csv_db_file): # append predictions to existing database file
            database_df = Tr.read_to_df(csv_db_file)
            assert all(column_name in database_df.columns for column_name in unlabelled_df.columns), Fore.RED + "The columns of the unlabelled file do not match the columns of the database file." + Style.RESET_ALL
            candidate_df = pd.concat([database_df, unlabelled_df], ignore_index=True)
            candidate_df.drop_duplicates(subset='postcode', keep='first', inplace=True)
            candidate_df.to_csv(csv_db_file, index=False)
            print(Fore.GREEN + f"A database file has been updated and saved as {csv_db_file}." + Style.RESET_ALL)
        else:
            assert os.path.exists(labelled_file_path), Fore.RED + "Invalid labelled file path passed to append_data_from_csv." + Style.RESET_ALL   # create the db file by merging the labelled file with the new predictions
            labelled_df = pd.read_csv(labelled_file_path)
            candidate_df = pd.concat([labelled_df, unlabelled_df], ignore_index=True)
            candidate_df.to_csv(csv_db_file, index=False)
            print(Fore.GREEN + f"A database file has been created and saved as {csv_db_file}." + Style.RESET_ALL)
        return