from data.file_mapping import *
from src.model import Model
from src.data_transformation import Tr
import pandas as pd
from typing import Optional
from colorama import Fore, Style

RISK_DICT_BOUND = {10:(5,4),
                    9:(4,3),
                    8:(3,2),
                    7:(2,1.5),
                    6:(1.5,1),
                    5:(1,0.5),
                    4:(0.5,0.1),
                    3:(0.1,0.05),
                    2:(0.05,0.01),
                    1:(0.01, 0)}

class Tool(object):
    """Class to interact with a postcode database file."""

    def __init__(self, household_file: Optional[str] =HOUSEHOLDS_PER_SECTOR, postcode_file: Optional[str] =POSTCODES_LABELLED_FILE,):
        """
        Initialize the Tool object.
        Reads the unlabelled and labelled postcode files and appends the predictions to the postcode database file.
        If the postcode database file does not exist, it creates it by merging the labelled file with the unlabelled file.

        Parameters:
            - full_postcode_file : str, optional
                Filename of a .csv file containing geographic location
                data for postcodes.
            - household_file : str, optional
                Filename of a .csv file containing information on households
                by postcode.
        """

        self.postcode_labelled_path = postcode_file
        self.postcode_unlabelled_path = os.path.join(os.path.dirname(postcode_file), 'postcodes_unlabelled.csv')
        self.postcodedb_path = os.path.join("data", "csv", "postcode_db.csv")
        self.households_path = household_file

        self.risk_data_transformer = Tr('riskLabel', data_path=self.postcode_labelled_path, columns_2_exclude=['medianPrice'])
        self.risk_data_transformer.compose_X_y(RISK_DICT_BOUND)
        self.risk_model = Model(self.risk_data_transformer)
        self.risk_model.grid_search()
        
        self.house_price_transformer = Tr('medianPrice', data_path=self.postcode_labelled_path, columns_2_exclude=['riskLabel'])
        self.house_price_transformer.compose_X_y()
        self.house_price_model = Model(self.house_price_transformer)
        self.house_price_model.grid_search()
        
        
        if not os.path.exists(self.postcodedb_path):
            risk_predictions = self.risk_model.predict(self.risk_data_transformer.X_test)
            risk_predictions = self.risk_data_transformer.convert_target_class(risk_predictions, RISK_DICT_BOUND, to_float=True)
            house_price_predictions = self.house_price_model.predict(self.house_price_transformer.X_test)

            Tr.append_data_to_df(
                self.postcode_unlabelled_path,
                [(risk_predictions, "riskLabel"),
                 (house_price_predictions, "medianPrice")],
                self.postcodedb_path,
                self.postcode_labelled_path)

        self.postcodedb_df = pd.read_csv(self.postcodedb_path)
        print(self.postcodedb_df.head())

    def add_file_to_db(self, file_path: str, predictions_2_append: tuple):
        '''
        Add a new file to the postcode database file.

        Parameters:
            - file_path : str, the path to the file to add to the database.
            - predictions_2_append : tuple, the predictions to append to the database.
        '''
        Tr.append_data_to_df(file_path, predictions_2_append, self.postcodedb_path)
    
    def get_risk_value(self, postcodes:list[str]):
        '''
        Get risk value from the postcode database file.

        Parameters:
            - postcodes : list, the postcodes to get the risk value for.

        Returns:
            - risk_values : list[tuple[str, float]], the risk values for the postcodes.
        '''
        if not all(postcode in self.postcodedb_df['postcode'].values for postcode in postcodes):
            print(Fore.RED + "One or more postcodes are not in the database." + Style.RESET_ALL)
        risk_values = []
        for postcode in postcodes:
            row = self.postcodedb_df[self.postcodedb_df['postcode'] == postcode]
            risk = row['riskLabel'].values[0]
            price = row['medianPrice'].values[0]
            weighted_risk = risk * 0.05 * price
            risk_values.append((postcode, weighted_risk))
        return risk_values



