import argparse
from colorama import Fore, Style, init
from src.tool import Tool
from src.data_transformation import Tr
import argparse

def setup_cli():
    parser = argparse.ArgumentParser(description='UK Flood Risk Detection Tool')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Predict risk command
    predict_parser = subparsers.add_parser(
        'predict-risk', 
        help='Predict flood risk for postcodes based on the postcode db file'
    )
    predict_parser.add_argument(
        'postcodes', 
        nargs='+', 
        help='One or more UK postcodes to predict flood risk for (e.g., "SW1A 1AA")'
    )

    # Add file to postcode db
    add_file_parser = subparsers.add_parser(
        'add-file', 
        help='Add a file to the postcode db for use in predictions. Specify the file using -file, the method for risk prediction using -method-risk and the method for house price prediction using -method-house-price.'
    )
    add_file_parser.add_argument(
        'file', 
        type=str, 
        help='Path to the file to add to the postcode database (e.g., "data/file.csv")'
    )
    add_file_parser.add_argument(
        '--method-risk', 
        type=str, 
        help=(
            'Optional: Specify the prediction method to calculate flood risk '
            'from the file. If not specified, the default model will be used.'
            'Search for available methods using the list-methods command.'
        )
    )
    add_file_parser.add_argument(
        '--method-house-price', 
        type=str, 
        help=(
            'Optional: Specify the prediction method to calculate house prices '
            'from the file. If not specified, the default model will be used.'
            'Search for available methods using the list-methods command.'
        )
    )

    # List methods command
    subparsers.add_parser(
        'list-methods', 
        help='List available prediction methods for flood risk and house price calculations'
    )

    return parser


def main():
    init(autoreset=True)  # Initialize colorama for colored output
    parser = setup_cli()
    args = parser.parse_args()
    # Initialize the tool only if required
    tool = None
    if args.command in ['predict-risk', 'add-file', 'list-methods']:
        try:
            tool = Tool()
            print(Fore.GREEN + "Tool initialized successfully" + Style.RESET_ALL)
        except Exception as e:
            print(Fore.RED + f"Error initializing tool: {str(e)}" + Style.RESET_ALL)
            return

    if args.command == 'predict-risk':
        try:
            predictions = tool.get_risk_value(args.postcodes)
            for postcode, risk in predictions:
                print(Fore.GREEN + f"Postcode: {postcode} - Annual Risk Level: {risk:.2f}" + Style.RESET_ALL)
        except Exception as e:
            print(Fore.RED + f"Unexpected error during prediction: {str(e)}" + Style.RESET_ALL)

    elif args.command == 'add-file':
        try:
            df_2_append = Tr.read_to_df(args.file)
            predictions_risk = None
            predictions_house_price = None

            if args.method_risk:
                predictions_risk = tool.risk_model.predict(df_2_append, args.method_risk)
            if args.method_house_price:
                predictions_house_price = tool.house_price_model.predict(df_2_append, args.method_house_price)

            tool.add_file_to_db(
                args.file,
                (predictions_risk, "riskLabel"),
                (predictions_house_price, "medianPrice")
            )
            print(Fore.GREEN + "File added successfully" + Style.RESET_ALL)
        except Exception as e:
            print(Fore.RED + f"Error adding file: {str(e)}" + Style.RESET_ALL)

    elif args.command == 'list-methods':
        try:
            print(Fore.GREEN + "Available flood risk prediction methods:" + Style.RESET_ALL)
            tool.risk_model.get_available_models

            print(Fore.GREEN + "\nAvailable house price prediction methods:" + Style.RESET_ALL)
            tool.house_price_model.get_available_models
                
        except Exception as e:
            print(Fore.RED + f"Error listing methods: {str(e)}" + Style.RESET_ALL)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
    