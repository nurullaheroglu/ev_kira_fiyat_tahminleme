import joblib
from helper import main_df_generate
from config import generate_params, hyperparameter_optimization
import pandas as pd


def main():
   df = pd.read_csv("concat_df.csv", index_col="Unnamed: 0")
   X, y, df_final = main_df_generate(df)
   final_model = hyperparameter_optimization(X, y)
   joblib.dump(final_model, "final_model2.pkl")
   df_final.to_csv("house_rent_price_dataset2.csv")
   return final_model


if __name__ == "__main__":
   main()

