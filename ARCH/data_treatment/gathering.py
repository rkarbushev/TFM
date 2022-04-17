import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt


def plot_class_diff(data):
    print("Positives =", len(data[data["Label"] == 1]))
    print("Negatives =", len(data[data["Label"] == 0]))
    fig = plt.figure(figsize =(5, 5))
    plot_data = [len(data[data["Label"] == 0]), len(data[data["Label"] == 1])]
    plt.pie(plot_data, labels = ["Negative", "Positive"])
    plt.show(block=True)
    return None

def get_df_from_csv(path:str):
    time_o = datetime.now()
    df = pd.read_csv(path)
    print(f"DATA RETRIEVED FROM {path} AT: {time_o}")
    #plot_class_diff(df)
    return df