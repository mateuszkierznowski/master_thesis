import pandas as pd

stat_dict = {"olx": 0,
             "autoscout": 0,
             "otomoto": 0}

pd.DataFrame(stat_dict, index=[0]).to_csv("hej.csv")
df = pd.read_csv("hej.csv")