from typing import Union

from datetime import datetime
import pandas as pd

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler

from sklearn.model_selection import train_test_split


def combine_texts(data: pd.DataFrame):
    time_o = datetime.now()
    df_comb = pd.DataFrame()
    columns = data.columns
    print(columns)
    if len(columns) < 0 or type(data[columns[0]][0]) != str or type(data[columns[1]][0]) != str:
        print("ERROR, dataframe must have 1 column for each text")
        return None

    text_1 = columns[0]
    text_2 = columns[1]
    df_comb["Text"] = data[text_1] + " " + data[text_2]
    print(f"TEXT DF COMBINED AT: {time_o}")
    return df_comb


def get_split_data(df_text:pd.DataFrame, df_label:Union[pd.DataFrame, list], balanced:bool=True, as_vectors:bool=True):
    print("SPLITTING TEST_TRAIN_VAL DATA")
    X_train, X_test, y_train, y_test = train_test_split(df_text, df_label, test_size=0.2, random_state=42)
    # SMOTE
    if balanced:
        X_resampled, y_resampled = get_balanced_data(X_train, y_train, as_vectors)
        print("TEST:", len(X_test), len(y_test))
        
        return X_resampled, X_test, y_resampled, y_test
    
    # No balancing
    print("TRAIN:", len(X_train), len(y_train))
    print("TEST:", len(X_test), len(y_test))

    return X_train, X_test, y_train, y_test


def get_balanced_data(X_train, y_train, as_vectors:bool=True):
    # SMOTE
    if as_vectors:
        oversampler = SMOTE(random_state=42)
        X_resampled, y_resampled = oversampler.fit_resample(X_train, y_train)
        
        print(f"DATA BALANCED VIA SMOTE AT {datetime.now()}")
        print("TRAIN:", len(X_resampled), len(y_resampled))
        return X_resampled, y_resampled
    # RandomSampler
    oversampler = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = oversampler.fit_resample(X_train, y_train)
    
    print(f"DATA BALANCED VIA RANDOMOVERSAMPLER AT {datetime.now()}")
    print("TRAIN:", len(X_resampled), len(y_resampled))
    return X_resampled, y_resampled

