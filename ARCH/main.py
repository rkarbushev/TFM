import joblib # Used for saving models
import pandas as pd
from sklearn.preprocessing import label_binarize

from models.nlp import NLP
from models.svc import SVM
from models.cos_dist import COS_DIST
from data_treatment.gathering import get_df_from_csv
from data_treatment.cleaning import get_split_data, combine_texts
from data_treatment.data_viewer import *

AVAILABLE_MODELS = {
    "SPACY_SVC": "",
    "TF_IDF": "",
    "BERT": ""
}

class Architecture():
    """
    """
    # Variables definition
    models = {}
    nlp = None
    # For normal models 
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    # Cosine distance
    tokens_1 = []
    tokens_2 = []
    label_df = pd.DataFrame

    def __init__(self, lang:str = "es") -> None:
        self.nlp = NLP(lang)


    def get_data(self, path:str = "data/parmex_train.csv", as_vectors:bool=True, combine_text:bool=False):
        df = get_df_from_csv(path)
        df_label = df["Label"]
        df_text = pd.DataFrame()
        if combine_text:
            df_text = combine_texts(df)
            if df_text is None:
                return "ERROR"
            tokens = self.nlp.get_tokens(df_text, as_vectors=as_vectors)
            self.X_train, self.X_test, self.y_train, self.y_test = get_split_data(tokens, df_label, balanced=True, as_vectors=True)
        else:
            self.label_df = df_label
            text_1_df = df["Text1"]
            text_2_df = df["Text2"]
            self.tokens_1, self.tokens_2 = self.nlp.get_separate_tokens(text_1_df, text_2_df)
            print("SEPARATED TEXTS IN TWO DIFFERENT DFs")


    def train_model(self, model:dict, force_train:bool = False ):
        m_name = model["name"]
        aux_model = model["model"]
        if force_train:
            # TRAIN MODEL
            aux_model._fit(self.X_train, self.y_train)
            # SAVE MODEL
            joblib.dump(aux_model, f'models/trained/{m_name}.pkl')

        # CHECK IF MODEL IS STORED
        try:
            saved_model = joblib.load(f'models/trained/{m_name}.pkl')
            print(f"RETURNING STORED {m_name} MODEL")
            return saved_model
        except FileNotFoundError:
            # TRAIN MODEL
            aux_model._fit(self.X_train, self.y_train)
            # SAVE MODEL
            joblib.dump(aux_model, f'models/trained/{m_name}.pkl')

            # self.models[m_name] = {
            #     "instance": model["model"],
            #     "trained_instance": saved_model
            # }
            return aux_model
    
    def predict_data(self, model_name: str):
        try:
            saved_model = joblib.load(f'models/trained/{model_name}.pkl')
        except FileNotFoundError:
            return print(f"{model_name} IS NOT STORED, IT NEEDS TO BE TRAINED FIRST")
        
        print(saved_model)
        # PREDICT DATA WITH SAVED MODEL
        predictions = saved_model._predict(self.X_test)
        pprint(class_report(self.y_test, predictions))
        # confussion_matrix(self.y_test, predictions)
        # print("+++++++++++++++++++++++++++")
        # plot_learning_curve(saved_model.pipeline, "accuracy vs. training set size", self.X_train, self.y_train, cv=5)


def get_svc_model():
    arch = Architecture()
    svc = SVM()
    model = {
        "name": "SVM",
        "model": svc
    }
    
    res = arch.train_model(model)

    if res is None:
        return "ERROR SOMETHING WENT WRONG TRAINING"
    
    arch.predict_data(model["name"])

    return "SVM DONE"


def get_cos_dist_model():
    arc = Architecture()
    arc.get_data(combine_text=False)

    cos_dist = COS_DIST()
    cos_dist.get_predictions(arc.tokens_1, arc.tokens_2, arc.label_df)
    return "All done"