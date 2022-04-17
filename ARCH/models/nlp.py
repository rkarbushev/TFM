from datetime import datetime
import spacy
import pandas as pd

class NLP():
    model = None
    def __init__(self, lang:str = "es") -> None:
        try:
            self.model = spacy.load("es_core_news_lg")
            print("SPACY MODEL SUCCESSFULY LOADED")
        except Exception:
            print("Something went wrong initializing NLP model", Exception)

    def get_tokens(self, data: pd.DataFrame, as_vectors:bool = True):
        print("GETTING TEXT TOKENS")
        time_o = datetime.now()
        tokens=[]
        for text in data["Text"]: 
            tok = self.model(text)
            if as_vectors:
                tokens.append(tok.vector)
            else:
                tokens.append(tok)
        print(f"TOKENS RETRIEVED FROM TEXT AT: {time_o}")
        print(len(tokens))

        return tokens

    def get_separate_tokens(self, text_1_df, text_2_df):
        time_o = datetime.now()
        print(f"GETTING SEPARATE TOKENS AT: {time_o}")
        tokens_1=[]
        tokens_2=[]
        for i, text in text_1_df.items(): 
            tok_1 = self.model(text)
            tok_2 = self.model(text_2_df[i])
            tokens_1.append(tok_1.vector)
            tokens_2.append(tok_2.vector)
        print(f"SEPARATE TOKENS READY: {datetime.now()}")
        print(len(tokens_1), len(tokens_2))
        return tokens_1, tokens_2