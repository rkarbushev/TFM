import scipy
import pandas as pd

from sklearn.metrics import precision_recall_fscore_support, classification_report

class COS_DIST():

    def cosine_distance_wordembedding(self, s1, s2):
        """gets two texts, obtains their word embeddings and then returns its cosine distance"""

        cosine = 1 - scipy.spatial.distance.cosine(s1, s2)
        return round((cosine),2)

    def get_predictions(self, tokens_1, tokens_2, label_df):
        predicted = []
        i = 0
        for tok1 in tokens_1:
            tok2 = tokens_2[i]
            pred = self.cosine_distance_wordembedding(tok1, tok2)
            if pred < 0.91:
                predicted.append(0)
            else:
                predicted.append(1)
            i = i+1

        pos = 0
        neg = 0
        for i in predicted:
            if i == 1:
                pos+=1
            else:
                neg+=1

        print(f"{pos}/1501 Positives")
        print(f"{neg}/6100 Negatives")

        predicted_series = pd.Series(predicted)

        print(precision_recall_fscore_support(label_df, predicted_series, average=None))
        print("-------------------------------------------------")
        print( classification_report(label_df, predicted_series))