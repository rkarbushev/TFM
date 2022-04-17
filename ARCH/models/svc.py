from datetime import datetime

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline


class SVM():
    """
    Support Vector Machine model
    """
    svc_model = None
    pipeline = None

    def __init__(self) -> None:
        self.svc_model = SVC()
        self.pipeline = Pipeline([
                            ('classifier', SVC()),
                        ])

    
    def _fit(self, X_train, y_train):
        print("TRAINING SVC MODEL")
        time_o = datetime.now()
        try:
            self.pipeline.fit(X_train, y_train)
            print(f"SVC MODEL TRAINED AT: {time_o}")
            return self.svc_model
        except Exception:
            print("Something went wrong training svc model:", Exception)
            return None

    def _predict(self, X_test):
        print("PREDICTING DATA WITH SVC MODEL")
        time_o = datetime.now()
        predictions = self.pipeline.predict(X_test)
        print(f"DATA PREDICTED WITH {self.svc_model} AT {time_o}")
        return predictions
    