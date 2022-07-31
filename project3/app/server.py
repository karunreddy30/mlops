from fastapi import FastAPI
from pydantic import BaseModel
from loguru import logger
import joblib
import json
from datetime import datetime
import time

from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

GLOBAL_CONFIG = {
    "model": {
        "featurizer": {
            "sentence_transformer_model": "all-mpnet-base-v2",
            "sentence_transformer_embedding_dim": 768
        },
        "classifier": {
            "serialized_model_path": "../data/news_classifier.joblib"
        }
    },
    "service": {
        "log_destination": "../data/logs.out"
    }
}

class PredictRequest(BaseModel):
    source: str
    url: str
    title: str
    description: str


class PredictResponse(BaseModel):
    scores: dict
    label: str


class TransformerFeaturizer(BaseEstimator, TransformerMixin):
    def __init__(self, dim, sentence_transformer_model):
        self.dim = dim
        self.sentence_transformer_model = sentence_transformer_model

    #estimator. Since we don't have to learn anything in the featurizer, this is a no-op
    def fit(self, X, y=None):
        return self

    #transformation: return the encoding of the document as returned by the transformer model
    def transform(self, X, y=None):
        X_t = []
        for doc in X:
            X_t.append(self.sentence_transformer_model.encode(doc))
        return X_t


class NewsCategoryClassifier:
    def __init__(self, config: dict) -> None:
        self.config = config
        """
        [TO BE IMPLEMENTED]
        1. Load the sentence transformer model and initialize the `featurizer` of type `TransformerFeaturizer` (Hint: revisit Week 1 Step 4)
        2. Load the serialized model as defined in GLOBAL_CONFIG['model'] into memory and initialize `model`
        """
        sentence_transformer_model = SentenceTransformer('sentence-transformers/{model}'.format(model=config['embedding_model']))
        featurizer = TransformerFeaturizer(config['embedding_dim'], sentence_transformer_model)
        classifier = joblib.load(config['classifier_model'])
        self.class_names = classifier.classes_

        self.pipeline = Pipeline([
            ('transformer_featurizer', featurizer),
            ('classifier', classifier)
        ])

    def predict_proba(self, model_input: dict) -> dict:
        """
        [TO BE IMPLEMENTED]
        Using the `self.pipeline` constructed during initialization,
        run model inference on a given model input, and return the
        model prediction probability scores across all labels

        Output format:
        {
            "label_1": model_score_label_1,
            "label_2": model_score_label_2
            ...
        }
        """
        pred_proba = self.pipeline.predict_proba([model_input['description']])
        return {self.class_names[i]: pred_proba[0][i] for i in range(len(self.class_names)) }

    def predict_label(self, model_input: dict) -> str:
        """
        [TO BE IMPLEMENTED]
        Using the `self.pipeline` constructed during initialization,
        run model inference on a given model input, and return the
        model prediction label

        Output format: predicted label for the model input
        """
        return self.pipeline.predict(model_input['description'])[0]


app = FastAPI()
NEWS_CLASSIFIER = None
LOG_FILE = None

@app.on_event("startup")
def startup_event():
    """
        [TO BE IMPLEMENTED]
        2. Initialize the `NewsCategoryClassifier` instance to make predictions online. You should pass any relevant config parameters from `GLOBAL_CONFIG` that are needed by NewsCategoryClassifier
        3. Open an output file to write logs, at the destimation specififed by GLOBAL_CONFIG['service']['log_destination']

        Access to the model instance and log file will be needed in /predict endpoint, make sure you
        store them as global variables
    """
    config = {
                'embedding_model': GLOBAL_CONFIG['model']['featurizer']['sentence_transformer_model'],
                'embedding_dim': GLOBAL_CONFIG['model']['featurizer']['sentence_transformer_embedding_dim'],
                'classifier_model': GLOBAL_CONFIG['model']['classifier']['serialized_model_path'],
                'log_destination': GLOBAL_CONFIG['service']['log_destination']
             }
    global NEWS_CLASSIFIER
    NEWS_CLASSIFIER = NewsCategoryClassifier(config)
    global LOG_FILE
    LOG_FILE = open(config['log_destination'], "a")
    logger.info("Setup completed")


@app.on_event("shutdown")
def shutdown_event():
    # clean up
    """
        [TO BE IMPLEMENTED]
        1. Make sure to flush the log file and close any file pointers to avoid corruption
        2. Any other cleanups
    """
    LOG_FILE.flush()
    LOG_FILE.close()
    logger.info("Shutting down application")


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    # get model prediction for the input request
    # construct the data to be logged
    # construct response
    """
        [TO BE IMPLEMENTED]
        1. run model inference and get model predictions for model inputs specified in `request`
        2. Log the following data to the log file (the data should be logged to the file that was opened in `startup_event`, and writes to the path defined in GLOBAL_CONFIG['service']['log_destination'])
        {
            'timestamp': <YYYY:MM:DD HH:MM:SS> format, when the request was received,
            'request': dictionary representation of the input request,
            'prediction': dictionary representation of the response,
            'latency': time it took to serve the request, in millisec
        }
        3. Construct an instance of `PredictResponse` and return
    """
    logger.info("Predict request recieved")
    content = {}
    start = time.time()
    content['timestamp'] = datetime.fromtimestamp(start).strftime("%Y-%m-%d, %H:%M:%S")
    content['request'] = request.dict()
    response = {
        'scores': NEWS_CLASSIFIER.predict_proba(request.dict()),
        'label': NEWS_CLASSIFIER.predict_label(request.dict())
    }
    content['response'] = response
    end = time.time()
    content['latency'] = (end - start) * 1000
    LOG_FILE.write(json.dumps(content))

    return PredictResponse(**response)


@app.get("/")
def read_root():
    return {"Hello": "World"}
