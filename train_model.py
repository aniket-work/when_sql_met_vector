import mlflow.pyfunc
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from mlflow_register import register_sentiment_model

# Define constants
MODEL_NAME          = "distilbert-base-uncased-finetuned-sst-2-english"
SENTIMENT_COLUMN    = "text"
MODEL_KEY           = "model"

class SentimentAnalysisModel(mlflow.pyfunc.PythonModel):
    
    def __init__(self, model_name):
        """
        Initialize the SentimentAnalysisModel with a pre-trained BERT-based model.

        Args:
            model_name (str): The name of the pre-trained BERT-based model to use.
        """
        print(f"Initializing SentimentAnalysisModel with model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        print("Model initialization complete.")

    def predict(self, context, model_input):
        """
        Predict sentiment using the initialized model.

        Args:
            context: Additional context data (not used in this example).
            model_input (pd.DataFrame): Input data with a "text" column containing text to analyze.

        Returns:
            pd.DataFrame: A DataFrame with sentiment predictions.
        """
        input_text = self._process_input(model_input)
        predictions = self._get_sentiment(input_text)
        return predictions

    def _process_input(self, model_input):
        """
        Process the input data, ensuring it's in a suitable format for sentiment analysis.

        Args:
            model_input (pd.DataFrame): Input data with a "text" column.

        Returns:
            List[str]: A list of text inputs to analyze.
        """
        print("Processing input data...")
        if SENTIMENT_COLUMN in model_input:
            if isinstance(model_input[SENTIMENT_COLUMN], str):
                input_text = [model_input[SENTIMENT_COLUMN]]
            else:
                input_text = model_input[SENTIMENT_COLUMN].astype(str).values.tolist()
        else:
            input_text = [model_input["0"].iloc[0]]
        print("Input data processing complete.")
        return input_text

    def _get_sentiment(self, input_text):
        """
        Perform sentiment analysis on the provided text.

        Args:
            input_text (List[str]): Text inputs to analyze.

        Returns:
            pd.DataFrame: A DataFrame with sentiment predictions.
        """
        print("Performing sentiment analysis...")
        inputs = self.tokenizer(input_text, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**inputs).logits

        predicted_class_id = logits.argmax().item()
        sentiment_label = self.model.config.id2label[predicted_class_id]
        print("Sentiment analysis complete.")

        return pd.DataFrame([sentiment_label])

# Initialize a SentimentAnalysisModel using the specified model name
sentiment_model = SentimentAnalysisModel(MODEL_NAME)

# Sample values for parameters, metrics, and tags
model_params = {"model_name": MODEL_NAME}
metrics = {"accuracy": 0.85, "precision": 0.88, "recall": 0.82, "f1_score": 0.85}
tags = {"framework": "PyTorch", "dataset": "Financial News"}

# Register the sentiment analysis model with MLflow
register_sentiment_model(MODEL_NAME, sentiment_model, MODEL_KEY, model_params, metrics, tags)