import mlflow
import mlflow.pyfunc

def register_sentiment_model(model_name, sentiment_model, model_key, model_params, metrics, tags):
    # Start an MLflow run with the specified name
    with mlflow.start_run(run_name="aniket_sentiment_analysis_run") as run:
        # Log parameters
        mlflow.log_params(model_params)
        
        # Log metrics
        mlflow.log_metrics(metrics)

        # Set tags
        mlflow.set_tags(tags)

        # Log the sentiment analysis model using the specified model key
        mlflow.pyfunc.log_model(model_key, python_model=sentiment_model)
