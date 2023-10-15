# Tutorial: Training and Running Sentiment Analysis Model with MLflow and Spark

## Prerequisites
Before starting, make sure you have the following prerequisites installed on your system:
- Python environment
- MLflow
- Apache Spark

## Setup

1. **Create a Python Environment:** Create a virtual Python environment and activate it.

2. **Install Required Dependencies:** Navigate to your project directory and install the required dependencies listed in the `requirements.txt` file.

## Training the Sentiment Analysis Model

1. **Training the Model:** In one terminal, run the following command to train the sentiment analysis model.
    ```python
   python train_model.py 
    ```
   
2. **Register the Model with MLflow:** The trained model will be registered with MLflow for later use. You can access it through the MLflow UI.

## Running the Sentiment Analysis Model

Now, let's make predictions using the trained model.
    
    
    ```python  
    python run.py 
    ```
    

1. **Run Model and Predict:** In another terminal, run the following command to run the model and make predictions.

## Visualizing MLflow Results

You can visualize the results using the MLflow UI. To do this, run the MLflow UI server with the following command:

Access the MLflow UI by opening a web browser and navigating to `http://localhost:5000`. Here, you can explore the model details, metrics, and more.

## Visualizing Spark Results
    http://localhost:4040/

## Installing Required Dependencies

Before you start, you'll need to install the required dependencies. Follow these steps:

1. **Create a Python Environment:** Create a virtual Python environment and activate it.

2. **Install Required Dependencies:** Navigate to your project directory and install the required dependencies listed in the `requirements.txt` file.

Congratulations! You've successfully trained and run a sentiment analysis model with MLflow and Spark.

Feel free to adapt this tutorial for your specific use case and extend it with additional features or data.
