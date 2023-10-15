import mlflow
import os
import sys
import time
from data_loader import read_csv_to_dataframe
from pyspark.sql.functions import col, expr, length
from pyspark.sql import SparkSession
from pyspark.sql.functions import struct, col

# Define constants
WEB_UI_PORT = 4040
EXPERIMENT_NAME = "Default"
MAX_LINE_LENGTH = 30
MODEL_COLUMN_NAME = "what's the sentiment"
CSV_FILE_PATH = 'data/news.csv'

# Set the Python environment for PySpark
os.environ['PYSPARK_PYTHON'] = 'python'

# Check if the web UI port is provided as a command-line argument
if len(sys.argv) > 1 and sys.argv[1] == "--webui-port":
    web_ui_port = sys.argv[2]
else:
    web_ui_port = WEB_UI_PORT  # Default port

# Initialize the MLflow experiment
print("\n============== LOAD LAST RUN ==================\n")
last_parent_run = set()
exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

# Retrieve the ID of the last run from the MLflow experiment
df = mlflow.search_runs([exp.experiment_id], order_by=["Created DESC"])
last_run_id = df.loc[0, 'run_id']
print(last_run_id)

# Initialize a Spark session
print("\n============== LOAD LAST RUN MODEL ==================\n")
spark = SparkSession.builder.appName("example").config("spark.ui.port", web_ui_port).getOrCreate()

# Load the trained model from the last run
logged_model = f'runs:/{last_run_id}/model'
print("\n============== LOAD MODEL AS SPARK UDF ==================")
loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model, result_type='string')

# Create a Spark DataFrame from a CSV file
print("\n============== CREATE SPARK DATAFRAME ==================\n")
df = read_csv_to_dataframe(spark, CSV_FILE_PATH)
df.explain()
df.show()

# Add model predictions to the DataFrame
print("\n============== ADD MODEL AS PREDICTIONS ==================\n")
df2 = df.withColumn(MODEL_COLUMN_NAME, loaded_model(struct(*map(col, df.columns))))
df2.explain()

# Define a user-defined function (UDF) for word wrapping text
def word_wrap(text):
    return '\n'.join([text[i:i + MAX_LINE_LENGTH] for i in range(0, len(text), MAX_LINE_LENGTH)])

# Register the UDF with Spark
spark.udf.register("word_wrap", word_wrap)

# Create a new DataFrame with word-wrapped text
wrapped_df = df2.withColumn("wrapped_text", expr("word_wrap(text)"))

# Show the DataFrame with word wrap and no truncation
wrapped_df.select("id", "company", "wrapped_text", "category", MODEL_COLUMN_NAME).show(truncate=False)

# Explain the DataFrame execution plans
df2.explain('simple')
df2.explain('extended')
df2.explain('cost')
df2.explain('formatted')

# Sleep for 1 hour (this is, just to keep spark UI running to analysis)
time.sleep(3600)
