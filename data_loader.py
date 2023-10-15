from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

def read_csv_to_dataframe(spark, file_path):
    """
    Read a CSV file and create a PySpark DataFrame.

    Args:
        spark (pyspark.sql.SparkSession): The SparkSession instance.
        file_path (str): The path to the CSV file.

    Returns:
        pyspark.sql.DataFrame: A PySpark DataFrame.
    """
    schema = StructType([
    StructField("id", IntegerType(), True),
    StructField("company", StringType(), True),
    StructField("text", StringType(), True),
    StructField("category", StringType(), True),
    
    ])  

    # Define the data for the new entries with longer 'text' statements
    data = [
        (1, 'Apple', 'Uncertainty looms over the future of Apple stocks as investors remain cautiously optimistic amidst global economic challenges', 'Product'),
        (2, 'Tesla', 'Tesla enthusiasts eagerly anticipate a flurry of innovative upgrades and cutting-edge technology as the electric vehicle giant continues to lead the automotive revolution', 'Automotive'),
        (3, 'Amazon', 'Investors are thrilled as Amazon\'s quarterly earnings report exceeds expectations, igniting hopes for robust growth in the e-commerce sector', 'E-commerce'),
        (4, 'Microsoft', 'Microsoft grapples with a myriad of regulatory challenges that threaten to alter the tech giant\'s course in a rapidly evolving industry', 'Technology'),
        (5, 'Google', 'Alphabet Inc. showcases remarkable financial performance, leaving analysts and investors in awe of the company\'s steadfast dominance in the technology sector', 'Technology'),
        (6, 'Facebook', 'Social media stocks soar on the back of positive sentiment and strong user engagement, but privacy concerns continue to lurk in the background', 'Technology'),
        (7, 'Netflix', 'Netflix\'s streaming service gains subscribers at a record pace, thanks to its compelling content library and a growing global audience', 'Entertainment'),
        (8, 'Disney', "Disney's latest movie release receives a mix of glowing reviews and critical scrutiny, setting the stage for intriguing discussions among movie buffs", 'Entertainment'),
        (9, 'Amazon', 'A widespread Amazon Web Services outage triggers disruptions across numerous websites, highlighting the need for robust cloud infrastructure', 'Technology'),
        (10, 'Tesla', 'Tesla surprises the financial world with an announcement of record-breaking quarterly profits, further cementing its position in the automotive industry', 'Automotive'),
        (11, 'Apple', 'Apple confirms the launch date of the highly anticipated iPhone 13, raising the stakes in the ever-evolving smartphone market', 'Product'),
        (12, 'Microsoft', 'Microsoft makes a bold move by acquiring a promising AI startup, promising to reshape the landscape of artificial intelligence', 'Technology')
    ]

    # Create the DataFrame with the new entries
    df = spark.createDataFrame(data, schema=['id', 'company', 'text', 'category'])

    # Show the updated DataFrame
    df.show(truncate=False)


    return df

