# Databricks notebook source
# MAGIC %pip install -U chromadb langchain transformers

# COMMAND ----------

# MAGIC %sh ls /dbfs/FileStore/ubuntu/archive/Ubuntu-dialogue-corpus

# COMMAND ----------

# MAGIC %sh ls /tmp/rkm/ubuntu_csvfiles

# COMMAND ----------

# import urllib.request

# base_url = "http://dataset.cs.mcgill.ca/ubuntu-corpus-1.0/"
# file_names = ["ubuntu_dataset.tgz.aa", "ubuntu_dataset.tgz.ab", "ubuntu_dataset.tgz.ac","ubuntu_dataset.tgz.ad","ubuntu_dataset.tgz.ae"]
# dbfs_path = "/dbfs/FileStore/ubuntu/cleaned/"

# for file_name in file_names:
#     url = base_url + file_name
#     destination = dbfs_path + file_name

#     urllib.request.urlretrieve(url, destination)

# COMMAND ----------

#!cat ubuntu_dataset.tgz.a* > ubuntu_dataset.tgz

# COMMAND ----------

# Specify the path to the folder containing the CSV files
folder_path = "/tmp/rkm/ubuntu_csvfiles/trainset.csv"

# Read the CSV file with column names
df = spark.read.csv(folder_path)

# Rename the columns
df = df.withColumnRenamed("_c0", "context") \
       .withColumnRenamed("_c1", "response") \
       .withColumnRenamed("_c2", "flag")

# Read CSV files into a DataFrame
#df = spark.read.format("csv").load(folder_path)

display(df)

# COMMAND ----------

# Specify the path to the folder containing the CSV files
folder_path = "FileStore/ubuntu/archive/Ubuntu-dialogue-corpus"

# Read CSV files into a DataFrame
df = spark.read.format("csv").option("header", "true").load(folder_path + "/*.csv")

# Process the DataFrame as needed
display(df)

# COMMAND ----------


