# Databricks notebook source
# MAGIC %pip install -U chromadb langchain transformers

# COMMAND ----------

# MAGIC %sh ls /dbfs/FileStore/ubuntu/archive/Ubuntu-dialogue-corpus

# COMMAND ----------

# MAGIC %sh ls /dbfs/FileStore/ubuntu

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

!cat /dbfs/FileStore/ubuntu/cleaned/ubuntu_dataset.tgz.a* > /dbfs/FileStore/ubuntu/cleaned/ubuntu_dataset.tgz

# COMMAND ----------

!tar -xzf /dbfs/FileStore/ubuntu/cleaned/ubuntu_dataset.tgz 

# COMMAND ----------

import tarfile

file_paths = [
    "/dbfs/FileStore/ubuntu/cleaned/ubuntu_dataset.tgz.aa",
    "/dbfs/FileStore/ubuntu/cleaned/ubuntu_dataset.tgz.ab",
    "/dbfs/FileStore/ubuntu/cleaned/ubuntu_dataset.tgz.ac",
    "/dbfs/FileStore/ubuntu/cleaned/ubuntu_dataset.tgz.ad",
    "/dbfs/FileStore/ubuntu/cleaned/ubuntu_dataset.tgz.ae"
]

extract_path = "/dbfs/FileStore/ubuntu/cleaned_extracted/"

for file_path in file_paths:
    with tarfile.open(file_path, "r:gz") as tar:
        tar.extractall(extract_path)

# COMMAND ----------

# Specify the path to the folder containing the CSV files
folder_path = "/FileStore/ubuntu/cleaned_extracted/ubuntu_csvfiles/trainset.csv"

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


