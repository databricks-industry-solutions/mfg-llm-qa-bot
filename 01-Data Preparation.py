# Databricks notebook source
# MAGIC %pip install -U chromadb langchain transformers

# COMMAND ----------

# MAGIC %sh ls /dbfs/FileStore/ubuntu/archive/Ubuntu-dialogue-corpus

# COMMAND ----------

# MAGIC %sh ls /dbfs/FileStore/ubuntu/cleaned/extracted/ubuntu_csvfiles

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

#!cat /dbfs/FileStore/ubuntu/cleaned/ubuntu_dataset.tgz.a* > /dbfs/FileStore/ubuntu/cleaned/ubuntu_dataset.tgz

# COMMAND ----------

# Specify the path to the folder containing the CSV files
folder_path = "/FileStore/ubuntu/cleaned/extracted/ubuntu_csvfiles/trainset.csv"

# Read the CSV file with column names
ubuntu_corpus = spark.read.csv(folder_path)

# # Rename the columns
ubuntu_corpus = ubuntu_corpus.withColumnRenamed("_c0", "context") \
       .withColumnRenamed("_c1", "response") \
       .withColumnRenamed("_c2", "flag")

display(ubuntu_corpus)

# COMMAND ----------

# %sql
# CREATE SCHEMA manu_llm_solution_accelerator

# COMMAND ----------

from bs4 import BeautifulSoup
from pyspark.sql.functions import col, udf, length, pandas_udf

#UDF to transform html content as text
@pandas_udf("string")
def html_to_text(html):
  return html.apply(lambda x: BeautifulSoup(x).get_text())

ubuntu_corpus_df =(ubuntu_corpus
                  .filter("flag = 1") # keep only good answer/question
                  # .filter(length("_Body") <= 1000) #remove too long questions
                  .withColumn("context", html_to_text("context")) #Convert html to text
                  .withColumn("response", html_to_text("response")))

# Save 'raw' content for later loading of questions
ubuntu_corpus_df.write.mode("overwrite").saveAsTable(f"manu_llm_solution_accelerator.dev.ubuntu_corpus")

# COMMAND ----------


