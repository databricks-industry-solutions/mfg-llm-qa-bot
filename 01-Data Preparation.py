# Databricks notebook source
# MAGIC %pip install -U chromadb langchain transformers

# COMMAND ----------

# MAGIC %sh ls /dbfs/FileStore/ubuntu/archive/Ubuntu-dialogue-corpus

# COMMAND ----------

# MAGIC %sh ls /dbfs/FileStore/ubuntu/cleaned/extracted/ubuntu_csvfiles

# COMMAND ----------

# %run ./_resources/00-init $catalog=manu_llm_solution_accelerator $db=dev

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

# MAGIC %sql
# MAGIC SELECT * 
# MAGIC FROM manu_llm_solution_accelerator.dev.ubuntu_corpus

# COMMAND ----------

# MAGIC %md
# MAGIC # KAGGLE Dataset 

# COMMAND ----------

# Specify the path to the folder containing the CSV files
folder_path = "FileStore/ubuntu/archive/Ubuntu-dialogue-corpus"

# Read CSV files into a DataFrame
df = spark.read.format("csv").option("header", "true").load(folder_path + "/*.csv")

# Remove duplicates based on all columns
df = df.dropDuplicates()

df = df.orderBy('date')

# Process the DataFrame as needed
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 2/ Clean & prepare our gardenig questions and best answers 
# MAGIC
# MAGIC We will perform some light preprocessing on the results:
# MAGIC - Create a conversation based on the dialogueID and folder for unique conversations
# MAGIC - Join questions and answers to form question-answer pairs
# MAGIC
# MAGIC *Note that this pipeline is basic. For more advanced ingestion example with Databricks lakehouse, try Delta Live Table: `dbdemos.instal('dlt_loan')`*

# COMMAND ----------

from pyspark.sql.functions import concat_ws, collect_list

# Group the chat logs by dialogueID and folder
grouped = df.groupBy('dialogueID','folder')

# Collect the texts for each dialogue into a list
conversation_corpus = grouped.agg(collect_list('text').alias('conversation'))

conversation_corpus = conversation_corpus.withColumn('conversation_text', concat_ws(', ', 'conversation'))

display(conversation_corpus)

# COMMAND ----------

from bs4 import BeautifulSoup
from pyspark.sql.functions import col, udf, length, pandas_udf

#UDF to transform html content as text
@pandas_udf("string")
def html_to_text(html):
  return html.apply(lambda x: BeautifulSoup(x).get_text())

conversation_corpus =(conversation_corpus
                  .filter(length("conversation_text") <= 1000) #remove too long questions
                  .withColumn("conversation_text", html_to_text("conversation_text"))) #Convert html to text

display(conversation_corpus)

# Save 'raw' content for later loading of questions
#conversation_corpus.write.mode("overwrite").saveAsTable(f"manu_llm_solution_accelerator.dev.ubuntu_corpus")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3/ Load our model to transform our docs to embeddings
# MAGIC
# MAGIC We will simply load a sentence to embedding model from hugging face and use it later in the chromadb client.

# COMMAND ----------

from langchain.embeddings import HuggingFaceEmbeddings

# Download model from Hugging face
hf_embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 4/ Index the documents (rows) in our vector database
# MAGIC
# MAGIC Now it's time to load the texts that have been generated, and create a searchable database of text for use in the `langchain` pipeline. <br>
# MAGIC These documents are embedded, so that later queries can be embedded too, and matched to relevant text chunks by embedding.
# MAGIC
# MAGIC - Collect the text chunks with Spark; `langchain` also supports reading chunks directly from Word docs, GDrive, PDFs, etc.
# MAGIC - Create a simple in-memory Chroma vector DB for storage
# MAGIC - Instantiate an embedding function from `sentence-transformers`
# MAGIC - Populate the database and save it

# COMMAND ----------

# MAGIC %md 
# MAGIC # Below Needs Updating

# COMMAND ----------

# Prepare a directory to store the document database. Any path on `/dbfs` will do.
dbutils.widgets.dropdown("reset_vector_database", "false", ["false", "true"], "Recompute embeddings for chromadb")
gardening_vector_db_path = demo_path+"/vector_db"

# Don't recompute the embeddings if the're already available
compute_embeddings = dbutils.widgets.get("reset_vector_database") == "true" or is_folder_empty(gardening_vector_db_path)

if compute_embeddings:
  print(f"creating folder {gardening_vector_db_path} under our blob storage (dbfs)")
  dbutils.fs.rm(gardening_vector_db_path, True)
  dbutils.fs.mkdirs(gardening_vector_db_path)

# COMMAND ----------

from langchain.docstore.document import Document
from langchain.vectorstores import Chroma

all_texts = spark.table("gardening_training_dataset")

print(f"Saving document embeddings under /dbfs{gardening_vector_db_path}")

if compute_embeddings: 
  # Transform our rows as langchain Documents
  # If you want to index shorter term, use the text_short field instead
  documents = [Document(page_content=r["text"], metadata={"source": r["source"]}) for r in all_texts.collect()]

  # If your texts are long, you may need to split them. However it's best to summarize them instead as show above.
  # text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=100)
  # documents = text_splitter.split_documents(documents)

  # Init the chroma db with the sentence-transformers/all-mpnet-base-v2 model loaded from hugging face  (hf_embed)
  db = Chroma.from_documents(collection_name="gardening_docs", documents=documents, embedding=hf_embed, persist_directory="/dbfs"+gardening_vector_db_path)
  db.similarity_search("dummy") # tickle it to persist metadata (?)
  db.persist()
