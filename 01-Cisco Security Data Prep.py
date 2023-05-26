# Databricks notebook source
# MAGIC %sh ls /dbfs/FileStore/cisco/datasheets

# COMMAND ----------

# DBTITLE 1,Install our vector database
# MAGIC %pip install -U chromadb==0.3.22 langchain==0.0.168 transformers==4.29.0 accelerate==0.19.0 bitsandbytes tokenizers pypdf pycryptodome typing-inspect==0.8.0 typing_extensions==4.5.0

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1/ Downloading and extracting the raw dataset
# MAGIC
# MAGIC Will need to update this with a repeatable way of downloading data

# COMMAND ----------

demo_path = '/FileStore/cisco/datasheets'

# COMMAND ----------

# Prepare a directory to store the document database. Any path on `/dbfs` will do.
dbutils.widgets.dropdown("reset_vector_database", "false", ["false", "true"], "Recompute embeddings for chromadb")
cisco_vector_dbpath = demo_path+"/vector_db"

# COMMAND ----------

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Don't recompute the embeddings if the're already available
compute_embeddings = dbutils.widgets.get("reset_vector_database") == "true" or is_folder_empty(cisco_vector_dbpath)

if compute_embeddings:
  dbutils.fs.rm(cisco_vector_dbpath, True)
  pathlst = dbutils.fs.ls('/FileStore/cisco/datasheets')
  display(pathlst)
  alldocslst=[]
  for path1 in pathlst:
    pdfurl = path1.path.replace(':', '')
    pdfurl = '/' + pdfurl
    print(pdfurl)
    loader = PyPDFLoader(pdfurl)
    pages = loader.load_and_split()


  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size = 1000,
      chunk_overlap  = 20,
      length_function = len,
  )

  docs = text_splitter.split_documents(pages)
  alldocslst = alldocslst + docs

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2/ Load our model to transform our docs to embeddings
# MAGIC
# MAGIC We will simply load a sentence to embedding model from hugging face and use it later in the chromadb client.

# COMMAND ----------

from langchain.embeddings import HuggingFaceEmbeddings

# Download model from Hugging face
hf_embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# COMMAND ----------

from langchain.docstore.document import Document
from langchain.vectorstores import Chroma

# COMMAND ----------

#torch.cuda.empty_cache()
if compute_embeddings:
  print(f"creating folder {cisco_vector_dbpath} under our blob storage (dbfs)")
  dbutils.fs.mkdirs(cisco_vector_dbpath) 
  db = Chroma.from_documents(collection_name="datasheets", documents=alldocslst, embedding=hf_embed, persist_directory="/dbfs"+cisco_vector_dbpath)
  db.similarity_search("dummy") # tickle it to persist metadata (?)
  db.persist()  

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## That's it, our Q&A dataset is ready.
# MAGIC
# MAGIC In this notebook, we leverage Databricks to prepare our Q&A dataset:

# COMMAND ----------


