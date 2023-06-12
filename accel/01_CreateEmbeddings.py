# Databricks notebook source
# MAGIC %pip install -U chromadb pypdf langchain transformers accelerate bitsandbytes einops sentence_transformers PyCryptodome

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run "./utils/configs"

# COMMAND ----------

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

embeddings = HuggingFaceEmbeddings(
  model_name="sentence-transformers/all-mpnet-base-v2"
)


# COMMAND ----------

dbutils.fs.rm(dbfsnormalize(configs['chroma_persist_dir']), True)

# COMMAND ----------

# from langchain.document_loaders import PyPDFLoader
# from langchain import HuggingFaceHub
# import chromadb
# from chromadb.utils import embedding_functions
# from langchain.schema import Document

# chroma_client = chromadb.Client(settings=chromadb.config.Settings(
#                 chroma_db_impl="duckdb+parquet",
#                 persist_directory=chroma_persist_dir))
                                
# sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# try:
#   chroma_client.delete_collection('mfg_collection')
# except:
#   pass

# collection = chroma_client.get_or_create_collection(name="mfg_collection",
#                                                     embedding_function=sentence_transformer_ef)


# pathlst = dbutils.fs.ls(data_dir)
# display(pathlst)
# alldocslstloader=[]
# for idx, path1 in enumerate(pathlst):
#   if not str(path1.path).endswith('.pdf'):
#     continue
#   pdfurl = path1.path.replace(':', '')
#   pdfurl = '/' + pdfurl
#   loader = PyPDFLoader(pdfurl)
#   alldocslstloader.append(loader)
#   pdfdocs = loader.load()
#   cleanpdfdocs = []
#   for doc in pdfdocs:
#     doc.page_content = doc.page_content.replace('\n', ' ').replace('\r', ' ').replace('\t', '   ')
#     cleandoc = Document(page_content = doc.page_content, metadata=doc.metadata)
#     print(cleandoc)
#     cleanpdfdocs.append(cleandoc)
#   splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", " \n", "  \n", " ", ""], keep_separator=False)
#   texts = splitter.split_documents(cleanpdfdocs)
#   metadata_lst = []
#   ids_lst = []
#   pages_lst=[]
#   #add more metadata here
#   for idx2, docs in enumerate(texts):
#     metadata_i = {'source': pdfurl, 'source_dbfs' : path1.path}
#     metadata_lst.append(metadata_i)
#     ids_i = f'id-{idx2}-{idx+1}'
#     ids_lst.append(ids_i)
#     pages_lst.append(docs.page_content)

#   collection.add(
#           documents=pages_lst,
#           metadatas=metadata_lst,
#           ids=ids_lst
#       )

# chroma_client.persist()


# COMMAND ----------

# print(collection.count())
# collection.peek()

# COMMAND ----------

# dbutils.fs.ls('/Users/ramdas.murali@databricks.com/chromadb')

# COMMAND ----------

# results = collection.query(
#     query_texts=["What does the NIOSH do?"],
#     n_results=3
# )
# print(results)

# COMMAND ----------

from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import HuggingFaceHub
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter




embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# chromadb path
chromadb_path = configs['chroma_persist_dir']
data_dir = configs['data_dir']
# make sure chromadb path is clear
dbutils.fs.rm(dbfsnormalize(chromadb_path), recurse=True)

pathlst = dbutils.fs.ls(dbfsnormalize(data_dir))
display(pathlst)
alldocslstloader=[]
for idx, path1 in enumerate(pathlst):
  if not str(path1.path).endswith('.pdf'):
    continue
  pdfurl = path1.path.replace(':', '')
  pdfurl = '/' + pdfurl
  loader = PyPDFLoader(pdfurl)
  alldocslstloader.append(loader)
  pdfdocs = loader.load()
  cleanpdfdocs = []
  for doc in pdfdocs:
    doc.page_content = doc.page_content.replace('\n', ' ').replace('\r', ' ').replace('\t', '   ')
    cleandoc = Document(page_content = doc.page_content, metadata=doc.metadata)
    print(cleandoc)
    cleanpdfdocs.append(cleandoc)
  splitter = RecursiveCharacterTextSplitter(chunk_size=configs['chunk_size'], 
                                            chunk_overlap=configs['chunk_overlap'], 
                                            separators=["\n\n", "\n", " \n", "  \n", " ", ""], 
                                            keep_separator=False)
  texts = splitter.split_documents(cleanpdfdocs)
  metadata_lst = []
  ids_lst = []
  pages_lst=[]
  #to add more metadata here
  for idx2, docs in enumerate(texts):
    metadata_i = {'source': pdfurl, 'source_dbfs' : path1.path}
    metadata_lst.append(metadata_i)
    ids_i = f'id-{idx2}-{idx+1}'
    ids_lst.append(ids_i)
    pages_lst.append(docs.page_content)
  # define logic for embeddings storage
  vectordb = Chroma.from_texts(
    collection_name='mfg_collection',
    texts=pages_lst, 
    embedding=embedding, 
    metadatas=metadata_lst,
    ids=ids_lst,
    persist_directory=chromadb_path
    )
  # persist vector db to storage
  vectordb.persist()

# COMMAND ----------

#Test the vectorstore

vectorstore = Chroma(collection_name='mfg_collection', 
       persist_directory=chromadb_path,
       embedding_function=embedding)


# COMMAND ----------

def similarity_search(question):
  matched_docs = vectorstore.similarity_search(question, k=12)
  sources = []
  content = []
  for doc in matched_docs:
    sources.append(
        {
            "page_content": doc.page_content,
            "metadata": doc.metadata,
        }
    )
    content.append(doc.page_content)
    

  return matched_docs, sources, content


matched_docs, sources, content = similarity_search('Who provides recommendations on workspace safety')
print(content)

# COMMAND ----------

matched_docs, sources, content = similarity_search('what happens if there are hazardous substances?')
content

# COMMAND ----------


