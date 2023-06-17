# Databricks notebook source
# MAGIC %pip install -U PyPDF==3.9.1 pycryptodome==3.18.0 langchain==0.0.197 transformers==4.30.1 accelerate==0.20.3  einops==0.6.1 xformers==0.0.20 sentence-transformers==2.2.2 PyCryptodome==3.18.0 typing-inspect==0.8.0 typing_extensions==4.5.0 faiss-cpu==1.7.4 tiktoken==0.4.0
# MAGIC

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run "./utils/configs"

# COMMAND ----------

dbutils.fs.rm(dbfsnormalize(configs['vector_persist_dir']), True)

# COMMAND ----------

# dbutils.fs.ls('/Users/ramdas.murali@databricks.com/chromadb')

# COMMAND ----------

from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import HuggingFaceHub
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# vectordb path
vectordb_path = configs['vector_persist_dir']
data_dir = configs['data_dir']
# make sure vectordb path is clear
dbutils.fs.rm(dbfsnormalize(vectordb_path), recurse=True)

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

  # For Chroma
  # vectordb = Chroma.from_texts(
  #   collection_name='mfg_collection',
  #   texts=pages_lst, 
  #   embedding=embeddings, 
  #   metadatas=metadata_lst,
  #   ids=ids_lst,
  #   persist_directory=vectordb_path
  #   )
  # # persist vector db to storage
  # vectordb.persist()
  
  # For FAISS
  vectordb = FAISS.from_texts(pages_lst, embeddings, metadatas=metadata_lst, ids=ids_lst)
  vectordb.save_local(vectordb_path)

# COMMAND ----------

#Test the vectorstore

# Load from Chroma
# vectorstore = Chroma(collection_name='mfg_collection', 
#        persist_directory=vectordb_path,
#        embedding_function=embeddings)


# Load from FAISS
vectorstore = FAISS.load_local(vectordb_path, embeddings)


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

matched_docs
