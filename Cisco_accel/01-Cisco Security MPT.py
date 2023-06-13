# Databricks notebook source
!wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcusparse-dev-11-7_11.7.3.50-1_amd64.deb -O /tmp/libcusparse-dev-11-7_11.7.3.50-1_amd64.deb && \
  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcublas-dev-11-7_11.10.1.25-1_amd64.deb -O /tmp/libcublas-dev-11-7_11.10.1.25-1_amd64.deb && \
  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcusolver-dev-11-7_11.4.0.1-1_amd64.deb -O /tmp/libcusolver-dev-11-7_11.4.0.1-1_amd64.deb && \
  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcurand-dev-11-7_10.2.10.91-1_amd64.deb -O /tmp/libcurand-dev-11-7_10.2.10.91-1_amd64.deb && \
  dpkg -i /tmp/libcusparse-dev-11-7_11.7.3.50-1_amd64.deb && \
  dpkg -i /tmp/libcublas-dev-11-7_11.10.1.25-1_amd64.deb && \
  dpkg -i /tmp/libcusolver-dev-11-7_11.4.0.1-1_amd64.deb && \
  dpkg -i /tmp/libcurand-dev-11-7_10.2.10.91-1_amd64.deb

# COMMAND ----------

# DBTITLE 1,Install our vector database
# MAGIC %pip install -U chromadb==0.3.22 langchain==0.0.168 transformers==4.29.0 accelerate==0.19.0 bitsandbytes tokenizers pycryptodome typing-inspect==0.8.0 typing_extensions==4.5.0 einops xformers flash-attn pypdf

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

# MAGIC %md
# MAGIC ##3/ Download model and store on device

# COMMAND ----------

from torch import cuda, bfloat16
import transformers


device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

model = transformers.AutoModelForCausalLM.from_pretrained(
    'mosaicml/mpt-7b-instruct',
    trust_remote_code=True,
    torch_dtype=bfloat16,
    max_seq_len=2048
)
model.eval()
model.to(device)
print(f"Model loaded on {device}")

# COMMAND ----------

tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b") # mosaicml/mpt-7b-instruct

# COMMAND ----------

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

# mtp-7b is trained to add "<|endoftext|>" at the end of generations
stop_token_ids = tokenizer.convert_tokens_to_ids(["<|endoftext|>"])

# define custom stopping criteria object
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

stopping_criteria = StoppingCriteriaList([StopOnTokens()])

# COMMAND ----------

generate_text = transformers.pipeline(
    model=model, 
    tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    device=device,
    # we pass model parameters here too
    stopping_criteria=stopping_criteria,  # without this model will ramble
    temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    top_p=0.15,  # select from top tokens whose probability add up to 15%
    top_k=0,  # select from top 0 tokens (because zero, relies on top_p)
    max_new_tokens=64,  # mex number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)

# COMMAND ----------

from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline

# template for an instrution with no input
prompt = PromptTemplate(
    input_variables=["instruction"],
    template="{instruction}")

# template for an instruction with input
prompt_with_context = PromptTemplate(
    input_variables=["instruction", "context"],
    template="{instruction}\n\nInput:\n{context}")

hf_pipeline = HuggingFacePipeline(pipeline=generate_text)

llm_chain = LLMChain(llm=hf_pipeline, prompt=prompt)

# COMMAND ----------

print(llm_chain.predict(instruction="Explain to me the difference between nuclear fission and fusion.").lstrip())

# COMMAND ----------

# MAGIC %md
# MAGIC # Load in vector store and use for retrieval

# COMMAND ----------

from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma

db = Chroma(collection_name="datasheets", embedding_function=hf_embed, persist_directory="/dbfs"+cisco_vector_dbpath)
retriever = db.as_retriever(search_kwargs={"k": 3})

qa_chain = RetrievalQA.from_chain_type(llm=hf_pipeline, 
                                  chain_type="stuff", 
                                  retriever=retriever)

# COMMAND ----------

def get_similar_docs(question, similar_doc_count):
  return db.similarity_search(question, k=similar_doc_count)

# Let's test it with Secure Firewall:
for doc in get_similar_docs("how does Cisco Secure firewall work?", 5):
  print(doc)
  #print(doc.page_content)

# COMMAND ----------

def answer_question(question):
  #torch.cuda.empty_cache()
  #similar_docs = get_similar_docs(question, similar_doc_count=5)
  result = qa_chain(question)
  result_html = f"<p><blockquote style=\"font-size:24\">{question}</blockquote></p>"
  result_html += f"<p><blockquote style=\"font-size:18px\">{result['result']}</blockquote></p>"
  result_html += "<p><hr/></p>"
  for d in get_similar_docs(question, similar_doc_count=5):
    source_id = d.metadata["source"]
    result_html += f"<p><blockquote>{d.page_content}<br/>(Source: <a href=\"{source_id}\">{source_id}</a>)</blockquote></p>"
  displayHTML(result_html)

# COMMAND ----------

answer_question("What Cisco product should we use for protecting our iPhones?")

# COMMAND ----------

answer_question("Can you describe to me how Cisco Secure Firewall works?")

# COMMAND ----------

answer_question("When should I use Secure DDoS Protection?")

# COMMAND ----------


