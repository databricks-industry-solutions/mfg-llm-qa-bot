# Databricks notebook source
# MAGIC %pip install -U chromadb==0.3.22 langchain==0.0.168 transformers==4.29.0 accelerate==0.19.0 bitsandbytes tokenizers pypdf pycryptodome typing-inspect==0.8.0 typing_extensions==4.5.0

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cluster Setup
# MAGIC
# MAGIC - Run this on a cluster with Databricks Runtime 13.0 ML GPU. It should work on 12.2 ML GPU as well.
# MAGIC - To run this notebook's examples _without_ distributed Spark inference at the end, all that is needed is a single-node 'cluster' with a GPU
# MAGIC   - A10 and V100 instances should work, and this example is designed to fit the model in their working memory at some cost to quality
# MAGIC   - A100 instances work best, and perform better with minor modifications commented below
# MAGIC - To run the examples using distributed Spark inference at the end, provision a cluster of GPUs (and change the repartitioning at the end to match GPU count)
# MAGIC
# MAGIC *Note that `bitsandbytes` is not needed if running on A100s and the code is modified per comments below to not load in 8-bit.*

# COMMAND ----------

demo_path = '/FileStore/cisco/datasheets'
cisco_vector_dbpath = demo_path+"/vector_db"

# COMMAND ----------

from langchain.docstore.document import Document
from langchain.vectorstores import Chroma

from langchain.embeddings import HuggingFaceEmbeddings
 
# Download model from Hugging face
hf_embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
db = Chroma(collection_name="datasheets", embedding_function=hf_embed, persist_directory="/dbfs"+cisco_vector_dbpath)

# COMMAND ----------

print(db.get())

# COMMAND ----------

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from langchain import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.chains.question_answering import load_qa_chain
 
def build_qa_chain():
  torch.cuda.empty_cache()
  model_name = "databricks/dolly-v2-7b"  #"databricks/dolly-v2-12b" # can use dolly-v2-3b or dolly-v2-7b for smaller model and faster inferences.
 
  # Increase max_new_tokens for a longer response
  # Other settings might give better results! Play around
  instruct_pipeline = pipeline(model=model_name, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto", 
                               return_full_text=True, max_new_tokens=2048, top_p=0.95, top_k=50)
  # Note: if you use dolly 12B or smaller model but a GPU with less than 24GB RAM, use 8bit. This requires %pip install bitsandbytes
  # instruct_pipeline = pipeline(model=model_name, load_in_8bit=True, trust_remote_code=True, device_map="auto")
  # For GPUs without bfloat16 support, like the T4 or V100, use torch_dtype=torch.float16 below
  # model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
 
  # Defining our prompt content.
  # langchain will load our similar documents as {context}
  # You are a gardener and your job is to help providing the best gardening answer. 
  #Use only information in the following paragraphs to answer the question at the end. Explain the answer with reference to these paragraphs. If you don't know, say that you do not know.
  
  template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
 
  Instruction: 
  {instruction}
 
  {context}
 
  Question: {question}
 
  Response:
  """
  prompt = PromptTemplate(input_variables=['instruction', 'context', 'question'], template=template)
 
  hf_pipe = HuggingFacePipeline(pipeline=instruct_pipeline)
  # Set verbose=True to see the full prompt:
  return load_qa_chain(llm=hf_pipe, chain_type="stuff", prompt=prompt, verbose=True)
qa_chain = build_qa_chain()

# COMMAND ----------

def get_similar_docs(question, similar_doc_count):
  return db.similarity_search(question, k=similar_doc_count)
 
# Let's test it with Secure Firewall:
for doc in get_similar_docs("how does Secure firewall work?", 1000):
  print(doc.page_content)

# COMMAND ----------

def answer_question(question):
  torch.cuda.empty_cache()
  similar_docs = get_similar_docs(question, similar_doc_count=7)
  instruction = "You are a Cisco technical support engineer. Answer the question appropriately. If you dont know the answer say I dont know."
  result = qa_chain({"input_documents": similar_docs, "instruction":instruction, "question": question})
  result_html = f"<p><blockquote style=\"font-size:24\">{question}</blockquote></p>"
  result_html += f"<p><blockquote style=\"font-size:18px\">{result['output_text']}</blockquote></p>"
  result_html += "<p><hr/></p>"
  for d in result["input_documents"]:
    source_id = d.metadata["source"]
    result_html += f"<p><blockquote>{d.page_content}<br/>(Source: <a href=\"{source_id}\">{source_id}</a>)</blockquote></p>"
  displayHTML(result_html)

# COMMAND ----------

answer_question("how does Secure firewall work?")

# COMMAND ----------

answer_question("What product should I use if I want to see a list of malicious IPs?")

# COMMAND ----------

answer_question("What product should I use if I want my customer data to be protected from the public internet?")

# COMMAND ----------


