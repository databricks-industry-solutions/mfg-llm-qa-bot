# Databricks notebook source
# MAGIC %md
# MAGIC #### Open source MTP-7B model in both Hugging Face transformers and LangChain.

# COMMAND ----------

# MAGIC %pip install -U langchain==0.0.203 transformers==4.30.1 accelerate==0.20.3 einops==0.6.1 xformers==0.0.20 sentence-transformers==2.2.2 typing-inspect==0.8.0 typing_extensions==4.5.0 faiss-cpu==1.7.4 tiktoken==0.4.0

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run "./utils/configs"

# COMMAND ----------

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from torch import cuda, bfloat16,float16
import transformers
from langchain import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline
from langchain.chains import RetrievalQA

# COMMAND ----------

vector_persist_dir = configs['vector_persist_dir']
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Load from FAISS
vectorstore = FAISS.load_local(vector_persist_dir, embeddings)

def similarity_search(question, filter={}, fetch_k=100, k=12):
  matched_docs = vectorstore.similarity_search(question, filter=filter, fetch_k=fetch_k, k=k)
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


matched_docs, sources, content = similarity_search('Who provides recommendations on workspace safety on Acetone', {'Name':'ACETONE'})
print(content)
print(matched_docs)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initializing the Hugging Face Pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC The first thing we need to do is initialize a `text-generation` pipeline with Hugging Face transformers. The Pipeline requires three things that we must initialize first, those are:
# MAGIC
# MAGIC * A LLM, in this case it will be `mosaicml/mpt-7b-instruct`.
# MAGIC
# MAGIC * The respective tokenizer for the model.
# MAGIC
# MAGIC * ~A stopping criteria object.~
# MAGIC
# MAGIC We'll explain these as we get to them, let's begin with our model.
# MAGIC
# MAGIC We initialize the model and move it to our CUDA-enabled GPU. Using Colab this can take 5-10 minutes to download and initialize the model.

# COMMAND ----------

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

print(f"{configs['model_name']} using configs {automodelconfigs}")

model = transformers.AutoModelForCausalLM.from_pretrained(
    configs['model_name'],
    **automodelconfigs
)

#  model.to(device) -> `.to` is not supported for `4-bit` or `8-bit` models.
model.eval()
model.to(device)
if 'RedPajama' in configs['model_name']:
  model.tie_weights()

print(f"Model loaded on {device}")

# COMMAND ----------

# MAGIC %md
# MAGIC The pipeline requires a tokenizer which handles the translation of human readable plaintext to LLM readable token IDs. The MPT-7B model was trained using the `EleutherAI/gpt-neox-20b` tokenizer, which we initialize like so:

# COMMAND ----------

token_model= configs['tokenizer_name']
tokenizer = transformers.AutoTokenizer.from_pretrained(token_model)

# COMMAND ----------

# MAGIC %md
# MAGIC Finally we need to define the _stopping criteria_ of the model. The stopping criteria allows us to specify *when* the model should stop generating text. If we don't provide a stopping criteria the model just goes on a bit of a tangent after answering the initial question.

# COMMAND ----------

#If Stopping Criteria is needed
# from transformers import StoppingCriteria, StoppingCriteriaList

# # mtp-7b is trained to add "<|endoftext|>" at the end of generations
# stop_token_ids = tokenizer.convert_tokens_to_ids(["<|endoftext|>"])

# # define custom stopping criteria object
# class StopOnTokens(StoppingCriteria):
#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
#         for stop_id in stop_token_ids:
#             if input_ids[0][-1] == stop_id:
#                 return True
#         return False

# stopping_criteria = StoppingCriteriaList([StopOnTokens()])

# COMMAND ----------

# MAGIC %md
# MAGIC Now we're ready to initialize the HF pipeline. There are a few additional parameters that we must define here. Comments explaining these have been included in the code.

# COMMAND ----------

# device=device, -> `.to` is not supported for `4-bit` or `8-bit` models.
generate_text = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    device=device,
    pad_token_id=tokenizer.eos_token_id,
    **pipelineconfigs
)

# COMMAND ----------

llm = HuggingFacePipeline(pipeline=generate_text)

promptTemplate = PromptTemplate(
        template=configs['prompt_template'], input_variables=["context", "question"])
chain_type_kwargs = {"prompt":promptTemplate}

#filterdict={'Name':'ACETALDEHYDE'}
filterdict={}
retriever = vectorstore.as_retriever(search_kwargs={"k": configs['num_similar_docs'], "filter":filterdict}, search_type = "similarity")

qa_chain = RetrievalQA.from_chain_type(llm=llm, 
                                       chain_type="stuff", 
                                       retriever=retriever, 
                                       return_source_documents=True,
                                       chain_type_kwargs=chain_type_kwargs,
                                       verbose=False)

# COMMAND ----------

filterdict={'Name':'ACETONE'}
retriever.search_kwargs = {"k": 10, "filter":filterdict, "fetch_k":100}
res = qa_chain({"query":"What happens with acetaldehyde exposure"})
print(res)

print(res['result'])

# COMMAND ----------

# MAGIC %md
# MAGIC Confirm this is working:

# COMMAND ----------

filterdict={}
retriever.search_kwargs = {"k": 10, "filter":filterdict, "fetch_k":100}
res = qa_chain({"query":"Explain to me the difference between nuclear fission and fusion."})
res

# COMMAND ----------

filterdict={}
retriever.search_kwargs = {"k": 10, "filter":filterdict, "fetch_k":100}
res = qa_chain({'query':'what should we do if OSHA is involved?'})
res



# COMMAND ----------

# MAGIC %md
# MAGIC We still get the same output as we're not really doing anything differently here, but we have now added MTP-7B-instruct to the LangChain library. Using this we can now begin using LangChain's advanced agent tooling, chains, etc, with MTP-7B.

# COMMAND ----------

# del qa_chain
# del tokenizer
# del model
# with torch.no_grad():
#     torch.cuda.empty_cache()
# import gc
# gc.collect()
