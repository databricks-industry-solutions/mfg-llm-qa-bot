# Databricks notebook source
# MAGIC %md
# MAGIC #### Open source MTP-7B model in both Hugging Face transformers and LangChain.

# COMMAND ----------

# MAGIC %pip install -U chromadb==0.3.26 langchain==0.0.197 transformers==4.30.1 accelerate==0.20.3 bitsandbytes==0.39.0 einops==0.6.1 xformers==0.0.20

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run "./utils/configs"

# COMMAND ----------

chroma_persist_dir = configs['chroma_persist_dir']
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceHubEmbeddings


vectorstore = Chroma(
        collection_name="mfg_collection",
        persist_directory=chroma_persist_dir,
        embedding_function=HuggingFaceHubEmbeddings(repo_id='sentence-transformers/all-MiniLM-L6-v2')
)


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

from langchain.prompts import PromptTemplate

def getPromptTemplate():
  prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

  {context}

  Question: {question}"""

  PROMPT = PromptTemplate(
      template=prompt_template, input_variables=["context", "question"]
  )

  return PROMPT

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

from torch import cuda, bfloat16,float16
import transformers

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

model = transformers.AutoModelForCausalLM.from_pretrained(
    'mosaicml/mpt-7b-instruct',
    trust_remote_code=True,
    device_map='auto', torch_dtype=float16, load_in_8bit=True, #rkm testing
    #torch_dtype=bfloat16,
    max_seq_len=1440
)
model.eval()
#model.to(device)
print(f"Model loaded on {device}")

# COMMAND ----------

# MAGIC %md
# MAGIC The pipeline requires a tokenizer which handles the translation of human readable plaintext to LLM readable token IDs. The MPT-7B model was trained using the `EleutherAI/gpt-neox-20b` tokenizer, which we initialize like so:

# COMMAND ----------

tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

# COMMAND ----------

# MAGIC %md
# MAGIC Finally we need to define the _stopping criteria_ of the model. The stopping criteria allows us to specify *when* the model should stop generating text. If we don't provide a stopping criteria the model just goes on a bit of a tangent after answering the initial question.

# COMMAND ----------

#not Used

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

# MAGIC %md
# MAGIC Now we're ready to initialize the HF pipeline. There are a few additional parameters that we must define here. Comments explaining these have been included in the code.

# COMMAND ----------

generate_text = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    #device=device,
    # we pass model parameters here too
    stopping_criteria=stopping_criteria,  # without this model will ramble
    temperature=configs['temperature'],  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    top_p=0.80,  # select from top tokens whose probability add up to 80%
    top_k=0,  # select from top 0 tokens (because zero, relies on top_p)
    max_new_tokens=configs['max_new_tokens'],  # mex number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)

# COMMAND ----------

from langchain import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline
from langchain.chains import RetrievalQAWithSourcesChain, RetrievalQA
from langchain import LLMChain

chain_type_kwargs = {"prompt":getPromptTemplate()}

llm = HuggingFacePipeline(pipeline=generate_text)

retriever = vectorstore.as_retriever(search_kwargs={"k": configs['num_similar_docs']}) #, "search_type" : "similarity"

qa_chain = RetrievalQA.from_chain_type(llm=llm, 
                                       chain_type="stuff", 
                                       retriever=retriever, 
                                       return_source_documents=True,
                                       chain_type_kwargs=chain_type_kwargs,
                                       verbose=False)

# COMMAND ----------

res = qa_chain({"query":"When is medical attention needed"})
print(res)

print(res['result'])

# COMMAND ----------

# MAGIC %md
# MAGIC Confirm this is working:

# COMMAND ----------

res = qa_chain({"query":"Explain to me the difference between nuclear fission and fusion."})
res

# COMMAND ----------

res = qa_chain({'query':'what should we do if OSHA is involved?'})
res



# COMMAND ----------

# MAGIC %md
# MAGIC We still get the same output as we're not really doing anything differently here, but we have now added MTP-7B-instruct to the LangChain library. Using this we can now begin using LangChain's advanced agent tooling, chains, etc, with MTP-7B.

# COMMAND ----------

del qa_chain
del tokenizer
del model
cuda.empty_cache()


# COMMAND ----------

import gc
gc.collect()

# COMMAND ----------

with torch.no_grad():
    torch.cuda.empty_cache()

# COMMAND ----------


