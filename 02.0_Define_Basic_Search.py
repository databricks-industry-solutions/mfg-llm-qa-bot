# Databricks notebook source
# MAGIC %md You may find this notebook on https://github.com/databricks-industry-solutions/mfg-llm-qa-bot.

# COMMAND ----------

# MAGIC %md ##Define Basic Search
# MAGIC
# MAGIC In this notebook, we will test out loading the vector database for similarity search. Additionally, we create a simple example of combining the open sourced LLM (defined in the /utils/configs) and the similarity search as a retriever. Think of this as a stand-alone implementation without any MLflow packaging
# MAGIC
# MAGIC
# MAGIC <p>
# MAGIC     <img src="https://github.com/databricks-industry-solutions/mfg-llm-qa-bot/raw/main/images/Basic-similarity-search.png" width="700" />
# MAGIC </p>
# MAGIC
# MAGIC This notebook was tested on the following infrastructure:
# MAGIC * DBR 13.3ML (GPU)
# MAGIC * g5.2xlarge(AWS) - however comparable infra on Azure should work (A10s)

# COMMAND ----------

# MAGIC %md
# MAGIC CUDA [memory management flag](https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

# COMMAND ----------

# MAGIC %sh export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'

# COMMAND ----------

# MAGIC %md
# MAGIC Install Libraries

# COMMAND ----------

# MAGIC %pip install --upgrade langchain==0.1.6 sqlalchemy==2.0.27 transformers==4.37.2 databricks-vectorsearch==0.22 mlflow[databricks] xformers==0.0.24  accelerate==0.27.0 google-search-results wikipedia

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run "./utils/configs"

# COMMAND ----------

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from torch import cuda, bfloat16,float16
import transformers
from langchain import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline
from langchain.chains import RetrievalQA
from databricks.vector_search.client import VectorSearchClient
from langchain.vectorstores import DatabricksVectorSearch
from langchain.embeddings import DatabricksEmbeddings
import torch
import gc

# COMMAND ----------

def reconvertVals(configs):
  ''' Convert string values in config to the right types'''
  for key in configs:
    if key in ['trust_remote_code', 'return_full_text', 'low_cpu_mem_usage'] and configs[key] is not None:
      configs[key] = bool(configs[key])
    if key in 'torch_dtype' and isinstance(configs['torch_dtype'], str) and configs['torch_dtype'] in 'torch.bfloat16':
      configs['torch_dtype'] = torch.bfloat16
    if key in 'torch_dtype' and isinstance(configs['torch_dtype'], str) and configs['torch_dtype'] in 'torch.float16':
      configs['torch_dtype'] = torch.float16


# COMMAND ----------

# MAGIC %md
# MAGIC The first thing we need to do is initialize a `text-generation` pipeline with Hugging Face transformers. The Pipeline requires three things that we must initialize first, those are:
# MAGIC
# MAGIC * A LLM, in this case it will be defined in the /utils/configs notebook
# MAGIC
# MAGIC * The respective tokenizer for the model.
# MAGIC
# MAGIC We'll explain these as we get to them, let's begin with our model.
# MAGIC
# MAGIC We initialize the model using the externalized configs such as automodelconfigs and pipelineconfigs
# MAGIC
# MAGIC
# MAGIC [Langchain source](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/vectorstores/databricks_vector_search.py)
# MAGIC
# MAGIC

# COMMAND ----------

#configs for the model are externalized in var automodelconfigs
try:
  device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
  reconvertVals(automodelconfigs)
  reconvertVals(pipelineconfigs)
  print(f"{configs['model_name']} using configs {automodelconfigs}")
  #account for small variations in code for loading models between models
  if 'mpt' in configs['model_name']:
    modconfig = transformers.AutoConfig.from_pretrained(configs['model_name'] ,
      trust_remote_code=True
    )
    #modconfig.attn_config['attn_impl'] = 'triton'
    model = transformers.AutoModelForCausalLM.from_pretrained(
        configs['model_name'],
        config=modconfig,
        **automodelconfigs
    )
  elif 'flan' in configs['model_name']:
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
        configs['model_name'],
        **automodelconfigs
    )
  else:
    model = transformers.AutoModelForCausalLM.from_pretrained(
        configs['model_name'],
        **automodelconfigs
    )

  #  model.to(device) -> `.to` is not supported for `4-bit` or `8-bit` models.
  listmc = automodelconfigs.keys()

  # if 'load_in_4bit' not in listmc and 'load_in_8bit' not in listmc:
  #   model.eval()
  #   model.to(device)
  
  if 'RedPajama' in configs['model_name']:
    model.tie_weights()

  print(f"Model loaded on {device}")

except Exception as e:
  print('-----')
  print(e)
  gc.collect()
  torch.cuda.empty_cache()   




# COMMAND ----------

# MAGIC %md
# MAGIC The pipeline requires a tokenizer which handles the translation of human readable plaintext to LLM readable token IDs. The Huggingface model card will give you info on the tokenizer

# COMMAND ----------

token_model= configs['tokenizer_name']
#load the tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(token_model)

# COMMAND ----------

# MAGIC %md
# MAGIC Finally we need to define the _stopping criteria_ of the model. The stopping criteria allows us to specify *when* the model should stop generating text. If we don't provide a stopping criteria the model just goes on a bit of a tangent after answering the initial question.

# COMMAND ----------

#If Stopping Criteria is needed
from transformers import StoppingCriteria, StoppingCriteriaList

# for example. mpt-7b is trained to add "<|endoftext|>" at the end of generations
stop_token_ids = tokenizer.convert_tokens_to_ids(["<|endoftext|>"])
print(stop_token_ids)
print(tokenizer.eos_token)
print(stop_token_ids)

# define custom stopping criteria object
class StopOnTokens(StoppingCriteria):
  def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
    for stop_id in stop_token_ids:
      if input_ids[0][-1] == stop_id:
        return True
    return False

stopping_criteria = StoppingCriteriaList([StopOnTokens()])

# COMMAND ----------

# device=device, -> `.to` is not supported for `4-bit` or `8-bit` models.
# accelerate lib prints this
# The model has been loaded with `accelerate` and therefore cannot be moved to a specific device. Please discard the `device` argument when creating your pipeline object.
if 'load_in_4bit' not in listmc and 'load_in_8bit' not in listmc:
  generate_text = transformers.pipeline(
      model=model, tokenizer=tokenizer,
      #device=device,
      pad_token_id=tokenizer.eos_token_id,
      #stopping_criteria=stopping_criteria,
      **pipelineconfigs
  )
else:
  generate_text = transformers.pipeline(
      model=model, tokenizer=tokenizer,
      pad_token_id=tokenizer.eos_token_id,
      #stopping_criteria=stopping_criteria,
      **pipelineconfigs
  )       




# COMMAND ----------

def get_retriever():
    '''Get the langchain vector retriever from the Databricks object '''
    vsc = VectorSearchClient(workspace_url=configs["DATABRICKS_URL"], personal_access_token=configs['DATABRICKS_TOKEN'])  
    index = vsc.get_index(endpoint_name=configs['vector_endpoint_name'], 
                          index_name=f"{configs['source_catalog']}.{configs['source_schema']}.{configs['vector_index']}")

    index.describe()
    # Create the langchain retriever. text_columns-> chunks column
    # return columns metadata_name and path along with results.
    # embedding is None for Databricks managed embedding
    vectorstore = DatabricksVectorSearch(
        index, text_column="chunks", embedding=None, columns=['metadata_name', 'path']
    )
    #filter isnt working here
    return vectorstore.as_retriever(search_kwargs={"k": configs["num_similar_docs"]}, search_type = "similarity")


# test our retriever
retriever = get_retriever()
similar_documents = retriever.get_relevant_documents("How can I contact OSHA?")
print(f"Relevant documents: {similar_documents}")

# COMMAND ----------

# MAGIC %md
# MAGIC Now we're ready to initialize the HF pipeline. There are a few additional parameters that we must define here. Comments explaining these have been included in the code.
# MAGIC The easiest way to tackle NLP tasks is to use the pipeline function. It connects a model with its necessary pre-processing and post-processing steps. This allows you to directly input any text and get an answer.
# MAGIC
# MAGIC This is the critical element to understand how the Databricks vectorstore is being passed to the QA chain as a retriever (the retrieval augmentation)
# MAGIC
# MAGIC Additional ref docs [here](https://api.python.langchain.com/en/latest/chains/langchain.chains.retrieval_qa.base.RetrievalQA.html)

# COMMAND ----------

llm = HuggingFacePipeline(pipeline=generate_text)

promptTemplate = PromptTemplate(
        template=configs['prompt_template'], input_variables=["context", "question"])
chain_type_kwargs = {"prompt":promptTemplate, "verbose":True} #change to verbose true for printing out entire prompt 

# metadata filtering logic internal implementation, if interested, in 
# def similarity_search_with_score_by_vector in
# https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/vectorstores/faiss.py

# To test metadata based filtering.
#filterdict={'Name':'ACETALDEHYDE'}
filterdict={}

#get the langchain wrapper around the databricks Vector search
retriever = get_retriever()

#retriever = vectorstore.as_retriever(search_kwargs={"k": configs['num_similar_docs'], "filter":filterdict}, search_type = "similarity")

qa_chain = RetrievalQA.from_chain_type(llm=llm, 
                                       chain_type="stuff", 
                                       retriever=retriever, 
                                       return_source_documents=False,
                                       chain_type_kwargs=chain_type_kwargs,
                                       verbose=False)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md Optionally dynamically pass a filter into the chain to pre-filter docs

# COMMAND ----------

#filterdict={'Name':'ACETALDEHYDE'} #doesnt work
print(retriever.search_kwargs)
# fetch_k Amount of documents to pass to search algorithm
#retriever.search_kwargs = {"k": 6, "filter":filterdict, "fetch_k":30}
question = {"query": "What issues can acetone exposure cause"}
answer = qa_chain.invoke(question)
print(answer)
#print(res['result'])

# COMMAND ----------

#filterdict={'Name':'ACETONE'}

# fetch_k Amount of documents to pass to search algorithm
retriever.search_kwargs = {"k": 6, "filter":filterdict, "fetch_k":30}
res = qa_chain.invoke({"query":"What issues can acetone exposure cause"})
print(res)

print(res['result'])

# COMMAND ----------

filterdict={}
retriever.search_kwargs = {"k": 6, "filter":filterdict, "fetch_k":20}
res = qa_chain.invoke({"query":"Explain to me the difference between nuclear fission and fusion."})
res

#print(res['result'])

# COMMAND ----------

filterdict={}
retriever.search_kwargs = {"k": 6, "filter":filterdict, "fetch_k":40}
res = qa_chain.invoke({'query':'what should we do if OSHA is involved?'})
res

#print(res['result'])


# COMMAND ----------

# MAGIC %md
# MAGIC Cleanup(Optional)

# COMMAND ----------

del qa_chain
del tokenizer
del model
with torch.no_grad():
    torch.cuda.empty_cache()
import gc
gc.collect()

# COMMAND ----------

import os
os.environ["SERPAPI_API_KEY"] = "0ced23b46c0376cae41078ea129e0b791c1d46d224a5e00b3aef19593e35567a"
os.environ["OPENAI_API_KEY"] = "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"

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

overall_chain({'query':'properties of acetone', 'input':'100'})

# COMMAND ----------

from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents import load_tools
from langchain.llms import OpenAI

llmai = OpenAI(temperature=0)
tools = load_tools(["wikipedia"], llm=llm)

agent = initialize_agent(tools,
                         llmai,
                         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                         verbose=False)

out = agent({'input':"what are properties of acetone", 'metadata':'ACETONE'})
print(f"{agent.agent.input_keys} {agent.agent.return_values}")
print(out)


# COMMAND ----------

from langchain.chains import TransformChain

def retrieval_transform(inputs: dict) -> dict:
    docs = retriever.get_relevant_documents(query=inputs["input"])
    docsc = [d.page_content for d in docs if inputs['metadata'] in d.metadata['metadata_name']]
    combineddocs = "\n---\n".join(docsc) + "\n--\n" + inputs['output']
    docs_dict = {
        "query": inputs["input"],
        "contexts":  combineddocs
    }
    return docs_dict

retrieval_chain = TransformChain(
    input_variables=["input", "output", "metadata"], #output from wiki chain
    output_variables=["query", "contexts"],
    transform=retrieval_transform
)

print(f"{retrieval_chain.input_keys}-{retrieval_chain.output_keys}")


out = retrieval_chain({'input':'whats the color of acetone?', 'output':'blahblahblah', 'metadata':'ACETONE'})
print(out)

# COMMAND ----------

tempstr = """Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {contexts}
    Question: {query}

    Only return the helpful answer below and nothing else.
    Helpful answer:
    """

# COMMAND ----------

promptTemplate = PromptTemplate(
        template=tempstr, input_variables=["contexts", "query"])

# COMMAND ----------

from langchain.chains import LLMChain
qa_chain = LLMChain(llm=llm, prompt=promptTemplate )
out = qa_chain({'question':'what are the properties of acetone', 'context':'Right to Know  Hazardous Substance Fact Sheet       Common Name:  ACETONE  Synonyms:  Dimethyl Ketone  Chemical Name:  2 -Propanone  Date:  February 2011         Revision:  June 2015  CAS Number:  67-64-1 RTK Substance Number:  0006  DOT Number:  UN 1090   Description and Use  Acetone  is a clear, colorles s liquid with a sweet odor.  It is used as a solvent for fats, oils, waxes, resins, plastics, and varnishes, for making other chemicals, and in nail polish remover.    ODOR THRESHOLD = 13 to 62 ppm   Odor thresholds vary greatly.  Do not rely on odor alone to determine potentially hazardous\n---\nCommon Name:  ACETONE  Synonyms:  Dimethyl Ketone; 2 -Propanone  CAS No:  67 -64-1 Molecular Formula:  C3H6O RTK Substance No:  0006  Descrip tion:  Clear, colorless liquid with a sweet odor  HAZARD DATA  Hazard Rating  Firefighting   Reactivity  1 - Health  3 - Fire 0 - Reactivity  DOT#:  UN 1090  ERG Guide #:  127 Hazard Class:  3              (Flammable)  FLAMMABLE LIQUID.  Use dry chemical, CO 2, water spray or alcohol -  resistant foam as extinguishing agents.  Water may not be effective in fighting fires.  POISONOUS GASES ARE PRODUCED IN FIRE.  CONTAINERS MAY EXPLODE IN FIRE.  Use\n---\nPeroxide  is an Organic Peroxide  and is a DANGEROUS FIRE and EXPLOSION HAZARD when exposed to HEAT, SPARKS, FLAME or CONTAMINATION.  IDENTIFICATION Acetyl Acetone Peroxide  is a colorless to light yellow liquid with a sharp smell.  Because Acetyl Acetone Peroxide  is an Organic Peroxide , it is often shipped or used in a solution or as a paste.  It is used as a catalyst to make resins, vinyl, polyolefins, and silicons.  REASON FOR CITATION * Acetyl Acetone Peroxide  is on the Hazardous Substance List because it is cited by DOT. * Definitions are provided on page 5.  HOW TO DETERMINE IF YOU\n---\nexplosive peroxides . Acet one attacks PLASTICS.   SPILL/LEAKS   PHYSICAL PROPERTIES  Isolation Distance:   Spill:  50 meters (150 feet)  Fire:  800 meters (1/2 mile)  Absorb liquids in dry sand, earth, or a similar material   and place into sealed containers for disposal.  Use only non -sparking tools and equipment.  Metal containers involving the transfer of Acetone    should be grounded and bonded.  Keep Acetone  out of confined spaces, such as   sewers, because of the possibility of an explosion.  DO NOT wash into sewer as Acetone  is dange rous to   aquatic life in high concentrations.\n---\nhazardous exposures.    Reasons for Citation   Acetone  is on the Right to Know Hazardous Substance List because it is cited by OSHA, ACGIH, DOT, NIOSH, NFPA and EPA.   This chemical is on the Special Health Hazard Substance List.         SEE GLOSSARY ON PAGE 5.  FIRST AID  Eye Contact   Immediately flush with large amounts of water for at least 15 minutes, lifting upper and lower lids.  Remove contact lenses, if worn, while rinsing.   Skin Contact   Quickly remove contaminated clothing.  Immedi ately wash contaminated skin with large amounts of soap and water.   Inhalation   Remove the\n---\nHazard Rating Key: 0=minimal; 1=slight; 2=moderate; 3=serious; 4=severe    Acetone  can affect you when inhaled and may be absorbed throug h the skin.   Acetone  can cause skin irritation. Prolonged or repeated exposure can cause drying and cracking of the skin with redness.   Exposure can irritate the eyes, nose and throat.   Exposure to high concentrations can cause headache, nausea and vomiting, dizziness, lightheadedness and even passing out.    Acetone  may affect the kidneys and liver.   Acetone  is a FLAMMABLE LIQUID and a DANGEROUS FIRE HAZARD.    Workplace Exposure Limits  OSHA:\n---\nto form explosive peroxides .  Store in tightly closed containers in a cool, well -ventilated area.   Acetone  attacks PLASTICS.   Sources of ignition, such as smoking and open flames, are prohibited where Acetone  is used, handled, or stored.   Metal containers involving the transfer of Aceto ne should be grounded and bonded.   Use explosion -proof electrical equipment and fittings wherever Acetone  is used, handled, manufactured, or stored.   Use only non -sparking tools and equipment, especially when opening and closing containers of Acetone .                 Occupational Health\n--\n'})
print(out)

# COMMAND ----------

from langchain.chains import SequentialChain
from langchain.memory import SimpleMemory
overall_chain = SequentialChain(
                input_variables=['input', 'metadata'],
                #memory=SimpleMemory(memories={"budget": "100 GBP"}),
                chains=[agent, retrieval_chain, qa_chain],
                verbose=True)
overall_chain({'input':'what are the properties of Acetone?', 'metadata':'ACETONE'})

# COMMAND ----------


