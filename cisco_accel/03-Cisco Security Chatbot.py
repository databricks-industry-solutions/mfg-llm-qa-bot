# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC # Chat Bot with langchain and Dolly
# MAGIC
# MAGIC ## Chat Bot Prompt engineering
# MAGIC
# MAGIC In this example, we will improve our previous Q&A to create a chat bot.
# MAGIC
# MAGIC The main thing we'll be adding is a memory between the different question so that our bot can answer having the context of the previous Q&A.
# MAGIC
# MAGIC
# MAGIC <img style="float:right" width="800px" src="https://raw.githubusercontent.com/databricks-demos/dbdemos-resources/main/images/product/llm-dolly/llm-dolly-chatbot.png">
# MAGIC
# MAGIC ### Keeping memory between multiple questions
# MAGIC
# MAGIC The main challenge for our chat bot is that we won't be able to use the entire discussion history as context to send to dolly. 
# MAGIC
# MAGIC First of all this is expensive, but more importantly this won't support long discussion as we'll endup with a text longer than our max window size for our mdoel.
# MAGIC
# MAGIC The trick is to use a summarize model and add an intermediate step which will take the summary of our discussion and inject it in our prompt.
# MAGIC
# MAGIC We will use an intermediate summarization task to do that, using `ConversationSummaryMemory` from `langchain`.
# MAGIC
# MAGIC
# MAGIC **Note: This is a more advanced content, we recommend you start with the Previous notebook**

# COMMAND ----------

# MAGIC %pip install -U transformers langchain chromadb accelerate bitsandbytes

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

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM
from langchain import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationSummaryBufferMemory

def build_qa_chain():
  torch.cuda.empty_cache()
  # Defining our prompt content.
  # langchain will load our similar documents as {context}
  template = """You are a chatbot having a conversation with a human. Your are asked to answer Cisco Security Product questions.
  Given the following extracted parts of a long document and a question, answer the user question. If you don't know, say that you do not know. 
  
  {context}

  {chat_history}

  {human_input}

  Response:
  """
  prompt = PromptTemplate(input_variables=['context', 'human_input', 'chat_history'], template=template)

  # Increase max_new_tokens for a longer response
  # Other settings might give better results! Play around
  model_name = "databricks/dolly-v2-7b" # can use dolly-v2-3b, dolly-v2-7b or dolly-v2-12b for smaller model and faster inferences.
  instruct_pipeline = pipeline(model=model_name, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto", 
                               return_full_text=True, max_new_tokens=256, top_p=0.95, top_k=50)
  hf_pipe = HuggingFacePipeline(pipeline=instruct_pipeline)

  # Add a summarizer to our memory conversation
  # Let's make sure we don't summarize the discussion too much to avoid losing to much of the content

  # Models we'll use to summarize our chat history
  # We could use one of these models: https://huggingface.co/models?filter=summarization. facebook/bart-large-cnn gives great results, we'll use t5-small for memory
  summarize_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small", device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
  summarize_tokenizer = AutoTokenizer.from_pretrained("t5-small", padding_side="left", model_max_length = 512)
  pipe_summary = pipeline("summarization", model=summarize_model, tokenizer=summarize_tokenizer) #, max_new_tokens=500, min_new_tokens=300
  # langchain pipeline doesn't support summarization yet, we added it as temp fix in the companion notebook _resources/00-init 
  hf_summary = HuggingFacePipeline_WithSummarization(pipeline=pipe_summary)
  #will keep 500 token and then ask for a summary. Removes prefix as our model isn't trained on specific chat prefix and can get confused.
  memory = ConversationSummaryBufferMemory(llm=hf_summary, memory_key="chat_history", input_key="human_input", max_token_limit=500, human_prefix = "", ai_prefix = "")

  # Set verbose=True to see the full prompt:
  print("loading chain, this can take some time...")
  return load_qa_chain(llm=hf_pipe, chain_type="stuff", prompt=prompt, verbose=True, memory=memory)

# COMMAND ----------

class ChatBot():
  def __init__(self, db):
    self.reset_context()
    self.db = db

  def reset_context(self):
    self.sources = []
    self.discussion = []
    # Building the chain will load Dolly and can take some time depending on the model size and your GPU
    self.qa_chain = build_qa_chain()
    displayHTML("<h1>Hi! I'm a chat bot specialized in Cisco. How Can I help you today?</h1>")

  def get_similar_docs(self, question, similar_doc_count):
    return self.db.similarity_search(question, k=similar_doc_count)

  def chat(self, question):
    # Keep the last 3 discussion to search similar content
    self.discussion.append(question)
    similar_docs = self.get_similar_docs(" \n".join(self.discussion[-3:]), similar_doc_count=2)
    # Remove similar doc if they're already in the last questions (as it's already in the history)
    similar_docs = [doc for doc in similar_docs if doc.metadata['source'] not in self.sources[-3:]]

    result = self.qa_chain({"input_documents": similar_docs, "human_input": question})
    # Cleanup the answer for better display:
    answer = result['output_text'].capitalize()
    result_html = f"<p><blockquote style=\"font-size:24\">{question}</blockquote></p>"
    result_html += f"<p><blockquote style=\"font-size:18px\">{answer}</blockquote></p>"
    result_html += "<p><hr/></p>"
    for d in result["input_documents"]:
      source_id = d.metadata["source"]
      self.sources.append(source_id)
      result_html += f"<p><blockquote>{d.page_content}<br/>(Source: <a href=\"{source_id}\">{source_id}</a>)</blockquote></p>"
    displayHTML(result_html)

chat_bot = ChatBot(db)

# COMMAND ----------

chat_bot.chat("What product should I use if I want to see a list of malicious IPs?")

# COMMAND ----------


