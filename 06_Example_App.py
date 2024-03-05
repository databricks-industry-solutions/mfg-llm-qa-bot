# Databricks notebook source
# MAGIC %md You may find this notebook on https://github.com/databricks-industry-solutions/mfg-llm-qa-bot.

# COMMAND ----------

# MAGIC %md ##Example Application
# MAGIC
# MAGIC This is an example application that you can leverage to make an api call to the model that's now hosted in Databricks model serving. This application can be run from this notebook. For more info on gradio, visit https://www.gradio.app/guides/quickstart
# MAGIC
# MAGIC
# MAGIC <p>
# MAGIC     <img src="https://github.com/databricks-industry-solutions/mfg-llm-qa-bot/raw/main/images/Example-App.png" width="700" />
# MAGIC </p>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Tested on
# MAGIC
# MAGIC * Single Node Cluster i3.xlarge
# MAGIC * 13.3 LTS

# COMMAND ----------

# MAGIC %pip uninstall --yes typing_extensions

# COMMAND ----------

# MAGIC %pip install gradio typing_extensions==4.9.0 mlflow

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./utils/configs

# COMMAND ----------

os.environ['DATABRICKS_URL']=configs["DATABRICKS_URL"]
os.environ['DATABRICKS_TOKEN']=configs["DATABRICKS_TOKEN"]

# COMMAND ----------

# MAGIC %md
# MAGIC Running this will start the form right within this notebook.
# MAGIC
# MAGIC Samples
# MAGIC * compare properties of acetone and acetaldehyde. Show me the response in a table
# MAGIC * What are the exposure limits for acetyl methyl carbinol?
# MAGIC * When should OSHA get involved in a chemical event? what is their contact?
# MAGIC *
# MAGIC
# MAGIC You can also see the public URL where you can run this from a browser
# MAGIC
# MAGIC ```Running on public URL: https://some_generated_url.gradio.live```

# COMMAND ----------

import os
import requests
import numpy as np
import pandas as pd
import json
import gradio as gr

endpoint=configs['serving_endpoint_name']
endpoint_url = f"""{os.environ['DATABRICKS_URL']}/serving-endpoints/{endpoint}/invocations"""


def create_tf_serving_json(data):
    return {
        "inputs": {name: data[name].tolist() for name in data.keys()}
        if isinstance(data, dict)
        else data.tolist()
    }


def score_model(dataset):
    url = endpoint_url
    headers = {
        "Authorization": f'Bearer {os.environ["DATABRICKS_TOKEN"]}',
        "Content-Type": "application/json",
    }

    ds_dict = (
        {"dataframe_split": dataset.to_dict(orient="split")}
        if isinstance(dataset, pd.DataFrame)
        else create_tf_serving_json(dataset)
    )
    data_json = json.dumps(ds_dict, allow_nan=True)

    response = requests.request(method="POST", headers=headers, url=url, data=data_json)
    if response.status_code != 200:
        raise Exception(
            f"Request failed with status {response.status_code}, {response.text}"
        )

    return response.json()


def callLLM(question, filter):
    filterdict={}
    if not filter.strip() == '':
        filterdict={'Name':f'{filter}'}
    dict = {'questions':[f'{question}'], 'filter':[filterdict]}
    assemble_question = pd.DataFrame.from_dict(dict)
    data = score_model(assemble_question)
    answer = data["predictions"]["answer"]
    source = data["predictions"]["source"]
    source = source.replace(',', '\n')
    return [answer, source]


with gr.Blocks( theme=gr.themes.Soft()) as demo:
    with gr.Row():
        gr.HTML(show_label=False, value="<img src='https://databricks.gallerycdn.vsassets.io/extensions/databricks/databricks/0.3.15/1686753455931/Microsoft.VisualStudio.Services.Icons.Default' height='30' width='30'/><div font size='1'>Manufacturing</div>")
    with gr.Row():    
        gr.Markdown(
                """
            # Chemical Industry Q&A Bot
            This bot has been trained on chemical fact sheets from https://web.doh.state.nj.us/rtkhsfs/factsheets.aspx. For the purposes of this demo, we have only downloaded the chemicals that start with A. The fact sheets were transformed into embeddings and are used as a retriever for the model. Langchain was then used to compile the model, which is then hosted on Databricks MLflow. The application simply makes an API call to the model that's hosted in Databricks.
            """
            )
    with gr.Row():
        input = gr.Textbox(placeholder="ex. What should I do if I spill acetone? or What happens if arsenic gets on my skin?", label="Question")
        inputfilter = gr.Textbox(placeholder="ACETONE", label="Filter (Optional)")
    with gr.Row():
        output = gr.Textbox(label="Prediction")
        callLLM_btn = gr.Button("Respond", size="sm", scale=0)
    with gr.Row():
        outputsrc = gr.Textbox(label="Chunks from Sources", visible=True, lines=5, max_lines=10)

    callLLM_btn.click(fn=callLLM, inputs=[input, inputfilter], outputs=[output, outputsrc], api_name="callLLM")
    
demo.launch(share=True)  


# COMMAND ----------

# MAGIC %md 
# MAGIC To terminate the running gradio process that hosts the web application. Comment this out if you are going to do a "Run All".

# COMMAND ----------

# MAGIC %sh
# MAGIC ps -eaf | awk '/gradio/  {print $2}' | head -1 | xargs kill -9 

# COMMAND ----------

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC |  Gradio | Build Machine Learning Web Apps in Python |  Apache Software License  |   https://pypi.org/project/gradio/ |

# COMMAND ----------


