import os
import requests
import numpy as np
import pandas as pd
import json
import gradio as gr

endpoint_url = f"""{os.environ['DATABRICKS_HOST']}/serving-endpoints/llm-qabot-endpoint/invocations"""

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


def greet(question):
    assemble_question = pd.DataFrame({'question':[
  f'{question}'
]})
    data = score_model(assemble_question)
    answer = data["predictions"][0]["answer"]

    return answer

demo = gr.Interface(
    fn=greet, 
    inputs="text", 
    outputs="text")

demo.launch()  

