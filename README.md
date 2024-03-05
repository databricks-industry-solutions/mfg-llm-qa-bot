![image](https://raw.githubusercontent.com/databricks-industry-solutions/.github/main/profile/solacc_logo_wide.png)

[![CLOUD](https://img.shields.io/badge/CLOUD-ALL-blue?logo=googlecloud&style=for-the-badge)](https://cloud.google.com/databricks)
[![POC](https://img.shields.io/badge/POC-10_days-green?style=for-the-badge)](https://databricks.com/try-databricks)

## Business Problem
The goal of this solution accelerator is to demonstrate how you can create a QA bot using open sourced models. Due to privacy requirements, many customers can't use proprietary models. This has been designed so that you can easily switch between the newest and best LLMs from Huggingface. 

## Scope
A customer can easily take this solution accelerator and replace it with a knowledge base of articles (i.e. in PDF format) to replicate the application that's been built. 

## New Features (Mar 2024)
* [Databricks Vector search](https://docs.databricks.com/en/generative-ai/vector-search.html) (previously FAISS)
* Different model types
  * [Foundational Model](https://docs.databricks.com/en/machine-learning/foundation-models/index.html) (Notebooks 2.1->3.1->4.1)
  * [External Model](https://docs.databricks.com/en/generative-ai/external-models/index.html) (Notebooks 2.1->3.1->4.1)
  * [Custom PyFunc model](https://mlflow.org/docs/latest/llms/custom-pyfunc-for-llms/notebooks/custom-pyfunc-advanced-llm.html) (Notebooks 2.3->3.3->4.3)
  * Compound AI models (with [Langchain Agents](https://www.langchain.com/agents) using Wikipedia) (Notebooks 2.2)
* [UC Model registry](https://docs.databricks.com/en/machine-learning/manage-model-lifecycle/upgrade-models.html) (Notebooks 5)
* Self hosted Gradio app (previously on huggingface.co/spaces) (Notebook 6)




___
- <ramdas.murali@databricks.com>
- <william.block@databricks.com>,
- Testing - <veronica.gomes@databricks.com>
- Thanks to Bala Amavasai for his valuable assistance

___


![image](https://github.com/databricks-industry-solutions/mfg-llm-qa-bot/raw/main/images/EntireProcess.png)

___

&copy; 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.

| library                                | description             | license    | source                                              |
|----------------------------------------|-------------------------|------------|-----------------------------------------------------|
| Langchain                              | Develop LLM applications  | MIT        | https://pypi.org/project/langchain              |
| Huggingface                                 | Huggingface is a hub LLM apps      | Apache 2.0        | https://pypi.org/project/huggingface/            |
|  Gradio | Build Machine Learning Web Apps in Python |  Apache 2.0  |   https://pypi.org/project/gradio/ |

## Getting started

Although specific solutions can be downloaded from our websites, we recommend cloning these repositories onto your databricks environment. Not only will you get access to latest code, but you will be part of a community of experts driving industry best practices and re-usable solutions, influencing our respective industries. 

<img width="500" alt="add_repo" src="https://user-images.githubusercontent.com/4445837/177207338-65135b10-8ccc-4d17-be21-09416c861a76.png">

To start using a solution accelerator in Databricks simply follow these steps: 

1. Clone solution accelerator repository in Databricks using [Databricks Repos](https://www.databricks.com/product/repos)
2. Attach the RUNME notebook to any cluster running a DBR 11.0 or later runtime, and execute the notebook via Run-All. A multi-step-job describing the accelerator pipeline will be created, and the link will be provided. Execute the multi-step-job to see how the pipeline runs.
3. Within /utils, you can edit the types of models you would like to test out from Huggingface. You will want to follow the instructions there to set up your huggingface token. Additionally, you can change the configs here to point to your knowledge base to create your own vector database.

The cost associated with running the accelerator is the user's responsibility.


## Project support 

Please note the code in this project is provided for your exploration only, and are not formally supported by Databricks with Service Level Agreements (SLAs). They are provided AS-IS and we do not make any guarantees of any kind. Please do not submit a support ticket relating to any issues arising from the use of these projects. The source in this project is provided subject to the Databricks [License](./LICENSE). All included or referenced third party libraries are subject to the licenses set forth below.

Any issues discovered through the use of this project should be filed as GitHub Issues on the Repo. They will be reviewed as time permits, but there are no formal SLAs for support. 
