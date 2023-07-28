![image](https://raw.githubusercontent.com/databricks-industry-solutions/.github/main/profile/solacc_logo_wide.png)

[![CLOUD](https://img.shields.io/badge/CLOUD-ALL-blue?logo=googlecloud&style=for-the-badge)](https://cloud.google.com/databricks)
[![POC](https://img.shields.io/badge/POC-10_days-green?style=for-the-badge)](https://databricks.com/try-databricks)

## Business Problem
The goal of this solution accelerator is to demonstrate how you can create a QA bot using open sourced models. Due to privacy requirements, many customers can't use propreitary models. This has been designed so that you can easily switch between the newest and best LLMs from Huggingface. 

## Scope
A customer can easily take this solution accelerator and replace it with a knowledge base of articles (i.e. in PDF format) to replicate the application that's been built. 

___
<william.block@databricks.com>,
<ramdas.murali@databricks.com>,
<bala.amavasai@databricks.com>
___


![image](/images/Entire-process.png)

___

&copy; 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.

| library                                | description             | license    | source                                              |
|----------------------------------------|-------------------------|------------|-----------------------------------------------------|
| Langchain                              | Develop LLM applications  | MIT        | https://pypi.org/project/langchain              |
| Huggingface                                 | Huggingface is a hub LLM apps      | Apache Software License (Apache 2.0)        | https://pypi.org/project/huggingface/            |

## Getting started

Although specific solutions can be downloaded as .dbc archives from our websites, we recommend cloning these repositories onto your databricks environment. Not only will you get access to latest code, but you will be part of a community of experts driving industry best practices and re-usable solutions, influencing our respective industries. 

<img width="500" alt="add_repo" src="https://user-images.githubusercontent.com/4445837/177207338-65135b10-8ccc-4d17-be21-09416c861a76.png">

To start using a solution accelerator in Databricks simply follow these steps: 

1. Clone solution accelerator repository in Databricks using [Databricks Repos](https://www.databricks.com/product/repos)
2. Navigate to the /accel folder. The notebooks are organized in this folder for your convenience.
3. Within /accel/utils, you can edit the types of models you would like to test out from Huggingface. Additionally, you can change the configs here to point to your knowledge base to create your own vector database.

The cost associated with running the accelerator is the user's responsibility.


## Project support 

Please note the code in this project is provided for your exploration only, and are not formally supported by Databricks with Service Level Agreements (SLAs). They are provided AS-IS and we do not make any guarantees of any kind. Please do not submit a support ticket relating to any issues arising from the use of these projects. The source in this project is provided subject to the Databricks [License](./LICENSE). All included or referenced third party libraries are subject to the licenses set forth below.

Any issues discovered through the use of this project should be filed as GitHub Issues on the Repo. They will be reviewed as time permits, but there are no formal SLAs for support. 
