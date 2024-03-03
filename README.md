# hibo-mistral-7b-fc-v1 - dataset and fine-tuned model

<div align="center">
    <img src="./img/banner2.webp" width="100%" />
</div>

---

## 🚀 Overview

Welcome to the `thibaud-perrin/hibo-mistral-7b-fc-v1` GitHub repository! This repository is home to the fine-tuned model based on `mistralai/Mistral-7B-v0.1`, aimed at instruction following and function calling tasks. It includes Jupyter notebooks for training the model, testing its performance, and generating the dataset used during training.

## 📁 Repository Contents

- **mistral_7b_instruct.ipynb**: Notebook for fine-tuning the `Mistral-7B` model into `thibaud-perrin/hibo-mistral-7b-fc-v1`, including the model training.
- **test_hibo_mistral_7b_fc_v1.ipynb**: Notebook for testing the fine-tuned model.
- **generating_dataset.ipynb**: Notebook for generating and publishing the dataset used for training the model on Hugging Face.

## 🛠 Installation

To use the notebooks in this repository, you will need to install the necessary dependencies. You can install all required packages by running:

```bash
pipenv install
```

Ensure you have Python 3.11 or later installed on your machine.

## 🤖 Using the Model

The fine-tuned model can be accessed and used directly from Hugging Face for various NLP tasks, including instruction following and function calling. Here's a quick example using the Hugging Face Transformers library:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "thibaud-perrin/hibo-mistral-7b-fc-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

messages = [
    {"role": "system", "content": "your system prompt"},
    {"role": "user", "content": "Your prompt here"}
]
inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
outputs = model.generate(inputs["input_ids"])
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## 📘 Notebooks Description

### mistral_7b_instruct.ipynb

This notebook walks you through the process of fine-tuning the `mistralai/Mistral-7B-v0.1` model on a custom dataset for instruction following and function calling tasks. It covers everything from installing dependencies, loading datasets and training the model.

### test_mistral_7b_instruct.ipynb

Use this notebook to test the fine-tuned model's capabilities. It demonstrates how to load the model and run it on test example.

### generating_dataset.ipynb

This notebook details the process of creating the dataset that was used for training the model. It includes steps for data preparation, processing, and publishing the dataset on Hugging Face.
