# GenAI-RAG-Agent

Welcome to GenAI-RAG Agent, your cutting-edge virtual Agent powered by Generative AI (GenAI) models and the innovative Retrieval-Augmented Generation (RAG) approach!

## About

GenAI-RAG Agent is designed to be your ultimate companion, providing precise and relevant answers to your queries through state-of-the-art AI technology. With the integration of the RAG model, GenAI-RAG Agent ensures that you receive accurate information tailored to your needs.

## Features

- **Advanced AI Capabilities:** Leveraging **Generative AI (GenAI)** models for intelligent responses.
- **Retrieval-Augmented Generation (RAG):** Incorporating the RAG model for precise and relevant answers. Used **Pinecone** for **Vector Database**.
- **Web API:** Simple **FastAPI** interface for user-agent interaction.

## Getting Started

To start using GenAI-RAG Agent, follow these simple steps:

### 1. Environment Setup

Follow these steps to set up your environment:
- Clone the Repository:

```bash
git clone https://github.com/zaaachos/GenAI-RAG-Agent.git
```

- Install Dependencies:
  
It is highly recommended, to use **conda** as your virtual enviroment:
```bash
conda create -n chatbot python=3.9
```
```bash
conda activate chatbot
```

### 2. Dependencies
Install the necessary dependencies by running:
```bash
pip install -r requirements.txt
```

### 3. Application
Run the Application Locally. Once dependencies are installed, you can run the FastAPI application locally by executing:

```bash
uvicorn main:app --reload
```

This will start the `uvicorn` server, and you can access the application at http://localhost:8000 in your web browser.