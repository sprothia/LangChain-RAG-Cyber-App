# Langchain CyberAgent RAG App with Streamlit

## Overview

This project is a Retrieval-Augmented Generation (RAG) application built using Python's Langchain library and Streamlit for a web-based user interface. The application processes PDF documents, vectorizes their content, and allows users to input text queries to generate summaries, extract topics, and find related YouTube videos. The application is optimized to persist vectorized data, reducing the need to reprocess documents on subsequent runs, thus enhancing performance.

## Features

- **Document Loading**: Load and process multiple PDF documents for retrieval and analysis.
- **Text Vectorization**: Split documents into chunks and vectorize the content using OpenAI's embeddings.
- **Persistent Vector Database**: Save the vectorized data to disk, allowing for quick retrieval without re-vectorization on subsequent runs.
- **Streamlit UI**: A simple and interactive web interface for entering text queries and viewing results.
- **Multi-Query Retrieval**: Generate multiple variations of user queries to enhance document retrieval.
- **Summary and Topic Extraction**: Summarize the content based on context and extract key topics.
- **YouTube Search Integration**: Automatically search YouTube for relevant videos based on extracted topics.

## Installation

To run this project, you need to have Python installed on your machine. Follow these steps to set up and run the application:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo/langchain-rag-app.git
   cd rag_langchain
   
2. **Download requirements.txt**:
   ```bash
   pip install -r requirements.txt


3. **Insert your OpenAI API Key**:
