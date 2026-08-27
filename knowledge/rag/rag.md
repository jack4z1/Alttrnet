# Retrieval-Augmented Generation — Sample

*Sample document for testing the ALTTRNET ingestion pipeline. This is
not the official Wikipedia article; replace with real content.*

## What is RAG

Retrieval-augmented generation (RAG) is a technique that lets a large
language model use information retrieved from external sources when it
answers a question. Instead of relying only on the knowledge stored in
its weights, the model first looks up relevant documents and then
grounds its answer in those documents.

## Why it helps

RAG reduces hallucinations because the model has the relevant material
in front of it. It also lets the model use domain-specific or updated
information that is not in its training data, and it allows answers to
include sources that users can verify. Because new information only
requires updating the external store, there is less need to retrain
the model.

## How it works

Documents are split into chunks and converted into embeddings. The
embeddings are stored in a vector database. Given a user query, a
retriever selects the most relevant chunks, and those chunks are added
to the prompt. The model then generates an answer using both the query
and the retrieved context.

## Limitations

RAG does not eliminate every problem. A model can still hallucinate
around the source material, and it can misinterpret retrieved content
when the context is misleading. RAG systems may also struggle when
sources conflict, and combining details from multiple sources can merge
outdated and updated information in misleading ways.
