# ==================================================
# EWU RAG Chatbot – CPU Version for Hugging Face Spaces
# ==================================================

import os
import json
import requests
from bs4 import BeautifulSoup

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import CrossEncoder

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

import gradio as gr

# ------------------------------
# Hugging Face token (for private models)
# ------------------------------
HF_TOKEN = os.environ.get("HF_TOKEN")  # Set secret in HF Spaces

# ------------------------------
# JSON processing
# ------------------------------
def flatten_json(data, parent_key="", file_name=""):
    docs = []
    if isinstance(data, dict):
        for key, value in data.items():
            new_key = f"{parent_key}.{key}" if parent_key else key
            docs.extend(flatten_json(value, new_key, file_name))
    elif isinstance(data, list):
        for i, item in enumerate(data):
            new_key = f"{parent_key}[{i}]"
            docs.extend(flatten_json(item, new_key, file_name))
    else:
        text_value = str(data).strip()
        if text_value:
            docs.append(
                Document(
                    page_content=f"{parent_key}: {text_value}",
                    metadata={"source": file_name, "path": parent_key}
                )
            )
    return docs

def load_json_docs(data_directory="./data"):
    docs = []
    print("Loading JSON files...")
    for file_name in os.listdir(data_directory):
        if not file_name.endswith(".json"):
            continue
        file_path = os.path.join(data_directory, file_name)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            file_docs = flatten_json(data, file_name, file_name)
            docs.extend(file_docs)
            print(f"  - {file_name}: {len(file_docs)} entries processed")
        except Exception as e:
            print(f"  - Error loading {file_name}: {e}")
    print(f"\nTotal JSON documents created: {len(docs)}")
    return docs

def chunk_documents(docs, chunk_size=2000, overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunked_docs = splitter.split_documents(docs)
    print(f"Total text chunks after splitting: {len(chunked_docs)}")
    return chunked_docs

# ------------------------------
# Scraping dynamic EWU info
# ------------------------------
DYNAMIC_URLS = {
    "Admission deadlines": [
        "https://ewubd.edu/undergraduate-dates-deadline",
        "https://ewubd.edu/graduate-dates-deadline"
    ],
    "Events": ["https://ewubd.edu/events"],
    "Faculty": [
        "https://fse.ewubd.edu/computer-science-engineering/faculty-members",
        "https://fse.ewubd.edu/electrical-electronic-engineering/faculty-members",
        "https://flass.ewubd.edu/english-department/faculty-members"
    ],
    "Grading": ["https://www.ewubd.edu/grades-rules-and-regulations"],
    "Tuition fees": ["https://ewubd.edu/undergraduate-tuition-fees"]
}

def scrape_dynamic_docs(urls_dict):
    docs = []
    headers = {"User-Agent": "Mozilla/5.0"}
    for category, urls in urls_dict.items():
        for url in urls:
            try:
                r = requests.get(url, headers=headers, timeout=10)
                r.raise_for_status()
                soup = BeautifulSoup(r.text, "html.parser")
                body = soup.find('body')
                page_text = body.get_text(separator="\n", strip=True) if body else soup.get_text(separator="\n", strip=True)
                if page_text:
                    docs.append(Document(
                        page_content=f"LATEST {category} info from {url}:\n{page_text}",
                        metadata={"source": url, "category": category}
                    ))
            except Exception as e:
                print(f"Failed to scrape {url}: {e}")
    return docs

# ------------------------------
# Vectorstore
# ------------------------------
def build_vectorstore(docs):
    model_name = "sentence-transformers/all-mpnet-base-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectorstore = FAISS.from_documents(docs, embeddings)
    print("Vectorstore created successfully.")
    return vectorstore

# ------------------------------
# Cross-encoder reranker
# ------------------------------
class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-12-v2"):
        self.model = CrossEncoder(model_name, device="cpu")

    def rerank(self, query, docs, top_k=3):
        if not docs:
            return []
        pairs = [(query, d.page_content) for d in docs]
        scores = self.model.predict(pairs)
        scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:top_k]]

# ------------------------------
# CPU-friendly LLM (Phi-2)
# ------------------------------
def load_llm(model_id="2001Samir/phi2-7b-instruct", hf_token=None):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map={"": "cpu"},  # Force CPU
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        use_auth_token=hf_token,
        trust_remote_code=True
    )
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512, temperature=0.2)
    return HuggingFacePipeline(pipeline=pipe)

# ------------------------------
# Build RAG chain
# ------------------------------
def build_rag_chain(llm, vectorstore, reranker, k_retrieve=25, k_rerank=10):
    def retrieve_and_rerank(question):
        retrieved_docs = vectorstore.similarity_search(question, k=k_retrieve)
        dynamic_priority = [d for d in retrieved_docs if "ewubd.edu" in d.metadata.get("source", "")]
        static_others = [d for d in retrieved_docs if d not in dynamic_priority]
        combined_docs = dynamic_priority + static_others
        reranked = reranker.rerank(question, combined_docs[:k_rerank], top_k=k_rerank)
        return "\n\n".join([d.page_content for d in reranked])

    template = """
<|system|>
You are an assistant for East West University.
Answer based ONLY on the provided context.
If unknown, reply: "I don't have enough information to answer that question."
</s>
<|user|>
CONTEXT:
{context}

QUESTION:
{question}
</s>
<|assistant|>
"""
    prompt = PromptTemplate.from_template(template)
    return ({"context": retrieve_and_rerank, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())

# ------------------------------
# Main setup
# ------------------------------
data_directory = "./data"
docs = load_json_docs(data_directory)
chunked_docs = chunk_documents(docs)
dynamic_docs = scrape_dynamic_docs(DYNAMIC_URLS)
chunked_dynamic_docs = chunk_documents(dynamic_docs)
all_docs = chunked_docs + chunked_dynamic_docs

vectorstore = build_vectorstore(all_docs)
reranker = Reranker()
llm = load_llm(hf_token=HF_TOKEN)
rag_chain = build_rag_chain(llm, vectorstore, reranker)

# ------------------------------
# Gradio interface
# ------------------------------
def answer_fn(question):
    try:
        resp = rag_chain.invoke(question)
        start = resp.find("<|assistant|>")
        if start != -1:
            return resp[start+len("<|assistant|>"):].strip()
        return resp.strip()
    except Exception as e:
        return f"Error: {e}"

iface = gr.Interface(
    fn=answer_fn,
    inputs=gr.Textbox(lines=2, placeholder="Ask a question about EWU..."),
    outputs=gr.Textbox(lines=20, interactive=False),
    title="EWU RAG Chatbot (CPU)",
    description="EWU info: deadlines, events, faculty, grading, tuition. CPU-friendly deployment."
)

iface.launch()
