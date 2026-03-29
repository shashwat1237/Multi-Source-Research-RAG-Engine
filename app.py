import os
import requests
import gradio as gr
import numpy as np
from bs4 import BeautifulSoup
import trafilatura

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_groq import ChatGroq



GROQ_API_KEY = #hidden



def fetch_text(url):
    try:
        
        downloaded = trafilatura.fetch_url(url)

        if downloaded:
            text = trafilatura.extract(downloaded)
            if text and len(text) > 200:
                print(f"[Trafilatura SUCCESS] {url} ({len(text)} chars)")
                return text


        print(f"[Fallback scraper] {url}")

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }

        res = requests.get(url, timeout=10, headers=headers)

        if res.status_code != 200:
            print(f" Failed: {url} (status {res.status_code})")
            return ""

        soup = BeautifulSoup(res.text, "html.parser")

        for tag in soup(["script", "style", "noscript"]):
            tag.extract()

        text = soup.get_text(separator=" ")
        text = " ".join(text.split())

        print(f"[Fallback SUCCESS] {url} ({len(text)} chars)")

        return text

    except Exception as e:
        print(f" Error fetching {url}: {e}")
        return ""


def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200
    )
    return splitter.split_text(text)



def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def deduplicate(chunks, embeddings, threshold=0.85):
    if len(chunks) < 2:
        return chunks

    vectors = embeddings.embed_documents(chunks)
    keep_indices = []

    for i, vec_i in enumerate(vectors):
        keep = True
        for j in keep_indices:
            vec_j = vectors[j]
            sim = cosine_similarity(vec_i, vec_j)

            if sim > threshold:
                keep = False
                break

        if keep:
            keep_indices.append(i)

    return [chunks[i] for i in keep_indices]



def process(urls, question):
    try:
        if not urls.strip():
            raise ValueError("Please enter at least one URL")

        url_list = [u.strip() for u in urls.split(",") if u.strip()]
        print("Processing URLs:", url_list)

        texts = []
        for url in url_list:
            text = fetch_text(url)
            if text:
                texts.append(text)

        if not texts:
            raise ValueError(" Failed to fetch content from all URLs")

        combined_text = "\n".join(texts)

        if len(combined_text) < 200:
            raise ValueError(" Extracted content too small")

        # Chunk
        chunks = chunk_text(combined_text)

        if len(chunks) < 2:
            raise ValueError(" Not enough data after chunking")

        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        
        chunks = deduplicate(chunks, embeddings)

        if len(chunks) < 2:
            raise ValueError(" Too much duplication")

        
        vectorstore = FAISS.from_texts(chunks, embeddings)

        llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        temperature=0.2,
        model_name="llama-3.3-70b-versatile"  
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
            return_source_documents=True
        )

        result = qa_chain.invoke({"query": question})

        answer = result.get("result") or result.get("output_text") or "No answer generated."

        
        sources = []
        for doc in result.get("source_documents", [])[:2]:
            sources.append(doc.page_content[:300])

        sources_text = "\n\n---\n\n".join(sources) if sources else "No sources found."

        return answer, sources_text

    except Exception as e:
        return f" Error: {str(e)}", ""



with gr.Blocks() as app:
    gr.Markdown("# Multi-Source Research Aggregator (FIXED)")
    gr.Markdown("Enter URLs (comma-separated) and ask questions")

    urls_input = gr.Textbox(
        label=" URLs",
        placeholder="https://example.com, https://example2.com"
    )

    question_input = gr.Textbox(
        value="Summarize all sources",
        label=" Question"
    )

    run_button = gr.Button("Run")

    answer_output = gr.Textbox(label=" Answer")
    sources_output = gr.Textbox(label=" Sources")

    run_button.click(
        fn=process,
        inputs=[urls_input, question_input],
        outputs=[answer_output, sources_output]
    )



if __name__ == "__main__":
    app.launch(share=True)
