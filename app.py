import re
import gradio as gr
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# --- Expanded glossary for short/definition queries ---
GLOSSARY = {
    "rag": "Retrieval-Augmented Generation",
    "llm": "Large Language Model",
    "vector store": "A database that stores text as numerical embeddings for efficient similarity search",
    "embeddings": "Numerical representations of text capturing semantic meaning"
}

def is_short_query(q: str) -> bool:
    ql = q.lower().strip()
    return (
        len(ql.split()) <= 3 or
        any(phrase in ql for phrase in ["full form", "define", "what is", "stand for"])
    )

# Global variables for vector store and LLM
vs = None
llm = None

def create_rag_chain():
    global vs, llm

    # Load documents
    URLS = [
        "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
        "https://python.langchain.com/docs/expression_language/how_to/rag",
        "https://python.langchain.com/docs/how_to#retrieval",
    ]
    docs = []
    for u in URLS:
        docs.extend(WebBaseLoader(u).load())

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    # Vector store with improved embedding model
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = FAISS.from_documents(splits, embedding=emb)

    # Improved retriever with similarity search
    retriever = vs.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6}
    )

    # Model with tuned parameters
    model_id = "google/flan-t5-large"
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    gen_pipe = pipeline(
        "text2text-generation",
        model=mdl,
        tokenizer=tok,
        max_new_tokens=200,
        min_new_tokens=40,
        truncation=True,
        do_sample=False,
        num_beams=3
    )
    llm = HuggingFacePipeline(pipeline=gen_pipe)

    # Enhanced context cleaning
    def join_docs(docs):
        text = "\n".join(d.page_content for d in docs)
        text = re.sub(r'\[edit\]', '', text)
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'(?m)^\s*(Contents|See also|References|External links|This page was last edited.*)\s*$', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    # Flexible main prompt
    prompt = PromptTemplate.from_template(
        "You are an assistant for question-answering tasks. "
        "Use ONLY the provided context to answer clearly and concisely. "
        "If the answer is not in the context, say 'I don't know.' "
        "Do not copy the context verbatim. Provide a synthesized response tailored to the question. "
        "Ignore any markdown like '[edit]' or citation numbers like '[1]'. "
        "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )

    rag_chain = (
        {
            "context": retriever | join_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
    )
    return rag_chain

print("Creating RAG chain... This may take a moment.")
rag_chain = create_rag_chain()
print("RAG chain created successfully.")

# Short-query path with debug logging
def short_answer(q: str) -> str:
    ql = q.lower()
    for k, v in GLOSSARY.items():
        if k in ql:
            print(f"Glossary match for '{q}': {v}")
            return v

    # Fallback using tighter retriever
    base = vs.as_retriever(search_type="similarity", search_kwargs={"k": 1})
    def join_docs(docs):
        text = "\n".join(d.page_content for d in docs)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    short_prompt = PromptTemplate.from_template(
        "Provide a ONE-SENTENCE definition or the expanded form if the question asks for it. "
        "Use the context if needed. If unknown from the context, say 'I don't know.'\n\n"
        "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    chain = ({"context": base | join_docs, "question": RunnablePassthrough()} | short_prompt | llm)
    result = str(chain.invoke(q))
    print(f"Short answer for '{q}': {result}")
    return result

# Main answer function with debug logging
def get_answer(question):
    q = question.strip()
    if is_short_query(q):
        print(f"Short query detected: '{q}'")
        return short_answer(q)
    print(f"Running RAG chain for: '{q}'")
    docs = vs.as_retriever(search_type="similarity", search_kwargs={"k": 6}).invoke(q)
    print("Retrieved context:", [doc.page_content for doc in docs])
    result = str(rag_chain.invoke(q))
    print(f"RAG answer: {result}")
    return result

iface = gr.Interface(
    fn=get_answer,
    inputs=gr.Textbox(lines=2, placeholder="Enter your question here..."),
    outputs="text",
    title="Retrieval-Augmented Generation (RAG) Web App",
    description="Ask a question about RAG. The app uses selected web sources as its knowledge base."
)

if __name__ == "__main__":
    iface.launch()