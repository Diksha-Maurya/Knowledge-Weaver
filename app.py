
import re
import gradio as gr
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


def create_rag_chain():

    # Load the document
    loader = WebBaseLoader("https://en.wikipedia.org/wiki/Retrieval-augmented_generation")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    splits = text_splitter.split_documents(docs)

    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = FAISS.from_documents(splits, embedding=emb)
    retriever = vs.as_retriever(k=1)

    model_id = "google/flan-t5-large"
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    gen_pipe = pipeline(
        "text2text-generation",
        model=mdl,
        tokenizer=tok,
        max_new_tokens=160,
        min_new_tokens=40,
    )
    llm = HuggingFacePipeline(pipeline=gen_pipe)

    prompt = PromptTemplate.from_template(
        "You are an assistant for question-answering tasks. "
        "Use only the following context to answer the question. "
        "Do not copy the context verbatim. Your answer should be a new, synthesized response. "
        "Ignore any markdown like '[edit]' or citation numbers like '[1]' in your final answer. "
        "Answer in 3-5 complete sentences.\\n\\n"
        "Context:\\n{context}\\n\\nQuestion: {question}\\n\\nAnswer:"
    )

    def join_docs(docs):
        joined_content = "\\n".join(d.page_content for d in docs)
        cleaned_content = re.sub(r'\\[edit\\]', '', joined_content)
        cleaned_content = re.sub(r'\\[\\d+\\]', '', cleaned_content)
        return cleaned_content

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

def get_answer(question):
    return rag_chain.invoke(question)

iface = gr.Interface(
    fn=get_answer,
    inputs=gr.Textbox(lines=2, placeholder="Enter your question here..."),
    outputs="text",
    title="Retrieval-Augmented Generation (RAG) Web App",
    description="Ask a question about Retrieval-Augmented Generation. The app will use a Wikipedia article as its knowledge base to answer."
)

if __name__ == "__main__":
    iface.launch()