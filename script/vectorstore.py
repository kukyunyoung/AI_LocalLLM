from langchain_community.vectorstores import FAISS
from script.textSplitter import split_text
from script.pdfReader import show_metadata, loadPdftoText, loadUnstructuredPdfToText
from flask import jsonify
from env.env import os_path_exists
from langchain_community.embeddings import HuggingFaceEmbeddings

faiss_db = None

embeddings_model = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sroberta-nli",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

def initFaiss():
    global faiss_db  # 전역 변수 사용 선언
    #split_docs = split_text(loadPdftoText())
    split_docs = split_text(loadUnstructuredPdfToText())
    print(split_docs)
    # FAISS 를 통해 벡터 저장소 생성
    faiss_db = FAISS.from_documents(split_docs, embeddings_model)
    faiss_db.add_documents(split_docs)
    saveFaiss()

def saveFaiss():
    global faiss_db  # 전역 변수 사용 선언
    if faiss_db is not None:
        FAISS.save_local(faiss_db, "./db/faiss_db")

def loadFaiss():
    global faiss_db  # 전역 변수 사용 선언
    try:
        if not os_path_exists("./db/faiss_db"):
            raise FileNotFoundError("FAISS database file not found.")
        faiss_db = FAISS.load_local(
            folder_path="./db/faiss_db",
            embeddings=embeddings_model,
            allow_dangerous_deserialization=True,
        )
    except (FileNotFoundError, Exception) as e:
        print(f"Error loading FAISS database: {e}")
        initFaiss()

def similarity_search(query):
    global faiss_db  # 전역 변수 사용 선언
    if faiss_db is not None:
        similar_docs = faiss_db.similarity_search(query)

    print(f"문서의 개수: {len(similar_docs)}")
    print("[검색 결과]\n")
    return [doc.page_content for doc in similar_docs]

def getRetriever():
    global faiss_db  # 전역 변수 사용 선언
    if faiss_db is not None:
        return faiss_db.as_retriever(search_type="mmr")
