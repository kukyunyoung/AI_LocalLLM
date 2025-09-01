from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from env.env import os_listdir, os_path_join
from unstructured.partition.pdf import partition_pdf
import pdfplumber, io

# pdf 최적화 관련 문서
# https://wikidocs.net/231565

def pdfToText(pdfFile):
    text=""
    with pdfplumber.open(io.BytesIO(pdfFile)) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

def show_metadata(docs):
    if docs:
        print("[metadata]")
        print(list(docs[0].metadata.keys()))
        print("\n[examples]")
        max_key_length = max(len(k) for k in docs[0].metadata.keys())
        for k, v in docs[0].metadata.items():
            print(f"{k:<{max_key_length}} : {v}")

# UnstructuredPDFLoader < poppler, tesseract 설치 및 환경변수 세팅 필요함
def loadUnstruct_PdfToText(pdfFile:bytes = None):
    def loadDocs(texts, file_path):
        loader = UnstructuredPDFLoader(file_path, languages=["kor","eng"])
        docs = loader.load()
        texts.extend([doc.page_content for doc in docs])

    if not pdfFile:
        DIRECTORY_PATH = "pdf/"
    texts = []

    if pdfFile is None:
        for filename in os_listdir(DIRECTORY_PATH):
            if filename.endswith(".pdf"):
                print(f"Loading {filename}...")
                file_path = os_path_join(DIRECTORY_PATH, filename)
                loadDocs(texts, file_path)
    else:
        print(f"Loading {pdfFile}...")
        loadDocs(texts, pdfFile)
        
    return texts

def loadPdftoText():
    DIRECTORY_PATH = "pdf/"
    texts = []

    # 디렉토리 내의 모든 PDF 파일 탐색
    for filename in os_listdir(DIRECTORY_PATH):
        if filename.endswith(".pdf"):
            print(f"Loading {filename}...")
            file_path = os_path_join(DIRECTORY_PATH, filename)
            loader = PyPDFLoader(file_path, extract_images=False)
            loads = loader.load()
            texts.extend([doc.page_content for doc in loads])

    return texts

def loadDocuPdftoText():
    DIRECTORY_PATH = "docu_pdf/"
    texts = []

    # 디렉토리 내의 모든 PDF 파일 탐색
    for filename in os_listdir(DIRECTORY_PATH):
        if filename.endswith(".pdf"):
            print(f"Loading {filename}...")
            file_path = os_path_join(DIRECTORY_PATH, filename)
            loader = PyPDFLoader(file_path, extract_images=False)
            loads = loader.load()
            texts.extend([doc.page_content for doc in loads])

    return texts

def extract_pdf_elements(filepath):
    """
    PDF 파일에서 이미지, 테이블, 그리고 텍스트 조각을 추출합니다.
    path: 이미지(.jpg)를 저장할 파일 경로
    fname: 파일 이름
    """
    return partition_pdf(
        filename=filepath,
        extract_images_in_pdf=False,  # PDF 내 이미지 추출 활성화
        infer_table_structure=False,  # 테이블 구조 추론 활성화
        chunking_strategy="by_title",  # 제목별로 텍스트 조각화
        max_characters=20000,  # 최대 문자 수
        new_after_n_chars=3800,  # 이 문자 수 이후에 새로운 조각 생성
        combine_text_under_n_chars=2000,  # 이 문자 수 이하의 텍스트는 결합
    )