# RAG-LLM

RAG 시스템을 활용한 사용자 대화 및 pdf 요약 후 목차생성 프로젝트입니다.

> 해당 프로젝트는 사용자에게 **Retrieval**을 통해 답변을 제공해 LLM의 고질적 문제인 **hallucination**을 해결하고자 제작되었습니다.  
> 또한 기밀이 중요한 기업이나 군사보안쪽에서 AI 활용도가 낮은 문제를 해결하고자 **로컬 버전**도 제작하였습니다.  
> (그래픽카드 성능 이슈로 테스트해보지 못하여 로컬버전 문서작성은 생략합니다)

[Groq 모델 출처] (https://console.groq.com/docs/models)

[Local Llama Bllossom 모델 출처] (https://huggingface.co/MLP-KTLim/llama-3-Korean-Bllossom-8B)



### 목차
1. RAG의 개념  
2. pdfReader.py  
3. vectorstore.py  
4. llamaModule_groq.py  

---

# RAG의 개념

RAG는 총 8단계로 구성됩니다

<img width="1920" height="1080" alt="rag-1" src="https://github.com/user-attachments/assets/720721da-8b8a-4210-afc6-71fe578368e2" />
<img width="1920" height="1080" alt="rag-2" src="https://github.com/user-attachments/assets/9f7c69b6-cfbc-4366-9a4f-9c3cccbc324b" />
[출처] (https://teddylee777.github.io/langchain/rag-tutorial/)

### 문서 로드 & 분할
제 프로젝트의 경우에는 pdf를 로드합니다.  
프로젝트의 pdf폴더에 있는 pdf파일을 모두 긁어와 텍스트 형태로 변환하고,  
해당 텍스트를 청크(chunk) 단위로 분할하여 인덱스로 저장합니다.

### 임베딩
앞서 청크 단위로 분할되어 있는 문서를 벡터로 변환합니다.  
벡터로 변환된 문서들은 코사인 유사도를 가지게 되는데 청크별로 내용이 관련이 깊으면 1에 가깝게, 관련이 적으면 0에 가깝게 맵핑이 됩니다.

### 벡터DB
임베딩된 벡터들을 DB에 저장합니다.  
Faiss를 사용하여 데이터화 하였고 DB 사용은 하지 않았습니다.

### 검색 (Retrieval)
앞서 청크 단위로 맵핑된 코사인 유사도와 사용자의 질문을 비교하여 관련이 제일 높은 청크를 사용하여 답변을 제공합니다.

### 프롬프트
검색된 결과(context)와 사용자의 질문(question)을 바탕으로 미리 입력된 프롬프트에 대응하여 답변을 제공합니다.

### 모델선택
모델은 Ollama를 사용하여 로컬모델을 사용하여도 되고 OpenAI, Groq 같은 API key를 활용해 설치 없이 사용할 수 있습니다.

### 결과
답변은 HTML을 이용하여 사용자에게 표시되도록 하였고  
JSON 파일로 저장 할 수 있는 기능도 만들었습니다.

---

# pdfReader.py

[Uploading pdfReader.py…]()


RAG에서 문서 로드 역할을 합니다.  
- **pdfToText()**: `pdfplumber` 라이브러리를 사용하여 텍스트를 쭉 읽어 저장합니다.  

```python
def pdfToText(pdfFile):
    text=""
    with pdfplumber.open(io.BytesIO(pdfFile)) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text
```
- **loadPdfToText()**: pdfToText()와 같은 역할을 하지만 인자 없이 `pdf/` 경로에 있는 파일을 모두 검색하여 저장합니다.

```python
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
```

- **loadUnstruct_PdfToText()**: 비정형 데이터(사진 등)를 텍스트로 변환해줍니다. `poppler`, `tesseract` 설치 및 환경변수 세팅이 필요합니다.

```python
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
```

---

# vectorstore.py
RAG에서 **분할, 임베딩, 벡터DB 역할**을 합니다.  
- `app.py` 실행 시 `loadFaiss()` 실행 → `/db/faiss_db` 폴더에 인덱스가 없으면 `/pdf` 폴더의 PDF를 읽어 **비정형 데이터 텍스트 변환** 진행
```python
def initFaiss():
    global faiss_db  # 전역 변수 사용 선언
    #split_docs = split_text(loadPdftoText())
    split_docs = split_text(loadUnstruct_PdfToText())
    print(split_docs)
    # FAISS 를 통해 벡터 저장소 생성
    faiss_db = FAISS.from_documents(split_docs, embeddings_model)
    faiss_db.add_documents(split_docs)
    saveFaiss()

def saveFaiss():
    global faiss_db  # 전역 변수 사용 선언
    if faiss_db is not None:
        FAISS.save_local(faiss_db, "./db/faiss_db")
```
- 사실 `pdfReader.py`에서 직접적으로 함수를 사용하지 않고 `vectorstore.py`에서 호출하기 때문에 **문서 로드 과정도 포함**됩니다.  
- 변환 후 `/db/faiss_db`에 FAISS 저장 → 이후 실행 시 해당 인덱스를 불러와 사용합니다.  

---

# llamaModule_groq.py
RAG에서 **검색, 프롬프트, 모델선택, 결과 생성** 역할을 합니다.  
답변을 생성할 때 `MakeTable_groq()` 혹은 `Conversation_groq()`을 사용합니다.  
두 과정 모두 `getRetriever()` 함수로 검색을 실행하고 이후 로직이 달라집니다.  

### MakeTable_groq()
인자로 **HTML에서 업로드된 PDF 파일**을 받습니다.  
3단계로 로직이 진행됩니다:
1. `prompt_makeTable` → 저장된 문서의 목차 형식을 가져옴  
2. `prompt_writeTable` → 사용자가 업로드한 PDF와 첫 번째 결과의 목차 형식을 바탕으로 PDF 목차 생성  
3. `prompt_answer` → 두 번째 결과의 내용을 채워 최종 완성, 사용자에게 제공  

→ 하나의 프롬프트에 모든 작업을 몰아넣는 대신 **단계별 안정성 확보** 방식입니다.  

### Conversation_groq()
인자로 **사용자 입력 + RAG 사용 여부**를 받습니다.  
- **RAG=False** → 일반 LLM 응답  
- **RAG=True** → 문서 기반 응답  

질문 시 임베드된 코사인 유사도와 사용자 질문을 비교 → **점수가 높은 청크만 답변에 사용**  
점수가 낮으면 "대답하지 않음"으로 설정된 프롬프트 적용  

### 기타
- **SaveJson() / LoadJson()** → 사용자가 파일 저장/불러오기 가능  

---

# 마무리

공부하면서 제작한 프로젝트이니 만큼 기초적인 내용을 다루고 있고 많은 내용을 담아내지는 못했다고 생각합니다.  
그래도 유용한 기술을 사용하여 짜임새 있는 프로젝트를 완성했고 여러 번의 시행착오를 거치며 마무리되었습니다.  

추후에는  
- **API 기반 파인튜닝**  
- **데이터 처리**  
를 중점적으로 학습하여 경험을 넓히고 싶습니다.  
