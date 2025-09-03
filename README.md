# RAG-LLM

> 본 프로젝트는 LLM의 고질적인 문제인 hallucination(사실과 다른 답변 생성)을 줄이고, 사용자에게 **근거 기반의 신뢰성 있는 응답을 제공**하는 것을 목적으로 합니다.
>
> 또한 사용자가 업로드한 PDF파일의 내용을 목차로 요약해 줌으로써 구조화된 목차와 세부 내용을 안정적으로 생성하고, 이로 하여금 사용자가 긴 문서도 효율적으로 탐색·활용할 수 있도록 지원합니다.
> 
> 이를 위해 PDF 문서를 대상으로 RAG(Retrieval-Augmented Generation) 구조를 구현하였으며, 다음과 같은 문제의식과 해결 과정을 거쳤습니다.
> ## 문제 정의
> 기존 LLM 서비스는 질문에 대해 근거 없는 답변을 생성하는 경우가 많음.
> 
> 특히 군사·기업 환경에서는 잘못된 정보 제공이 안전/보안 리스크로 이어질 수 있음.
>
> 추가로 대용량 문서 열람시 핵심내용에 대한 파악이 어려움.
> ## 해결 방법 탐색
> PDF 교보재를 기반으로 질의응답을 수행하는 RAG 구조 설계.
> 
> 문서 파싱 및 청크 단위 분할, 벡터 임베딩, 벡터스토어 검색 과정을 통해 근거 문서와 함께 답변을 제공하도록 구현.
> 
> 추가로 PDF 업로드 시 자동 목차 생성 및 요약 기능을 제공하여, 사용자가 문서 전체를 빠르게 파악하고 필요한 부분으로 바로 이동할 수 있도록 지원.
> ## 시행착오 및 개선
> 첫번째로 PDF 문서를 파싱하는 과정에서 인코딩 깨짐과 데이터 유실 문제가 발생했습니다.
>
> 초기에는 PDF를 단순히 텍스트 파일처럼 처리했지만, 실제로는 나무위키 인쇄물과 같은 비정형 PDF에서 글자 깨짐, 표·이미지 손실이 발생했습니다.
>
> 처음에는 PDF 파일을 단순히 텍스트라고 생각하고 작업을 진행했으나, 나무위키같은 PDF 인쇄물을 입력값으로 넣었을때 문제가 발생했습니다.
>
> 이 문제를 해결하기 위해 Poppler·Tesseract 기반의 UnstructuredPDFLoader 라이브러리를 도입하였고, 문서 특성에 따라 정형/비정형 PDF를 구분하여 서로 다른 로딩 방식을 적용했습니다.
>
> 그 결과, 텍스트가 올바르게 추출되고 표·이미지 기반 데이터까지 안정적으로 보존되어 데이터 유실 없는 파싱 파이프라인을 구축할 수 있었습니다.
>
> <br><br>
> 
> 두번째로 PDF 업로드를 통한 목차 생성 과정에서는 API 토큰 제한으로 인해 한 번에 전체 문서를 처리할 수 없는 문제가 발생했습니다.
> 
> 처음에는 긴 문서를 그대로 요청했으나, 응답이 잘리거나 누락되는 한계가 있었습니다. 이를 해결하기 위해 로직을 세 단계로 분리하였습니다.
> 
> 1. 목차 형식 추출 – PDF에서 제목·소제목만 뽑아 기본 구조를 생성.
> 2. 목차 머릿말 확장 – 추출된 목차 항목을 보완하여 각 파트의 개요를 채움.
> 3. 세부 내용 채우기 – 각 항목별로 내용을 분리 요청하여 순차적으로 완성.
>    
> 이 과정을 통해 토큰 제한을 우회하면서도 문서 전체에 대한 구조화된 목차와 세부 내용을 안정적으로 생성할 수 있었고, 사용자가 긴 문서도 효율적으로 탐색·활용할 수 있게 개선되었습니다.
> ## 추가 고려사항
> 군사시설이나 기업 환경은 클라우드 사용이 어려운 경우가 많아, 로컬 환경에서 동작 가능한 버전을 별도 제작.
> 
> GPU 성능 제약으로 모든 테스트를 완료하지는 못했지만, 보안 요건을 충족하는 온프레미스 AI 응용 가능성을 검증.

[Groq 모델 출처] (https://console.groq.com/docs/models)

[Local Llama Bllossom 모델 출처] (https://huggingface.co/MLP-KTLim/llama-3-Korean-Bllossom-8B)




# 목차
1. RAG의 개념  
2. pdfReader.py  
3. vectorstore.py  
4. llamaModule_groq.py



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


# pdfReader.py

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

<img width="872" height="814" alt="pdf_목차생성" src="https://github.com/user-attachments/assets/ee0e4fbd-6692-4bfa-b8ad-a9ddc6f6ac60" />


### Conversation_groq()
인자로 **사용자 입력 + RAG 사용 여부**를 받습니다.  
- **RAG=False** → 일반 LLM 응답  
- **RAG=True** → 문서 기반 응답  

질문 시 임베드된 코사인 유사도와 사용자 질문을 비교 → **점수가 높은 청크만 답변에 사용**  
점수가 낮으면 "대답하지 않음"으로 설정된 프롬프트 적용  

<img width="879" height="819" alt="리트리벌_관계없음" src="https://github.com/user-attachments/assets/a6cd77a7-2360-46eb-b810-ee09066adb65" />
<img width="878" height="818" alt="리트리벌_관계" src="https://github.com/user-attachments/assets/a16721c1-c4f6-48a7-8acc-804ecdb07271" />


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
