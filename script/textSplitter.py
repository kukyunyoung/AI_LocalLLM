from langchain_text_splitters import CharacterTextSplitter


def split_text(docs):
    # CharacterTextSplitter를 사용하여 텍스트를 청크(chunk)로 분할하는 코드
    text_splitter = CharacterTextSplitter(
        # 텍스트를 분할할 때 사용할 구분자를 지정합니다. 기본값은 "\n\n"입니다.
        separator="\n",
        # 분할된 텍스트 청크의 최대 크기를 지정합니다 (문자 수).
        chunk_size=256,
        # 분할된 텍스트 청크 간의 중복되는 문자 수를 지정합니다.
        chunk_overlap=15,
        # 텍스트의 길이를 계산하는 함수를 지정합니다.
        length_function=len,
    )
    # 추출한 텍스트를 사용하여 문서를 분할
    texts = text_splitter.create_documents(docs)

    print(
        f"첫 번째 문서의 길이: {len(texts[0].page_content)}"
    )  # 분할된 문서의 첫 번째 청크 길이를 출력합니다.
    print(
        f"첫 번째 문서의 청크 수: {len(texts)}"
    )  # 분할된 문서의 청크 수를 출력합니다.
    # 반복문으로 분할된 텍스트 청크를 출력합니다.
    # for i, doc in enumerate(texts):
    #     print(f"Document {i + 1}:")
    #     print(doc.page_content)
    return texts
