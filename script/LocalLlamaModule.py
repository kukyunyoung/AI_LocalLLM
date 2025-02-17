from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from flask import Response
from script.pdfReader import loadPdftoText

# 프롬프트 템플릿 정의
prompt_context = PromptTemplate.from_template(
    """
    #Requirements:
    Context의 내용으로 문서의 목차만 작성해주세요.
    반드시 한국어로 작성 해주세요.
    
    #Context
    {context}
    """
)

def questionOllama():

    # PDF 파일을 텍스트로 로드
    pdf = loadPdftoText()

    # ChatOllama 모델 설정
    llm = ChatOllama(model="llama3.1:70b", num_gpu=1, temperature=0.5)
    print(pdf)

    chain = prompt_context | llm

    # 응답을 스트리밍하는 함수
    def generate():
        try:
            # 스트리밍 응답 처리
            for response in chain.stream({"context" : pdf}):
                if response.content:  # 응답 내용이 있을 때만 전송
                    yield response.content
        except Exception as e:
            # 예외가 발생하면 에러 메시지 반환
            yield f"Error: {str(e)}"

    return Response(generate(), content_type="text/plain; charset=utf-8")