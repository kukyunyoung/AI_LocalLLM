from flask import Response
from langchain.prompts import PromptTemplate
from script.pdfReader import loadPdftoText, loadDocuPdftoText, pdfToText
from groq import Groq
from os import environ, listdir
import json, string, random

prompt_context = PromptTemplate.from_template(
    """
    #Requirements:
    Context의 내용으로 목차 작성해주세요.
    목차외에 내용은 작성하지 마세요.
    반드시 한국어로 작성 해주세요.
    
    #Context
    {context}
    """
)

prompt_table = PromptTemplate.from_template(
    """
    #Requirements:
    Context의 목차 틀을 가지고 Question의 내용에 대해 목차 만들어주세요.
    Context의 내용은 필요없습니다, 목차형식만 참고해주세요.
    Question에 있는 내용만 참고해주세요. 모르는 내용은 작성하지 마세요.
    목차의 부제목이 필요하면 추가하고 필요없다면 삭제하세요. 절대 포맷의 형태를 유지해야하는건 아닙니다.
    반드시 한국어로 작성 해주세요.
    
    #Context
    {context}
    #Question
    {question}
    """
)

prompt_answer = PromptTemplate.from_template(
    """
    #Requirements:
    Context의 각 목차별 주제의 내용을 채워주세요.
    내용은 Question의 내용을 참고해 구체적으로 작성 해 주세요.
    반드시 한국어로 작성 해주세요.
    
    #Context
    {context}
    #Question
    {question}
    """
)

# Groq 클라이언트 설정
client = Groq(api_key=environ.get("GROQ_API_KEY2")) # GROQ_API_KEY, GROQ_API_KEY2 토큰갯수 넘어가면 바꾸기
answer = []
response = client.chat.completions.create(model="llama-3.1-70b-versatile", messages=( {"role": "system", "content": ""},))

# GROQ API를 사용하여 모델 초기화
def InitModel(prompt : str):
    response = client.chat.completions.create(
        temperature=0.5,
        model="llama-3.1-70b-versatile",
        messages=(
            {"role": "system", "content": prompt},
        ),
        stream=True,
    )
    return response

### 질문에 대해 목차를 생성하고, 그 목차를 바탕으로 답변 생성
def questionLlama_Groq(pdfFile, isJson=False): # pdfFile: 사용자 업로드 PDF 파일
    
    # PDF 파일을 읽어 목차 생성, pdf 폴더에 있는 PDF 파일을 전부 읽음
    pdf = LoadJson() if isJson else loadPdftoText()

    # JSON 파일로 저장된 목차 데이터를 불러옴, 인자는 파일이름 (확장자 제외)
    # pdf = LoadJson('sample') 

    filled_prompt = prompt_context.format(context=pdf)
    print(filled_prompt)

    response = InitModel(filled_prompt)

    def generate():
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                answer.append(chunk.choices[0].delta.content)
        print("목차 틀 : \n" )
        print(*answer) # 목차 틀

        return AnswerTable(pdfFile)

    return generate()

### 생성된 목차를 사용하여 pdf 파일에 대한 목차 생성
answer_table = []
def AnswerTable(pdfFile):
    # 목차 형식을 유지하면서 질문에 대한 목차 생성
    filled_prompt = prompt_table.format(context=''.join(answer), question=pdfToText(pdfFile)) # pdfToText(pdfFile) : 업로드된 PDF 파일의 텍스트 추출

    response = InitModel(filled_prompt)

    def generate2():
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                answer_table.append(chunk.choices[0].delta.content)
        print('생성된 목차 \n') # 생성된 목차
        print(*answer_table)
        
        return AnswerDetail(pdfFile)

    return generate2()

### 생성된 목차를 사용하여 세부사항을 작성
answer_detail = []
answer_detail.append(answer_table)
def AnswerDetail(pdfFile):
    filled_prompt = prompt_answer.format(context=''.join(answer_table), question=pdfToText(pdfFile)) # pdfToText(pdfFile) : 업로드된 PDF 파일의 텍스트 추출

    response = InitModel(filled_prompt)
    
    # 질문에 대한 답변을 스트리밍
    def generate3():
        printTable=False
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                if(printTable==False):
                    yield ''.join(answer_table)
                    printTable=True
                answer_detail.append(chunk.choices[0].delta.content)
                yield chunk.choices[0].delta.content
        SaveJson(''.join(answer_detail))

    # Flask 응답으로 반환
    return Response(generate3(), content_type='text/plain; charset=utf-8')

# 목차와 세부사항들 JSON 파일로 저장
def SaveJson(data:list):
    # 랜덤하게 파일이름 생성
    letters = string.ascii_letters
    random_list = random.sample(letters, 10)
    fileName = ''.join(random_list)

    filepath = f'jsondata/{fileName}.json'
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4, sort_keys=True)

# JSON 파일 불러오기, 인자는 파일이름 (확장자 제외)
def LoadJson():
    filepath ='jsondata/'
    # jsondata 폴더에서 랜덤하게 하나 선택
    file_list = listdir(filepath)
    file_list_json = [file for file in file_list if file.endswith(".json")]
    random_file = random.choice(file_list_json)
    print(f"가져온 json 파일 : {random_file}" )

    with open(filepath+random_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data