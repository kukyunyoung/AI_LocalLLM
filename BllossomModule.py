import torch, string, random, json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from script.pdfReader import loadPdftoText, loadDocuPdftoText, pdfToText
from langchain.prompts import PromptTemplate
from transformers import TextIteratorStreamer
from flask import Response
from threading import Thread
from os import listdir
import time as Time

# 목차 생성
prompt_makeTable = PromptTemplate.from_template(
    """
    #Requirements:
    Context의 내용으로 목차 작성해주세요.
    목차는 예시로 작성해주세요.
    반드시 한국어로 작성 해주세요.
    
    #Context
    {context}
    """
)

# 생성된 목차로 문서 내용 작성
prompt_writeTable = PromptTemplate.from_template(
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

# 목차별 내용 작성
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

# 대화 생성
prompt_conversation = PromptTemplate.from_template( 
    """
    #Requirements:
    You are a helpful AI assistant. 
    Please answer the user's questions kindly. 
    당신은 유능한 AI 어시스턴트 입니다. 
    사용자의 질문에 대해 친절하게 답변해주세요.
    """
)

# transformer 모델 불러오기
model_id = 'MLP-KTLim/llama-3-Korean-Bllossom-8B'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()

# 비동기적으로 스트림 출력 생성
def generate_output(prompt, instruction, max_tokens=2048, cut_size=10): # 출력 최대길이 2048, 청크 단위 10
    messages = [
        {"role": "system", "content": f"{prompt}"},
        {"role": "user", "content": f"{instruction}"}
    ]
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda:0")

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)  # skip_prompt 사용해야 답변에 프롬프트 노출안됨

    thread = Thread(
        target=model.generate,
        args=(input_ids,),
        kwargs={
            "streamer": streamer,
            "max_new_tokens": max_tokens,
            "eos_token_id": terminators,
            "do_sample": True,
            "temperature": 0.1,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
        }
    )
    thread.start()

    for chunk in streamer:
        while len(chunk) >= cut_size:
            yield chunk[:cut_size]
            chunk = chunk[cut_size:]
        if len(chunk) > 0:
            yield chunk

stt = 0

def MakeTable(uploadPdf):
    old_table = []
    global stt
    stt = Time.time()
    
    def generate():
        # 1단계: MakeTable 목차 생성
        pdf_content = loadPdftoText()
        filled_prompt = prompt_makeTable.format(context=pdf_content)
        response = generate_output(filled_prompt, "문서의 목차를 작성해주세요.")

        for chunk in response:
            yield chunk
            old_table.append(chunk)
        
        print("old_table : ", old_table)

        # 2단계: WriteTable 호출로 목차 재작성
        filled_prompt = prompt_writeTable.format(context=old_table, question=pdfToText(uploadPdf))
        response = generate_output(filled_prompt, "Question 내용을 참고해서 Context 형태로 목차 재작성 해주세요.")
        
        new_table = []
        for chunk in response:
            yield chunk
            new_table.append(chunk)

        yield "\n\n\n\n\n\n\n\n\n\n"

        answer = []
        # 3단계: AnswerTable 호출로 목차별 내용 작성
        filled_prompt = prompt_answer.format(context=new_table, question=pdfToText(uploadPdf))
        response = generate_output(filled_prompt, "Question 내용을 참고해서 각 목차별 내용을 작성해주세요.")
        
        for chunk in response:
            yield chunk
            answer.append(chunk)

        yield "\n\n\n\n\n\n\n\n\n\n"

        # 소요 시간 출력
        endTime = Time.time()
        yield f"소요 시간 : {int(endTime - stt)}초\n"
        yield f"최종 생성된 문장 길이 : {len(''.join(old_table+new_table+answer))}"

    return Response(generate(), content_type='text/plain; charset=utf-8')

# `Conversation` 함수 내의 generator 정의
def Conversation(inputText: str):
    stt = Time.time()
    def generate():
        filled_prompt = prompt_conversation.format()
        response = generate_output(filled_prompt, inputText)
        
        answer = []
        for chunk in response:
            yield chunk
            answer.append(chunk)

        yield "\n\n\n\n"

        # 소요 시간 출력
        endTime = Time.time()
        yield f"소요 시간 : {int(endTime - stt)/60}분 {int(endTime - stt)%60}초\n"
        yield f"최종 생성된 문장 길이 : {len(''.join(answer))}"   

    return Response(generate(), content_type='text/plain; charset=utf-8')

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

