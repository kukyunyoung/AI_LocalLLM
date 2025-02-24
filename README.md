# AI_LocalLLM

RAG 시스템을 활용하여 로컬환경에서 llama-3-Korean-Bllossom-8B 모델을 사용한 목차생성 프로젝트입니다.

[모델 출처] (https://huggingface.co/MLP-KTLim/llama-3-Korean-Bllossom-8B)

사용자가 입력을 원할 때 

# RAG

RAG란 Retrieval-Augmented Generation (검색 증강 생성)을 의미하며 크게 4가지의 단계로 구성 되어 있습니다.

1. 문서 전처리 (textsplit, embed 등)
2. 전처리된 데이터를 벡터화 하여 저장 (vectorstore)
3. 사용자의 질문을 벡터로 변환하고 저장된 벡터 데이터에서 유사한 청크를 찾아 모델에게 제공 (retrieve)
4. LLM에서 답변 생성

