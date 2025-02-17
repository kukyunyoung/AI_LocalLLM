from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

# OpenAI의 "text-embedding-3-small" 모델을 사용하여 임베딩을 생성합니다.
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def embed_text(texts):
    # 텍스트를 임베딩합니다.
    embedding = embeddings.embed_documents(texts)
    return embedding

def similarity(a, b):
    return cosine_similarity([a], [b])[0][0]

def calulate_similarity(sentences):
    embedded_sentences = embed_text(sentences)
    for i, sentence in enumerate(embedded_sentences):
        for j, other_sentence in enumerate(embedded_sentences):
            if i < j:
                print(
                    f"[유사도 {similarity(sentence, other_sentence):.4f}] {i} \t <=====> \t {j}"
                )