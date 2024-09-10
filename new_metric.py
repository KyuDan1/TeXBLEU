import torch
import numpy as np
from transformers import GPT2TokenizerFast, GPT2Model
import re
import math


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f"Using device: {device}")

# 토크나이저 및 모델 로드를 함수 내부로 이동
def load_models_and_tokenizer():
    global tokenizer, gpt2_model, embedding_layer, positional_embedding, new_embeddings
    
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    gpt2_model = GPT2Model.from_pretrained('gpt2')
    embedding_layer = gpt2_model.wte.to(device)
    positional_embedding = gpt2_model.wpe.to(device)

    # 새로운 임베딩 로드 (필요한 경우)
    try:
        if device == 'cuda':
            new_embeddings_state = torch.load('new_embeddings.pth')
        else:
            new_embeddings_state = torch.load('new_embeddings.pth', map_location=torch.device('cpu'))

        new_vocab_size, embedding_dim = new_embeddings_state['weight'].shape
        new_embeddings = torch.nn.Embedding(new_vocab_size, embedding_dim).to(device)
        new_embeddings.load_state_dict(new_embeddings_state)
    except FileNotFoundError:
        print("Warning: new_embeddings.pth not found. Using default embeddings.")
        new_embeddings = None


def spacing(text):
    result = []
    for i, char in enumerate(text):
        if char == "\\":
            if i == 0 or text[i-1] != " ":
                result.append(" \\")
            else:
                result.append("\\")
        else:
            result.append(char)
    return ''.join(result)


def get_token_embeddings(sentence):
    sentence = spacing(sentence)
    tokens = tokenizer.encode(sentence, truncation=True, max_length=512)
    #print(f"Tokenized text: {tokens}")
    decoded_tokens = [tokenizer.decode([token]) for token in tokens]
    print(f"Decoded tokens: {decoded_tokens}")
    
    token_ids = torch.tensor(tokens).unsqueeze(0).to(device)
    positions = torch.arange(0, token_ids.size(1)).unsqueeze(0).to(device)
    
    if new_embeddings is not None:
        token_embeddings = torch.cat([embedding_layer.weight, new_embeddings.weight])[token_ids]
    else:
        token_embeddings = embedding_layer(token_ids)
    
    pos_embeddings = positional_embedding(positions) * 100
    
    
    return list(zip(token_embeddings[0], pos_embeddings[0]))

def cosine_distance(emb1, emb2):
    return 1 - torch.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()

def token_distance(token1, token2, w_emb=0.5, w_pos=0.5, alpha=2, beta=0.1):
    emb1, pos1 = token1
    emb2, pos2 = token2
    
    # 임베딩 거리에 지수 적용
    emb_dist = cosine_distance(emb1, emb2) ** alpha
    
    # 위치 거리에 비선형성 추가
    pos_dist = math.tanh(beta * torch.abs(pos1 - pos2).float().mean().item())
    
    distance = w_emb * emb_dist + w_pos * pos_dist
    
    return distance

def n_gram_similarity(ref_tokens, pred_tokens, n, max_d=2.0):
    ref_ngrams = [ref_tokens[i:i+n] for i in range(len(ref_tokens)-n+1)]
    pred_ngrams = [pred_tokens[i:i+n] for i in range(len(pred_tokens)-n+1)]
    
    L_n = min(len(ref_ngrams), len(pred_ngrams))
    if L_n == 0:
        return 0
    
    # core part //author
    total_distance = sum(
        sum(token_distance(ref_token, pred_token) 
            for ref_token, pred_token in zip(ref_ngram, pred_ngram))
        for ref_ngram, pred_ngram in zip(ref_ngrams[:L_n], pred_ngrams[:L_n])
    )
    
    return 1 - (total_distance / (L_n * n)) #1 - (total_distance / (L_n * n * max_d))

def texbleu(reference, prediction, max_n=2, weights=None):
    if weights is None:
        weights = [1/max_n] * max_n
    
    ref_tokens = get_token_embeddings(reference)
    pred_tokens = get_token_embeddings(prediction)
    
    n_gram_scores = [n_gram_similarity(ref_tokens, pred_tokens, n) 
                     for n in range(1, max_n+1)]
    
    # 길이 계산
    ref_length = len(ref_tokens)
    pred_length = len(pred_tokens)
    
    # Brevity penalty 계산
    if pred_length > ref_length:
        bp = 1
    else:
        bp = math.exp(1 - ref_length / pred_length)
    
    # 길이 차이에 대한 추가 페널티 계산
    length_ratio = min(ref_length, pred_length) / max(ref_length, pred_length)
    
    # BLEU 점수 계산
    bleu_score = math.exp(sum(w * math.log(max(s, 1e-10)) 
                      for w, s in zip(weights, n_gram_scores)))
    
    # 길이 페널티를 적용한 최종 점수 계산
    final_score = bleu_score # * length_ratio
    
    return round(final_score, 4)


load_models_and_tokenizer()