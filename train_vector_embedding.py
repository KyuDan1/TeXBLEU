import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Model, GPT2TokenizerFast
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import numpy as np

# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 커스텀 토크나이저 로드

tokenizer_dir = '.'  # tokenizer.json 파일이 있는 디렉토리 경로
tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_dir)


# GPT2 모델 로드 (임베딩 레이어만 사용)
try:
    gpt2_model = GPT2Model.from_pretrained('gpt2')
    embedding_layer = gpt2_model.wte.to(device)
except Exception as e:
    print(f"Error loading GPT2 model: {str(e)}")
    exit(1)

# 새로운 토큰에 대한 임베딩 초기화
model_vocab = set(tokenizer.convert_ids_to_tokens(range(gpt2_model.config.vocab_size)))
tokenizer_vocab = set(tokenizer.get_vocab().keys())
new_tokens = tokenizer_vocab - model_vocab
print(f"Found {len(new_tokens)} new tokens that require embeddings.")

new_embeddings = nn.Embedding(len(new_tokens), embedding_layer.embedding_dim).to(device)

# 데이터셋 클래스 정의
class CorpusDataset(Dataset):
    def __init__(self, directory, tokenizer):
        self.tokenizer = tokenizer
        self.files = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith('.tex')]
        self.data = []
        
        print(f"Found {len(self.files)} TeX files in the directory.")
        
        with ThreadPoolExecutor() as executor:
            self.data = list(tqdm(executor.map(self.process_file, self.files), total=len(self.files)))
        
        # 빈 리스트 제거
        self.data = [item for item in self.data if item]
        print(f"Processed {len(self.data)} non-empty files.")
    
    def process_file(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            encoded = self.tokenizer.encode(content, truncation=True, max_length=512)
            if encoded:
                return encoded
        except UnicodeDecodeError:
            print(f"Warning: Unable to read {file_path} with UTF-8 encoding. Trying with 'cp949'...")
            try:
                with open(file_path, 'r', encoding='cp949') as f:
                    content = f.read()
                encoded = self.tokenizer.encode(content, truncation=True, max_length=512)
                if encoded:
                    return encoded
            except Exception as e:
                print(f"Error processing file {file_path}: {str(e)}")
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
        return None
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        tokens = torch.tensor(self.data[idx])
        return tokens, len(tokens)

# Padding을 위한 collate function 정의
def collate_fn(batch):
    sequences, lengths = zip(*batch)
    max_len = max(lengths)
    
    padded_sequences = torch.zeros((len(sequences), max_len), dtype=torch.long)
    
    for i, seq in enumerate(sequences):
        end = lengths[i]
        padded_sequences[i, :end] = seq[:end]
    
    return padded_sequences.to(device), torch.tensor(lengths).to(device)

# 데이터 로더 생성
corpus_path = r'C:\Users\wjdrb\Downloads\drive-download-20240719T063242Z-001\temp_train\2301'
dataset = CorpusDataset(corpus_path, tokenizer)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

# 손실 함수와 옵티마이저 정의
criterion = nn.MSELoss()
optimizer = optim.Adam(new_embeddings.parameters(), lr=0.0001)

# 학습 루프
num_epochs = 100
for epoch in range(num_epochs):
    for batch, lengths in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        batch = batch.to(device)
        
        # 기존 임베딩과 새 임베딩 결합
        combined_embeddings = torch.cat([embedding_layer.weight, new_embeddings.weight])
        
        # 입력에 대한 임베딩 계산
        input_embeddings = combined_embeddings[batch]
        
        # 다음 토큰 예측을 위한 타겟 설정
        targets = torch.roll(input_embeddings, shifts=-1, dims=1)
        
        # 손실 계산 및 역전파
        loss = criterion(input_embeddings[:, :-1], targets[:, :-1])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# 학습된 임베딩 저장
torch.save(new_embeddings.state_dict(), 'new_embeddings.pth')

print("Training completed. New embeddings saved to 'new_embeddings.pth'.")
