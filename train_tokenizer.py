import re
import os

def load_tex_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file]

def process_line(line):
    return re.sub(r'(\$.*?\$|\\\[.*?\\\]|\\\(.*?\\\))', '', line)

def process_corpus(corpus):
    return [process_line(line) for line in corpus]

def find_tex_files(root_folder):
    tex_files = []
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.tex'):
                tex_files.append(os.path.join(root, file))
    return tex_files

def process_all_tex_files(root_folder):
    all_processed_corpus = []
    tex_files = find_tex_files(root_folder)
    
    for filepath in tex_files:
        corpus = load_tex_file(filepath)
        processed_corpus = process_corpus(corpus)
        all_processed_corpus.extend(processed_corpus)
    
    return all_processed_corpus

# 사용 예시
root_folder = r'C:\Users\wjdrb\Downloads\drive-download-20240719T063242Z-001'
processed_corpus = process_all_tex_files(root_folder)


corpus = ["F(x)={\sqrt{\frac{10x^{-8}}{-8}}}+C_{1} =-{\frac{5}{4x^{8}}}+C_{1}\quad{\mathrm {if~}}x<0.",
"F(x)=\sqrt{10}x^{-8/-8}+C_{1}=-\frac{5} {4}x^{8}+C_{1}\quad\mathrm{if~}x<0."]


from tqdm import tqdm
from tokenizers import Tokenizer, trainers, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from huggingface_hub import HfApi, HfFolder
from typing import List

def train_and_upload_tokenizer(corpus: List[str], vocab_size: int = 30000, repo_name: str = "Kyudan/TeXBLUE-Tokenizer"):
    # 토크나이저와 트레이너 초기화
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
                         vocab_size=vocab_size)
    tokenizer.pre_tokenizer = Whitespace()

    # tqdm을 사용하여 진행 상태 표시
    print("Training tokenizer...")
    corpus_with_progress = tqdm(corpus, desc="Training Progress", total=len(corpus))
    tokenizer.train_from_iterator(corpus_with_progress, trainer)

    # 토크나이저를 파일로 저장
    tokenizer.save("tokenizer.json")
    
    # Hugging Face Hub에 업로드
    api = HfApi()
    token = HfFolder.get_token()
    
    api.upload_file(
        path_or_fileobj="tokenizer.json",
        path_in_repo="tokenizer.json",
        repo_id=repo_name,
        token=token,
    )
    
    print(f"Tokenizer uploaded to Hugging Face Hub under the repository {repo_name}")

    train_and_upload_tokenizer(processed_corpus)