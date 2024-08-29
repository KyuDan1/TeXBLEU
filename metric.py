import re
from typing import List
import math
from collections import Counter
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import Tokenizer
from huggingface_hub import snapshot_download

def load_tokenizer_from_hub(repo_name: str = "Kyudan/TeXBLEU-Tokenizer") -> Tokenizer:
    # Hugging Face Hub에서 토크나이저 다운로드
        snapshot_dir = snapshot_download(repo_id=repo_name)
    
    # 다운로드한 토크나이저 불러오기
        tokenizer = Tokenizer.from_file(f"{snapshot_dir}/tokenizer.json")
        return tokenizer


class TeXBLEU:
    # List: 문자열들의 리스트.
    def __init__(self, repo_name: str):
        self.tokenizer = self._load_tokenizer_from_hub(repo_name)
    
    def _load_tokenizer_from_hub(self, repo_name: str) -> Tokenizer:
        return load_tokenizer_from_hub(repo_name)

    #스트링 형으로 이루어진 리스트와 int형 vocab size를 입력으로 받아서  Tokenizer 객체를 반환한다.
    def _train_bpe_tokenizer(self, corpus: List[str], vocab_size: int) -> Tokenizer:
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
                             vocab_size=vocab_size)
        # tokenizer의 전처리 단계로 공백을 기준으로 분리하는 Whitespace() 전처리기를 설정.
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.train_from_iterator(corpus, trainer)
        return tokenizer
    
    # 명령어 앞에 한칸 띄고, 여러 공백은 지우기.
    def preprocess_latex(self, text: str) -> str:
        # Add space before '\'
        text = re.sub(r'(?<![\\])(\\)', r' \1', text)
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    """
    input:   tokens = ["\\frac", "{", "1", "}", "{", "2", "}"]

    output:           ["0:\\frac", "1:{", "2:1", "3:}", "4:{", "5:2", "6:}"]

    """
    def add_positional_encoding(self, tokens: List[str]) -> List[str]:
        return [f"{i}:{token}" for i, token in enumerate(tokens)]
    
    def calculate_texbleu(self, reference: str, candidate: str, max_n: int = 4) -> float:
        ref_tokens = self.tokenizer.encode(self.preprocess_latex(reference)).tokens
        cand_tokens = self.tokenizer.encode(self.preprocess_latex(candidate)).tokens
        print(f"ref_tokens : {ref_tokens}")
        print(f"cand_tokens : {cand_tokens}")
        # 아래 두줄을 주석으로 두면 positional encoding 끄게됨.
        ref_tokens = self.add_positional_encoding(ref_tokens)
        cand_tokens = self.add_positional_encoding(cand_tokens)
        
        bp = self._brevity_penalty(ref_tokens, cand_tokens)
        
        scores = []
        for n in range(1, max_n + 1):
            scores.append(self._modified_precision(ref_tokens, cand_tokens, n)) # n은 n-gram의 n임.
        
        if 0 in scores:
            return 0
        
        score = bp * math.exp(sum(math.log(s) for s in scores) / max_n) #modified_precision들의 평균을 exp취한 것에 bp 를 곱함.
        return score
    
    def _brevity_penalty(self, ref_tokens: List[str], cand_tokens: List[str]) -> float:
        r = len(ref_tokens)
        c = len(cand_tokens)
        
        if c > r:
            return 1
        else:
            return math.exp(1 - r/c)
    
    def _modified_precision(self, ref_tokens: List[str], cand_tokens: List[str], n: int) -> float:
        ref_ngrams = Counter(self._get_ngrams(ref_tokens, n))
        cand_ngrams = Counter(self._get_ngrams(cand_tokens, n))
        
        max_counts = {}
        for ngram, count in cand_ngrams.items():
            max_counts[ngram] = max(0, count - max(0, count - ref_ngrams[ngram]))
        
        if len(cand_ngrams) == 0:
            return 0
        
        return sum(max_counts.values()) / sum(cand_ngrams.values())
    
    """
    예시:
    입력: tokens = ["The", "quick", "brown", "fox"], n = 2
    출력: [("The", "quick"), ("quick", "brown"), ("brown", "fox")]
    """
    def _get_ngrams(self, tokens: List[str], n: int) -> List[tuple]:
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

