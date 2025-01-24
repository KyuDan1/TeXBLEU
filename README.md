# TeXBLEU: Automatic Metric for Evaluate LaTeX Format

### Accepted in ICASSP 2025!

# Algorithm: Computing TexBLEU

```
Require: Reference text R, Prediction text P, max n-gram N, weights {w_1, ..., w_N}
Ensure: TexBLEU score

Function Preprocess(R, P):
    R, P ← coherently spaced R, P
    Return R, P

Function TokenDistance(t_1, t_2):
    e_1, p_1 ← embedding and position of t_1
    e_2, p_2 ← embedding and position of t_2
    d_emb ← cosDist(e_1, e_2)^α
    d_pos ← tanh(β · |p_1 - p_2|)
    
    Return (d_emb + d_pos) / 2

Function NGramSimilarity(R, P, n):
    R, P ← Preprocess(R, P)
    L_n ← min(|R| - n + 1, |P| - n + 1)
    d_total ← 0
    For i ← 1 to L_n:
        For j ← 1 to n:
            d_total ← d_total + TokenDistance(R[i+j-1], P[i+j-1])
    
    Return 1 - (d_total / (L_n · n))

T_R ← tokenize and embed R
T_P ← tokenize and embed P
For n ← 1 to N:
    sim_n ← NGramSimilarity(T_R, T_P, n)

TeXBLEU ← exp(Σ(n=1 to N) w_n log(sim_n(R,P)))
Return TeXBLEU
```

## Explanation

1. The `Preprocess` function processes the input texts to ensure consistent spacing.
2. The `TokenDistance` function calculates the distance between two tokens. This is a combination of embedding distance and positional distance.
3. The `NGramSimilarity` function computes the n-gram similarity for a given n.
4. The main algorithm tokenizes and embeds the input texts, then calculates the n-gram similarity for each n.
5. Finally, it computes and returns the TeXBLEU score.

This algorithm is used to measure the similarity between texts containing mathematical expressions.



# Proposed Method

## LaTeX Specialized Tokenizer and Embedding Model

We first created a tokenizer and embedding model based on mathematical expressions in LaTeX format. The most extensive source of LaTeX-based documents is the arXiv papers. We followed the method proposed by Clement to bulk download arXiv paper files. This method downloads the tex files of all papers in 2023 using a manifest file containing the metadata of the papers. In total, we collected `.tex` files of approximately 172 K papers.

Using this large corpus of LaTeX, we created a new tokenizer and embedding. When creating a tokenizer for LaTeX, it is advantageous to use byte pair encoding (BPE), which can handle unicode characters byte-by-byte. We created a BPE tokenizer using 172 K arXiv papers as a corpus. Unlike other tokenizers, this tokenizer can capture LaTeX grammar elements and tokenize them to reflect the features of LaTeX grammar. Based on this tokenizer and the arXiv corpus, we fine-tuned the publicly available pretrained gpt-2 embedding model.

[Results section to be inserted here]

## Token Distance

[Algorithm section to be inserted here]

When both the reference and prediction sentences were tokenized using our tokenizer and embedding model, we first defined the distance between the tokens. The token distance d(t1, t2) is defined as:

d(t1, t2) = (cosDist(e1, e2)^α + tanh(β · |p1 - p2|)) / 2

where t1, t2 are the tokens, e1, e2 are the token embeddings, p1, p2 are the tokens' positional embeddings, and α, β are the hyperparameters. The term 'cosDist' refers to the cosine distance, which is defined as 1 minus the cosine similarity. Specifically:

cosDist(e1, e2) = 1 - cosSim(e1, e2)

The reason for using cosine distance in Equation 1 is that, in the case of embedding vectors in natural language processing, the direction of the vectors is often more important than their magnitude, as is well known. Taking the power of cosine distance allows similar embeddings to be measured as even more similar and dissimilar embeddings as more distinct. It is already known that applying nonlinear power can adjust the variance. Therefore, we applied a power of α to the cosine distance.

For positional embeddings, the absolute difference in positions is important; therefore, we computed the L1 distance. In addition, the hyperbolic tangent function was used because it limits the output within the range of [-1,1], thereby reducing the influence of extreme values while being more sensitive to small differences.

## N-gram Similarity

In this section, we describe the n-gram technique. The n-gram similarity sim_n(R, P) for a reference sentence R and a predicted sentence P is given by:

sim_n(R, P) = 1 - (Σ_i=1^L_n Σ_j=1^n d(r_ij, p_ij)) / (L_n · n)

where n is the length of the n-gram, and L_n is the number of n-grams. Here, function d is the token distance function.

Specifically, as shown in Algorithm 1, TeXBLEU is calculated using BLEU's n-gram technique. First, in a tokenized sentence, n tokens are grouped to form L_n n-grams. For each n-gram, the token distance is calculated for n tokens. This process was repeated using a nested for-loop to sum all distances. Dividing by L_n and n yields the average n-gram distance. Finally, to shift the concept to a similarity score rather than a distance, n-gram similarity was defined as the n-gram distance subtracted from 1, similar to how it was done in Equation 2.

## Preprocessing

In LaTeX syntax for mathematical expressions, whether there is a space before commands start with a backslash typically does not affect compilation. Owing to this characteristic, metrics where spacing is crucial, such as WER or CER, were not suitable. To ensure consistent tokenization in the LaTeX format, we added a space before all LaTeX commands that started with a backslash. However, multiple spaces are reduced to a single space.

## TeXBLEU

Based on Algorithm 1 and the above explanation, TeXBLEU is finally equal to the following expression:

TeXBLEU = exp(Σ_n=1^N w_n log sim_n(R, P))

On the other hand, the original BLEU metric employs a brevity penalty. However, in the LaTeX format, even with differences in sentence length, the meaning can remain highly similar. For example, '\frac{}{}' and '/' both represent a fraction, where a single character and nine characters convey a similar meaning. Therefore, the brevity penalty used in the original BLEU was not applied to TeXBLEU. In fact, when the brevity penalty was applied, it resulted in a lower performance compared to TeXBLEU.
