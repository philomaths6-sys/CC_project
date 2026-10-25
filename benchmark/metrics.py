import re
from typing import List

def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())

def rouge_1_score(hypothesis: str, reference: str) -> float:
    hyp_tokens = tokenize(hypothesis)
    ref_tokens = tokenize(reference)
    if not hyp_tokens or not ref_tokens:
        return 0.0
    ref_counts = {}
    for t in ref_tokens:
        ref_counts[t] = ref_counts.get(t, 0) + 1
    overlap = 0
    for t in hyp_tokens:
        if ref_counts.get(t, 0) > 0:
            overlap += 1
            ref_counts[t] -= 1
    precision = overlap / len(hyp_tokens)
    recall = overlap / len(ref_tokens)
    return (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

def rouge_l_score(hypothesis: str, reference: str) -> float:
    hyp_tokens = tokenize(hypothesis)
    ref_tokens = tokenize(reference)
    m, n = len(hyp_tokens), len(ref_tokens)
    if m == 0 or n == 0:
        return 0.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if hyp_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    precision = lcs / m
    recall = lcs / n
    return (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

def keyword_hit_rate(hypothesis: str, keywords: List[str]) -> float:
    if not keywords:
        return 1.0
    hyp_lower = hypothesis.lower()
    hits = sum(1 for kw in keywords if kw.lower() in hyp_lower)
    return hits / len(keywords)

def compute_composite_quality(rouge_l: float, rouge_1: float, kw_hit: float) -> float:
    score = (0.50 * rouge_l) + (0.30 * rouge_1) + (0.20 * kw_hit)
    return round(score * 100.0, 2)
