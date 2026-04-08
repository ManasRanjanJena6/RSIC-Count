"""
Evaluation metrics for image captioning.
Includes BLEU, METEOR, ROUGE, CIDEr, SPICE.
"""

import numpy as np
from collections import defaultdict


def compute_bleu_scores(references, hypotheses, max_n=4):
    """
    Compute BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores.
    
    Args:
        references: List of reference lists (each image can have multiple refs)
        hypotheses: List of hypothesis strings
        max_n: Maximum n-gram order
    
    Returns:
        Dictionary with BLEU scores
    """
    try:
        from nltk.translate.bleu_score import corpus_bleu
        
        # Tokenize
        refs_tokenized = [[ref.split() for ref in refs] for refs in references]
        hyps_tokenized = [hyp.split() for hyp in hypotheses]
        
        scores = {}
        for n in range(1, max_n + 1):
            if n == 1:
                weights = (1.0, 0, 0, 0)
            elif n == 2:
                weights = (0.5, 0.5, 0, 0)
            elif n == 3:
                weights = (0.33, 0.33, 0.33, 0)
            elif n == 4:
                weights = (0.25, 0.25, 0.25, 0.25)
            
            score = corpus_bleu(refs_tokenized, hyps_tokenized, weights=weights)
            scores[f'BLEU-{n}'] = score * 100
        
        return scores
    
    except ImportError:
        print("Warning: NLTK not installed. Install with: pip install nltk")
        return {}


def compute_meteor_score(references, hypotheses):
    """
    Compute METEOR score.
    
    Args:
        references: List of reference lists
        hypotheses: List of hypothesis strings
    
    Returns:
        METEOR score
    """
    try:
        from nltk.translate.meteor_score import meteor_score
        
        scores = []
        for refs, hyp in zip(references, hypotheses):
            ref_tokens = [ref.split() for ref in refs]
            hyp_tokens = hyp.split()
            score = meteor_score(ref_tokens, hyp_tokens)
            scores.append(score)
        
        return {'METEOR': np.mean(scores) * 100}
    
    except ImportError:
        print("Warning: NLTK not installed for METEOR.")
        return {}


def compute_rouge_score(references, hypotheses):
    """
    Compute ROUGE-L score.
    
    Args:
        references: List of reference lists
        hypotheses: List of hypothesis strings
    
    Returns:
        ROUGE-L score
    """
    try:
        from rouge_score import rouge_scorer
        
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = []
        
        for refs, hyp in zip(references, hypotheses):
            # Average over all references
            ref_scores = [scorer.score(ref, hyp)['rougeL'].fmeasure for ref in refs]
            scores.append(np.mean(ref_scores))
        
        return {'ROUGE-L': np.mean(scores) * 100}
    
    except ImportError:
        print("Warning: rouge-score not installed. Install with: pip install rouge-score")
        return {}


def compute_cider_score(references, hypotheses):
    """
    Compute CIDEr score.
    
    Args:
        references: List of reference lists
        hypotheses: List of hypothesis strings
    
    Returns:
        CIDEr score
    """
    try:
        from pycocoevalcap.cider.cider import Cider
        
        # Format for CIDEr
        gts = {i: refs for i, refs in enumerate(references)}
        res = {i: [hyp] for i, hyp in enumerate(hypotheses)}
        
        scorer = Cider()
        score, scores = scorer.compute_score(gts, res)
        
        return {'CIDEr': score * 100}
    
    except ImportError:
        print("Warning: pycocoevalcap not installed. CIDEr unavailable.")
        return {}


def compute_count_accuracy(predicted_counts, true_counts, threshold=2):
    """
    Compute count prediction accuracy metrics.
    
    Args:
        predicted_counts: [N, num_categories] predicted counts
        true_counts: [N, num_categories] ground truth counts
        threshold: Accuracy threshold
    
    Returns:
        Dictionary with count metrics
    """
    mae = np.mean(np.abs(predicted_counts - true_counts))
    mse = np.mean((predicted_counts - true_counts) ** 2)
    rmse = np.sqrt(mse)
    
    # Accuracy within threshold
    within_threshold = np.mean(np.abs(predicted_counts - true_counts) <= threshold) * 100
    
    # Per-category MAE
    per_cat_mae = np.mean(np.abs(predicted_counts - true_counts), axis=0)
    
    return {
        'Count_MAE': mae,
        'Count_MSE': mse,
        'Count_RMSE': rmse,
        f'Count_Acc@{threshold}': within_threshold,
        'Per_Category_MAE': per_cat_mae.tolist()
    }


def compute_all_metrics(references, hypotheses, predicted_counts=None, true_counts=None):
    """
    Compute all available metrics.
    
    Args:
        references: List of reference caption lists
        hypotheses: List of generated captions
        predicted_counts: Optional predicted counts
        true_counts: Optional ground truth counts
    
    Returns:
        Dictionary with all metrics
    """
    metrics = {}
    
    # Caption metrics
    print("Computing BLEU...")
    metrics.update(compute_bleu_scores(references, hypotheses))
    
    print("Computing METEOR...")
    metrics.update(compute_meteor_score(references, hypotheses))
    
    print("Computing ROUGE...")
    metrics.update(compute_rouge_score(references, hypotheses))
    
    print("Computing CIDEr...")
    metrics.update(compute_cider_score(references, hypotheses))
    
    # Count metrics (if available)
    if predicted_counts is not None and true_counts is not None:
        print("Computing count metrics...")
        metrics.update(compute_count_accuracy(predicted_counts, true_counts))
    
    return metrics


def print_metrics(metrics, title="Evaluation Metrics"):
    """Pretty print metrics"""
    print("\n" + "="*60)
    print(f"{title:^60}")
    print("="*60)
    
    for metric, score in metrics.items():
        if isinstance(score, (int, float)):
            print(f"{metric:.<40} {score:>8.2f}")
        elif isinstance(score, list):
            print(f"{metric}:")
            for i, s in enumerate(score):
                print(f"  Category {i}: {s:.2f}")
    
    print("="*60 + "\n")
