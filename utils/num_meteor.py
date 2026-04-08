"""
Num-METEOR: Numerical METEOR Evaluation Metric for Remote Sensing Image Captioning

Num-METEOR extends the standard METEOR metric to specifically evaluate:
1. Numerical accuracy (counts, measurements, spatial relationships)
2. Number word matching in captions
3. Quantitative description quality

This metric is particularly important for remote sensing image captioning
where numerical information (object counts, sizes, positions) is crucial.
"""

import re
import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from typing import List, Dict, Tuple
import numpy as np

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class NumMeteorScorer:
    """
    Num-METEOR scorer for remote sensing image captioning evaluation.
    
    This metric evaluates:
    - Standard METEOR score for overall caption quality
    - Numerical accuracy for counts, measurements, and spatial descriptions
    - Number word matching between reference and hypothesis
    """
    
    def __init__(self):
        # Regular expressions for numerical patterns
        self.number_patterns = {
            'digits': r'\b\d+\b',  # 1, 2, 3, etc.
            'number_words': r'\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion)\b',
            'ordinals': r'\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|twentieth|thirtieth|fortieth|fiftieth|sixtieth|seventieth|eightieth|ninetieth|hundredth|thousandth)\b',
            'quantifiers': r'\b(several|many|few|multiple|numerous|various|some|several|couple|pair|triple|quadruple)\b',
            'spatial_numbers': r'\b(\d+(?:st|nd|rd|th)|\d+\s*(?:feet|foot|meters|meter|km|km|miles|mile|acres|acre|hectares|hectare))\b'
        }
        
        # Number word mappings
        self.word_to_number = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
            'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20,
            'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70,
            'eighty': 80, 'ninety': 90, 'hundred': 100, 'thousand': 1000,
            'million': 1000000, 'billion': 1000000000
        }
        
        # Ordinal mappings
        self.ordinal_to_number = {
            'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5,
            'sixth': 6, 'seventh': 7, 'eighth': 8, 'ninth': 9, 'tenth': 10,
            'eleventh': 11, 'twelfth': 12, 'thirteenth': 13, 'fourteenth': 14, 'fifteenth': 15,
            'sixteenth': 16, 'seventeenth': 17, 'eighteenth': 18, 'nineteenth': 19, 'twentieth': 20
        }
    
    def extract_numbers(self, text: str) -> List[float]:
        """Extract all numerical values from text."""
        numbers = []
        
        # Extract digits
        digit_matches = re.findall(self.number_patterns['digits'], text)
        for match in digit_matches:
            numbers.append(float(match))
        
        # Extract number words
        word_matches = re.findall(self.number_patterns['number_words'], text.lower())
        for word in word_matches:
            if word in self.word_to_number:
                numbers.append(float(self.word_to_number[word]))
        
        # Extract ordinals
        ordinal_matches = re.findall(self.number_patterns['ordinals'], text.lower())
        for ordinal in ordinal_matches:
            if ordinal in self.ordinal_to_number:
                numbers.append(float(self.ordinal_to_number[ordinal]))
        
        # Extract spatial numbers (like "3 feet", "2 meters")
        spatial_matches = re.findall(self.number_patterns['spatial_numbers'], text.lower())
        for match in spatial_matches:
            # Extract the number part
            number_part = re.findall(r'\d+', match)
            if number_part:
                numbers.append(float(number_part[0]))
        
        return numbers
    
    def calculate_numerical_accuracy(self, reference: str, hypothesis: str) -> float:
        """
        Calculate numerical accuracy between reference and hypothesis captions.
        
        Returns:
            float: Numerical accuracy score (0-1)
        """
        ref_numbers = self.extract_numbers(reference.lower())
        hyp_numbers = self.extract_numbers(hypothesis.lower())
        
        if not ref_numbers and not hyp_numbers:
            return 1.0  # Both have no numbers, perfect match
        
        if not ref_numbers:
            return 0.0  # Reference has no numbers but hypothesis does
        
        if not hyp_numbers:
            return 0.0  # Hypothesis has no numbers but reference does
        
        # Calculate accuracy based on matching numbers
        matches = 0
        for ref_num in ref_numbers:
            # Check for exact matches or close matches (within 10% tolerance)
            for hyp_num in hyp_numbers:
                if abs(ref_num - hyp_num) < 0.1 * max(abs(ref_num), abs(hyp_num), 1):
                    matches += 1
                    break
        
        # Calculate accuracy as ratio of matched numbers
        accuracy = matches / len(ref_numbers)
        return min(accuracy, 1.0)
    
    def calculate_number_word_accuracy(self, reference: str, hypothesis: str) -> float:
        """
        Calculate number word accuracy between reference and hypothesis.
        
        Returns:
            float: Number word accuracy score (0-1)
        """
        ref_words = word_tokenize(reference.lower())
        hyp_words = word_tokenize(hypothesis.lower())
        
        # Extract number words
        ref_number_words = [word for word in ref_words if word in self.word_to_number or word in self.ordinal_to_number]
        hyp_number_words = [word for word in hyp_words if word in self.word_to_number or word in self.ordinal_to_number]
        
        if not ref_number_words and not hyp_number_words:
            return 1.0  # Both have no number words, perfect match
        
        if not ref_number_words:
            return 0.0  # Reference has no number words but hypothesis does
        
        if not hyp_number_words:
            return 0.0  # Hypothesis has no number words but reference does
        
        # Calculate Jaccard similarity for number words
        ref_set = set(ref_number_words)
        hyp_set = set(hyp_number_words)
        
        intersection = len(ref_set.intersection(hyp_set))
        union = len(ref_set.union(hyp_set))
        
        if union == 0:
            return 1.0
        
        return intersection / union
    
    def calculate_num_meteor(self, reference: str, hypothesis: str, gt_count: int, ref_count: int) -> float:
        """
        Calculate Num-METEOR using research paper formula:
        Num-METEOR = METEOR / |count(GT) - count(Ref)|
        
        Args:
            reference: Reference caption
            hypothesis: Hypothesis caption
            gt_count: Ground truth object count
            ref_count: Reference object count
            
        Returns:
            float: Num-METEOR score
        """
        # Calculate standard METEOR
        ref_tokens = word_tokenize(reference.lower())
        hyp_tokens = word_tokenize(hypothesis.lower())
        meteor = meteor_score([ref_tokens], hyp_tokens)
        
        # Calculate count difference
        count_diff = abs(gt_count - ref_count)
        
        # Apply research paper formula
        if count_diff == 0:
            print(f"DEBUG: Perfect count match! Num-METEOR = {meteor:.4f}")
            return meteor  # No penalty if counts match
        else:
            num_meteor = meteor / count_diff
            print(f"DEBUG: Count mismatch! METEOR={meteor:.4f}, count_diff={count_diff}, Num-METEOR={num_meteor:.4f}")
            return num_meteor
    
    def compute_score(self, references: List[str], hypotheses: List[str], 
                 gt_counts: List[int] = None, ref_counts: List[int] = None) -> Dict[str, float]:
        """
        Compute Num-METEOR scores using research paper formula.
        
        Args:
            references: List of reference captions
            hypotheses: List of hypothesis captions
            gt_counts: List of ground truth object counts (optional)
            ref_counts: List of reference object counts (optional)
            
        Returns:
            Dict containing Num-METEOR scores
        """
        if len(references) != len(hypotheses):
            raise ValueError("Number of references and hypotheses must match")
        
        num_meteor_scores = []
        meteor_scores = []
        count_diffs = []
        
        for i, (ref, hyp) in enumerate(zip(references, hypotheses)):
            # Calculate standard METEOR
            meteor = meteor_score([word_tokenize(ref.lower())], word_tokenize(hyp.lower()))
            
            # Get counts if provided
            gt_count = gt_counts[i] if gt_counts and i < len(gt_counts) else 0
            ref_count = ref_counts[i] if ref_counts and i < len(ref_counts) else 0
            
            # Calculate Num-METEOR using research paper formula
            count_diff = abs(gt_count - ref_count)
            if count_diff == 0:
                num_meteor = meteor  # No penalty if counts match
            else:
                num_meteor = meteor / count_diff  # Divide by count difference
            
            num_meteor_scores.append(num_meteor)
            meteor_scores.append(meteor)
            count_diffs.append(count_diff)
        
        # Calculate averages
        results = {
            'Num-METEOR': np.mean(num_meteor_scores),
            'METEOR': np.mean(meteor_scores),
            'Count-Difference': np.mean(count_diffs),
            'Numerical-Accuracy': np.mean([self.calculate_numerical_accuracy(ref, hyp) for ref, hyp in zip(references, hypotheses)]),
            'Number-Word-Accuracy': np.mean([self.calculate_number_word_accuracy(ref, hyp) for ref, hyp in zip(references, hypotheses)]),
            'Num-METEOR-Std': np.std(num_meteor_scores),
            'METEOR-Std': np.std(meteor_scores)
        }
        
        return results

def compute_num_meteor(references: List[str], hypotheses: List[str], 
                 gt_counts: List[int] = None, ref_counts: List[int] = None) -> Dict[str, float]:
    """
    Convenience function to compute Num-METEOR scores.
    
    Args:
        references: List of reference captions
        hypotheses: List of hypothesis captions
        
    Returns:
        Dict containing Num-METEOR scores
    """
    scorer = NumMeteorScorer()
    return scorer.compute_score(references, hypotheses)

# Example usage
if __name__ == "__main__":
    # Test examples
    reference = "there are three tennis courts on the right side of the image"
    hypothesis = "there are 3 tennis courts on the right side of the image"
    
    scorer = NumMeteorScorer()
    score = scorer.calculate_num_meteor(reference, hypothesis)
    print(f"Num-METEOR score: {score:.4f}")
    
    # Test with multiple captions
    references = [
        "there are three tennis courts on the right side",
        "two buildings are located in the upper left corner",
        "several cars are parked near the road"
    ]
    
    hypotheses = [
        "there are 3 tennis courts on the right side",
        "two buildings are in the upper left corner",
        "multiple cars are parked near the road"
    ]
    
    results = compute_num_meteor(references, hypotheses)
    print("\nBatch results:")
    for metric, score in results.items():
        print(f"{metric}: {score:.4f}")
