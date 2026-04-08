"""
Vocabulary builder for RSIC-Count dataset.
Creates word-to-index and index-to-word mappings from captions.json
"""

import json
from collections import Counter
import pickle
import argparse
from pathlib import Path


class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.freq_threshold = freq_threshold
        self.stoi = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
        self.itos = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.vocab_size = 4
    
    def build_vocabulary(self, captions_file):
        """Build vocabulary from captions.json"""
        with open(captions_file, 'r') as f:
            captions_data = json.load(f)
        
        word_freq = Counter()
        
        # Handle both dictionary and list formats
        if isinstance(captions_data, dict):
            # Check if it's {"annotations": [...]} format
            if 'annotations' in captions_data and isinstance(captions_data['annotations'], list):
                captions_data = captions_data['annotations']
            else:
                # Format: {"img_id": ["caption1", "caption2"]}
                for img_id, captions in captions_data.items():
                    if isinstance(captions, list):
                        for caption in captions:
                            if isinstance(caption, str):
                                tokens = caption.lower().split()
                                word_freq.update(tokens)
                    elif isinstance(captions, str):
                        tokens = captions.lower().split()
                        word_freq.update(tokens)
                # Early return if not annotations format
                if word_freq:
                    # Add words above frequency threshold
                    for word, freq in word_freq.items():
                        if freq >= self.freq_threshold:
                            self.stoi[word] = self.vocab_size
                            self.itos[self.vocab_size] = word
                            self.vocab_size += 1
                    print(f"Vocabulary built: {self.vocab_size} tokens (threshold: {self.freq_threshold})")
                    return self
        
        if isinstance(captions_data, list):
            # Format: [{"filename": "...", "text_output": "..."}]
            for item in captions_data:
                if 'text_output' in item:
                    caption = item['text_output']
                elif 'caption' in item:
                    caption = item['caption']
                else:
                    continue
                tokens = caption.lower().split()
                word_freq.update(tokens)
        
        # Add words above frequency threshold
        for word, freq in word_freq.items():
            if freq >= self.freq_threshold:
                self.stoi[word] = self.vocab_size
                self.itos[self.vocab_size] = word
                self.vocab_size += 1
        
        print(f"Vocabulary built: {self.vocab_size} tokens (threshold: {self.freq_threshold})")
        return self
    
    def numericalize(self, text):
        """Convert text to sequence of indices"""
        tokens = text.lower().split()
        return [self.stoi.get(token, self.stoi["<unk>"]) for token in tokens]
    
    def save(self, filepath):
        """Save vocabulary to file"""
        vocab_dict = {
            'stoi': self.stoi,
            'itos': self.itos,
            'vocab_size': self.vocab_size,
            'freq_threshold': self.freq_threshold
        }
        with open(filepath, 'wb') as f:
            pickle.dump(vocab_dict, f)
        print(f"Vocabulary saved to {filepath}")
    
    @staticmethod
    def load(filepath):
        """Load vocabulary from file"""
        with open(filepath, 'rb') as f:
            vocab_dict = pickle.load(f)
        
        vocab = Vocabulary(freq_threshold=vocab_dict['freq_threshold'])
        vocab.stoi = vocab_dict['stoi']
        vocab.itos = vocab_dict['itos']
        vocab.vocab_size = vocab_dict['vocab_size']
        print(f"Vocabulary loaded: {vocab.vocab_size} tokens")
        return vocab


def main():
    parser = argparse.ArgumentParser(description="Build vocabulary for RSIC-Count dataset")
    parser.add_argument('--captions', type=str, default='data/captions.json',
                        help='Path to captions.json')
    parser.add_argument('--output', type=str, default='data/vocab.pkl',
                        help='Output vocabulary file')
    parser.add_argument('--freq_threshold', type=int, default=5,
                        help='Minimum word frequency')
    
    args = parser.parse_args()
    
    # Build and save vocabulary
    vocab = Vocabulary(freq_threshold=args.freq_threshold)
    vocab.build_vocabulary(args.captions)
    vocab.save(args.output)
    
    # Print sample mappings
    print("\nSample vocabulary mappings:")
    for i in range(min(20, vocab.vocab_size)):
        print(f"  {i}: {vocab.itos[i]}")


if __name__ == "__main__":
    main()
