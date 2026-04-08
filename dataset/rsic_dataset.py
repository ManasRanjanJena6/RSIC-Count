"""
PyTorch Dataset for RSIC-Count with count embeddings.
Loads pre-extracted features, captions, and object counts.
"""

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import json
from pathlib import Path
import random
import torchvision.transforms as transforms
from PIL import Image
import nltk
from nltk.corpus import wordnet
import warnings

# Download required NLTK data
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')


class RSICDataset(Dataset):
    def __init__(self, captions_file, counts_file, fc_dir, att_dir, vocab, split='train', max_len=25, augmentation_config=None):
        """
        Args:
            captions_file: Path to captions.json
            counts_file: Path to counts.json
            fc_dir: Directory with FC features (.npy files)
            att_dir: Directory with attention features (.npy files)
            vocab: Vocabulary object
            split: 'train', 'val', or 'test'
            max_len: Maximum caption length
            augmentation_config: Dict with augmentation settings
        """
        self.fc_dir = Path(fc_dir)
        self.att_dir = Path(att_dir)
        self.vocab = vocab
        self.max_len = max_len
        self.split = split
        self.augmentation_config = augmentation_config or {}
        
        # Load captions
        with open(captions_file, 'r') as f:
            captions_data = json.load(f)
        
        # Load counts
        with open(counts_file, 'r') as f:
            counts_data = json.load(f)
        
        # Parse dataset - handle both dictionary and list formats
        self.samples = []
        
        if isinstance(captions_data, dict):
            # Check for {"annotations": [...]} format
            if 'annotations' in captions_data and isinstance(captions_data['annotations'], list):
                captions_data = captions_data['annotations']
            else:
                # Format: {"img_id": ["caption1", "caption2"]}
                for img_id, captions in captions_data.items():
                    count_vec = counts_data.get(img_id, {})
                    fc_path = self.fc_dir / f"{img_id}.npy"
                    att_path = self.att_dir / f"{img_id}.npy"
                    
                    if not fc_path.exists() or not att_path.exists():
                        continue
                    
                    caption_list = captions if isinstance(captions, list) else [captions]
                    for caption in caption_list:
                        if isinstance(caption, str):
                            self.samples.append({
                                'img_id': img_id,
                                'caption': caption,
                                'count_vec': count_vec
                            })
                # If we found samples, we're done
                if self.samples:
                    print(f"Loaded {len(self.samples)} samples for {split} split")
                    return
        
        if isinstance(captions_data, list):
            # Format: [{"filename": "...", "text_output": "..."}]
            # Build count lookup by filename
            counts_by_filename = {}
            if isinstance(counts_data, list):
                for item in counts_data:
                    if 'filename' in item:
                        counts_by_filename[item['filename']] = item
            elif isinstance(counts_data, dict):
                counts_by_filename = counts_data
            
            for item in captions_data:
                if 'filename' not in item:
                    continue
                
                filename = item['filename']
                img_id = filename.replace('.png', '').replace('.jpg', '')
                
                # Get caption
                if 'text_output' in item:
                    caption = item['text_output']
                elif 'caption' in item:
                    caption = item['caption']
                else:
                    continue
                
                # Get counts
                count_vec = counts_by_filename.get(filename, counts_by_filename.get(img_id, {}))
                
                # Check if features exist
                fc_path = self.fc_dir / f"{img_id}.npy"
                att_path = self.att_dir / f"{img_id}.npy"
                
                if not fc_path.exists() or not att_path.exists():
                    continue
                
                self.samples.append({
                    'img_id': img_id,
                    'caption': caption,
                    'count_vec': count_vec
                })
        
        # Apply stratified train/val split (70/30) - split within each class
        categories = ['aeroplane', 'bridge', 'buildings', 'container_yard', 
                      'ground', 'ship', 'solar_panel', 'storage_tank']
        
        if split != 'all':
            # Group samples by class (based on which object has non-zero count)
            samples_by_class = {cat: [] for cat in categories}
            
            for sample in self.samples:
                count_dict = sample['count_vec']
                # Find which class this sample belongs to
                assigned = False
                for cat in categories:
                    if isinstance(count_dict, dict):
                        if count_dict.get(cat, 0) > 0:
                            samples_by_class[cat].append(sample)
                            assigned = True
                            break
                    elif isinstance(count_dict, (list, np.ndarray)):
                        # If count_vec is already a vector
                        idx = categories.index(cat)
                        if idx < len(count_dict) and count_dict[idx] > 0:
                            samples_by_class[cat].append(sample)
                            assigned = True
                            break
                
                # If no dominant class found, assign to first category
                if not assigned:
                    samples_by_class[categories[0]].append(sample)
            
            # Now split each class 70/30
            split_samples = []
            for cat in categories:
                cat_samples = samples_by_class[cat]
                if len(cat_samples) == 0:
                    continue
                    
                # Shuffle within class for randomness
                random.shuffle(cat_samples)
                
                split_idx = int(0.7 * len(cat_samples))
                
                if split == 'train':
                    split_samples.extend(cat_samples[:split_idx])
                else:  # val
                    split_samples.extend(cat_samples[split_idx:])
            
            self.samples = split_samples
        
        # If split='all', keep all samples
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_id = sample['img_id']
        
        # Load features
        fc_feat = np.load(self.fc_dir / f"{img_id}.npy").squeeze()
        att_feat = np.load(self.att_dir / f"{img_id}.npy").squeeze()
        
        # Apply caption augmentation for training
        caption = sample['caption']
        if self.split == 'train' and self.augmentation_config.get('enabled', False):
            caption = self._augment_caption(caption)
        
        # Numericalize caption
        caption_indices = [self.vocab.stoi['<start>']]
        caption_indices.extend(self.vocab.numericalize(caption))
        caption_indices.append(self.vocab.stoi['<end>'])
        
        # Truncate if too long
        caption_indices = caption_indices[:self.max_len]
        
        # Convert count dict to vector
        count_vec = self._count_dict_to_vector(sample['count_vec'])
        
        return {
            'fc': torch.tensor(fc_feat, dtype=torch.float32),
            'att': torch.tensor(att_feat, dtype=torch.float32),
            'caption': torch.tensor(caption_indices, dtype=torch.long),
            'count_vec': torch.tensor(count_vec, dtype=torch.float32),
            'img_id': img_id,
            'caption_text': caption
        }
    
    def _count_dict_to_vector(self, count_dict, num_categories=8):
        """Convert count dictionary to fixed-size vector"""
        categories = ['aeroplane', 'bridge', 'buildings', 'container_yard', 
                      'ground', 'ship', 'solar_panel', 'storage_tank']
        
        count_vec = np.zeros(num_categories, dtype=np.float32)
        for i, cat in enumerate(categories):
            count_vec[i] = count_dict.get(cat, 0)
        
        return count_vec
    
    def _augment_caption(self, caption):
        """Apply caption augmentation techniques"""
        if not self.augmentation_config.get('caption_augmentation', {}).get('enabled', False):
            return caption
        
        # Synonym replacement
        if self.augmentation_config.get('caption_augmentation', {}).get('synonym_replace', 0) > random.random():
            caption = self._replace_synonyms(caption)
        
        # Paraphrasing (simple word reordering)
        if self.augmentation_config.get('caption_augmentation', {}).get('paraphrase', False) and random.random() < 0.3:
            caption = self._simple_paraphrase(caption)
        
        return caption
    
    def _replace_synonyms(self, caption):
        """Replace some words with synonyms"""
        words = caption.split()
        for i, word in enumerate(words):
            if random.random() < 0.1:  # 10% chance per word
                try:
                    synonyms = wordnet.synsets(word)
                    if synonyms and len(synonyms[0].lemmas()) > 1:
                        synonym = random.choice(synonyms[0].lemmas()[1:]).name()
                        if synonym != word:
                            words[i] = synonym
                except:
                    pass  # Skip if no synonyms found
        return ' '.join(words)
    
    def _simple_paraphrase(self, caption):
        """Simple paraphrasing by reordering words"""
        words = caption.split()
        if len(words) > 3:
            # Swap two random words
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        return ' '.join(words)


def collate_fn(batch):
    """Custom collate function for batching"""
    fc_feats = torch.stack([item['fc'] for item in batch])
    att_feats = torch.stack([item['att'] for item in batch])
    count_vecs = torch.stack([item['count_vec'] for item in batch])
    
    # Pad captions to same length
    captions = [item['caption'] for item in batch]
    captions_padded = pad_sequence(captions, batch_first=True, padding_value=0)
    
    img_ids = [item['img_id'] for item in batch]
    caption_texts = [item['caption_text'] for item in batch]
    
    return {
        'fc': fc_feats,
        'att': att_feats,
        'captions': captions_padded,
        'count_vecs': count_vecs,
        'img_ids': img_ids,
        'caption_texts': caption_texts
    }
