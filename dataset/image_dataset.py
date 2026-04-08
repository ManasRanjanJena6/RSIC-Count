"""
Image Dataset for End-to-End Training
Loads raw images instead of pre-extracted features
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    """Dataset that loads images directly for backbone fine-tuning"""
    
    def __init__(self, captions_file, counts_file, image_dir, vocab, 
                 split='train', max_len=25, transform=None):
        """
        Args:
            captions_file: Path to captions.json
            counts_file: Path to counts.json
            image_dir: Directory with images
            vocab: Vocabulary object
            split: 'train', 'val', or 'all'
            max_len: Maximum caption length
            transform: Torchvision transforms
        """
        self.image_dir = Path(image_dir)
        self.vocab = vocab
        self.max_len = max_len
        self.split = split
        
        # Default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        
        # Load captions
        with open(captions_file, 'r') as f:
            captions_data = json.load(f)
        
        # Load counts
        with open(counts_file, 'r') as f:
            counts_data = json.load(f)
        
        # Build count lookup
        counts_by_filename = {}
        if isinstance(counts_data, list):
            for item in counts_data:
                if 'filename' in item:
                    counts_by_filename[item['filename']] = item
        elif isinstance(counts_data, dict):
            counts_by_filename = counts_data
        
        # Parse samples
        self.samples = []
        
        if isinstance(captions_data, list):
            for item in captions_data:
                if 'filename' not in item:
                    continue
                
                filename = item['filename']
                img_id = filename.replace('.png', '').replace('.jpg', '')
                
                # Get caption
                caption = item.get('text_output', item.get('caption', ''))
                if not caption:
                    continue
                
                # Get counts
                count_dict = counts_by_filename.get(filename, counts_by_filename.get(img_id, {}))
                
                # Check image exists
                img_path = self.image_dir / filename
                if not img_path.exists():
                    # Try without extension
                    img_path = self.image_dir / f"{img_id}.png"
                    if not img_path.exists():
                        img_path = self.image_dir / f"{img_id}.jpg"
                        if not img_path.exists():
                            continue
                
                self.samples.append({
                    'img_path': str(img_path),
                    'img_id': img_id,
                    'caption': caption,
                    'count_vec': count_dict
                })
        
        # Apply stratified split
        categories = ['aeroplane', 'bridge', 'buildings', 'container_yard',
                      'ground', 'ship', 'solar_panel', 'storage_tank']
        
        if split != 'all':
            samples_by_class = {cat: [] for cat in categories}
            
            for sample in self.samples:
                count_dict = sample['count_vec']
                assigned = False
                for cat in categories:
                    if isinstance(count_dict, dict):
                        if count_dict.get(cat, 0) > 0:
                            samples_by_class[cat].append(sample)
                            assigned = True
                            break
                    elif isinstance(count_dict, (list, np.ndarray)):
                        idx = categories.index(cat)
                        if idx < len(count_dict) and count_dict[idx] > 0:
                            samples_by_class[cat].append(sample)
                            assigned = True
                            break
                
                if not assigned:
                    samples_by_class[categories[0]].append(sample)
            
            # Split each class 70/30
            import random
            split_samples = []
            for cat in categories:
                cat_samples = samples_by_class[cat]
                if len(cat_samples) == 0:
                    continue
                
                random.shuffle(cat_samples)
                split_idx = int(0.7 * len(cat_samples))
                
                if split == 'train':
                    split_samples.extend(cat_samples[:split_idx])
                else:
                    split_samples.extend(cat_samples[split_idx:])
            
            self.samples = split_samples
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        img = Image.open(sample['img_path']).convert('RGB')
        img_tensor = self.transform(img)
        
        # Process caption
        caption = sample['caption']
        caption_indices = [self.vocab.stoi['<start>']]
        caption_indices.extend(self.vocab.numericalize(caption))
        caption_indices.append(self.vocab.stoi['<end>'])
        caption_indices = caption_indices[:self.max_len]
        
        # Convert count dict to vector
        count_vec = self._count_dict_to_vector(sample['count_vec'])
        
        return {
            'images': img_tensor,
            'captions': torch.tensor(caption_indices, dtype=torch.long),
            'count_vecs': torch.tensor(count_vec, dtype=torch.float32),
            'img_ids': sample['img_id']
        }
    
    def _count_dict_to_vector(self, count_dict, num_categories=8):
        """Convert count dictionary to fixed-size vector"""
        categories = ['aeroplane', 'bridge', 'buildings', 'container_yard',
                      'ground', 'ship', 'solar_panel', 'storage_tank']
        
        count_vec = np.zeros(num_categories, dtype=np.float32)
        for i, cat in enumerate(categories):
            if isinstance(count_dict, dict):
                count_vec[i] = count_dict.get(cat, 0)
            elif isinstance(count_dict, (list, np.ndarray)) and i < len(count_dict):
                count_vec[i] = count_dict[i]
        
        return count_vec


def collate_fn(batch):
    """Custom collate for batching"""
    images = torch.stack([item['images'] for item in batch])
    
    # Pad captions
    captions = [item['captions'] for item in batch]
    max_len = max(len(c) for c in captions)
    padded_captions = torch.zeros(len(captions), max_len, dtype=torch.long)
    for i, c in enumerate(captions):
        padded_captions[i, :len(c)] = c
    
    count_vecs = torch.stack([item['count_vecs'] for item in batch])
    img_ids = [item['img_ids'] for item in batch]
    
    return {
        'images': images,
        'captions': padded_captions,
        'count_vecs': count_vecs,
        'img_ids': img_ids
    }
