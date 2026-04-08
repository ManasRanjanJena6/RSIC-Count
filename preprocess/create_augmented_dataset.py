"""
Generate augmented captions.json and counts.json to match augmented features.
Creates 6 versions per image: original, hflip, vflip, rot90, rot180, rot270
"""

import json
import argparse
from pathlib import Path
import shutil

def create_augmented_dataset(captions_path, counts_path, output_dir):
    """Create augmented versions of captions and counts"""
    
    # Load original data
    with open(captions_path, 'r') as f:
        captions_data = json.load(f)
    
    with open(counts_path, 'r') as f:
        counts_data = json.load(f)
    
    # Augmentation suffixes
    aug_suffixes = ['', '_hflip', '_vflip', '_rot90', '_rot180', '_rot270']
    
    # Build lookup by image_id
    captions_by_img = {}
    if isinstance(captions_data, list):
        for item in captions_data:
            if 'filename' in item:
                img_id = item['filename'].replace('.png', '').replace('.jpg', '')
                if img_id not in captions_by_img:
                    captions_by_img[img_id] = []
                captions_by_img[img_id].append(item)
    
    # Create augmented captions
    augmented_captions = []
    augmented_counts = {}
    
    for img_id, items in captions_by_img.items():
        # Get count for this image
        count_dict = counts_data.get(img_id, counts_data.get(f"{img_id}.png", {}))
        
        for suffix in aug_suffixes:
            aug_id = f"{img_id}{suffix}"
            
            # Add captions for augmented version
            for item in items:
                aug_item = {
                    'filename': item.get('filename', f"{img_id}.png"),  # Keep original filename
                    'caption': item.get('caption', item.get('text_output', '')),
                    'augmented_id': aug_id  # Track which augmentation this is
                }
                augmented_captions.append(aug_item)
            
            # Add count for augmented version
            augmented_counts[aug_id] = count_dict
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save augmented files
    aug_captions_path = Path(output_dir) / 'captions_augmented.json'
    aug_counts_path = Path(output_dir) / 'counts_augmented.json'
    
    with open(aug_captions_path, 'w') as f:
        json.dump(augmented_captions, f, indent=2)
    
    with open(aug_counts_path, 'w') as f:
        json.dump(augmented_counts, f, indent=2)
    
    # Print stats
    print(f"\n{'='*60}")
    print(f"Augmented Dataset Created")
    print(f"{'='*60}")
    print(f"Original images: {len(captions_by_img)}")
    print(f"Augmentation types: {len(aug_suffixes)}")
    print(f"Total captions: {len(augmented_captions)}")
    print(f"Total count entries: {len(augmented_counts)}")
    print(f"Captions saved: {aug_captions_path}")
    print(f"Counts saved: {aug_counts_path}")
    print(f"{'='*60}")
    
    return aug_captions_path, aug_counts_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create augmented dataset files')
    parser.add_argument('--captions', type=str, default='data/captions.json',
                        help='Path to captions.json')
    parser.add_argument('--counts', type=str, default='data/counts.json',
                        help='Path to counts.json')
    parser.add_argument('--output', type=str, default='data',
                        help='Output directory for augmented files')
    
    args = parser.parse_args()
    create_augmented_dataset(args.captions, args.counts, args.output)
