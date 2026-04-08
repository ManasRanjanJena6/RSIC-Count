"""
Demo script for RSIC-Count++ system.
Interactive demo for caption generation with attention visualization.
"""

import torch
import numpy as np
import argparse
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

from models.multitask_count_caption import MultiTaskCaptioner
# from models.transformer_count import TransformerCountCaptioner  # Not used currently
# from models.visualization import visualize_attention, visualize_attention_grid  # Not used currently
from utils.beam_search import beam_search
from preprocess.build_vocab import Vocabulary
from preprocess.extract_feats import FeatureExtractor


def load_model(checkpoint_path, model_type, vocab, device):
    """Load trained model from checkpoint"""
    print(f"Loading {model_type} model from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint.get('args', {})
    
    # Create model based on type
    if model_type == 'multitask':
        model = MultiTaskCaptioner(
            vocab_size=vocab.vocab_size,
            feat_dim=checkpoint.get('feat_dim', 1024),
            count_vec_size=checkpoint.get('count_vec_size', 8),
            embed_dim=checkpoint.get('embed_dim', 512),
            hidden_dim=checkpoint.get('hidden_dim', 512),
            att_dim=checkpoint.get('att_dim', 512),
            count_embed_dim=checkpoint.get('count_embed_dim', 128),
            dropout=checkpoint.get('dropout', 0.5)
        ).to(device)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'multitask'")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Model loaded successfully!")
    return model


def generate_caption(model, image_path, count_vec, vocab, device, 
                     use_beam_search=True, beam_size=3, max_len=50, model_type='lstm'):
    """
    Generate caption for an image.
    
    Args:
        model: Trained model
        image_path: Path to image
        count_vec: Count vector [count_vec_size]
        vocab: Vocabulary object
        device: Device
        use_beam_search: Use beam search
        beam_size: Beam size
        model_type: 'lstm', 'multitask', or 'transformer'
    
    Returns:
        caption: Generated caption
        alphas: Attention weights (if available)
    """
    # Extract features
    print("Extracting features...")
    # Use ConvNeXt Base to match the checkpoint (1024 dim)
    extractor = FeatureExtractor(device=device, backbone_name='convnext_base')
    att_feat, fc_feat = extractor.extract(image_path)
    
    # Convert to tensors
    att_feat = torch.tensor(att_feat, dtype=torch.float32).to(device)  # [1, 2048, 7, 7]
    fc_feat = torch.tensor(fc_feat, dtype=torch.float32).to(device)    # [1, 2048]
    count_vec = torch.tensor(count_vec, dtype=torch.float32).unsqueeze(0).to(device)  # [1, count_vec_size]
    
    print("Generating caption...")
    
    with torch.no_grad():
        if model_type == 'multitask':
            # Multi-task model: use beam search for better quality
            if use_beam_search:
                caption, score = beam_search(
                    model, att_feat, fc_feat, count_vec,
                    vocab.stoi, vocab.itos,
                    beam_size=beam_size, max_len=max_len, device=device
                )
                print(f"Beam search score: {score:.2f}")
                return caption, None
            else:
                caption_indices, pred_counts, alphas = model.generate_with_predicted_counts(
                    att_feat, fc_feat, vocab.stoi, max_len=max_len
                )
                
                # Convert to text
                caption = " ".join([vocab.itos.get(idx, "<unk>") for idx in caption_indices
                                  if vocab.itos.get(idx, "") not in ["<start>", "<end>", "<pad>"]])
                
                print(f"\nPredicted counts: {pred_counts.cpu().numpy()}")
                return caption, alphas
        
        elif model_type == 'transformer':
            if use_beam_search:
                caption = model.beam_search_generate(
                    att_feat, fc_feat, count_vec,
                    vocab.stoi, vocab.itos,
                    beam_size=beam_size
                )
                return caption, None
            else:
                caption_indices = model.generate(att_feat, fc_feat, count_vec, vocab.stoi)
                caption = " ".join([vocab.itos.get(idx, "<unk>") for idx in caption_indices
                                  if vocab.itos.get(idx, "") not in ["<start>", "<end>", "<pad>"]])
                return caption, None
        
        else:  # LSTM
            if use_beam_search:
                caption, score = beam_search(
                    model, att_feat, fc_feat, count_vec,
                    vocab.stoi, vocab.itos,
                    beam_size=beam_size, max_len=max_len, device=device
                )
                print(f"Beam search score: {score:.2f}")
                return caption, None
            else:
                caption_indices, alphas = model.generate(
                    att_feat, fc_feat, count_vec, vocab.stoi, max_len=max_len
                )
                caption = " ".join([vocab.itos.get(idx, "<unk>") for idx in caption_indices
                                  if vocab.itos.get(idx, "") not in ["<start>", "<end>", "<pad>"]])
                return caption, alphas


def main():
    parser = argparse.ArgumentParser(description="RSIC-Count++ Demo")
    
    # Model parameters
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--model_type', type=str, default='multitask',
                       choices=['lstm', 'multitask', 'transformer'])
    parser.add_argument('--vocab', type=str, default='data/vocab.pkl')
    
    # Input
    parser.add_argument('--image', type=str, required=True, help='Path to image')
    parser.add_argument('--counts', type=str, default=None,
                       help='Path to counts JSON or comma-separated count values')
    
    # Generation parameters
    parser.add_argument('--beam_search', action='store_true', help='Use beam search')
    parser.add_argument('--beam_size', type=int, default=5, help='Beam search size')
    parser.add_argument('--max_len', type=int, default=25, help='Maximum caption length')
    parser.add_argument('--device', type=str, default='cuda')
    
    # Visualization
    parser.add_argument('--visualize', action='store_true', help='Visualize attention')
    parser.add_argument('--save_viz', type=str, default=None, help='Save visualization')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load vocabulary
    print("Loading vocabulary...")
    vocab = Vocabulary.load(args.vocab)
    
    # Load model
    model = load_model(args.checkpoint, args.model_type, vocab, device)
    
    # Parse counts
    categories = ['aeroplane', 'bridge', 'buildings', 'container_yard', 
                  'ground', 'ship', 'solar_panel', 'storage_tank']
    
    if args.counts:
        if args.counts.endswith('.json'):
            import json
            with open(args.counts, 'r') as f:
                counts_data = json.load(f)
            # Get image name from path and look up counts
            img_name = Path(args.image).stem
            count_dict = counts_data.get(img_name, counts_data.get(f"{img_name}.png", {}))
            count_vec = np.array([count_dict.get(cat, 0) for cat in categories], dtype=np.float32)
        else:
            # Parse comma-separated values
            values = [float(x) for x in args.counts.split(',')]
            count_vec = np.array(values, dtype=np.float32)
            if len(count_vec) != 8:
                print(f"Warning: Expected 8 count values, got {len(count_vec)}. Padding/truncating to 8.")
                if len(count_vec) < 8:
                    count_vec = np.pad(count_vec, (0, 8 - len(count_vec)))
                else:
                    count_vec = count_vec[:8]
    else:
        # Zero counts (model will predict if multi-task)
        count_vec = np.zeros(8, dtype=np.float32)
    
    print(f"\n{'='*60}")
    print("INPUT COUNTS:")
    for cat, val in zip(categories, count_vec):
        if val > 0:
            print(f"  {cat:20s}: {int(val)}")
    print(f"{'='*60}")
    
    # Generate caption
    caption, alphas = generate_caption(
        model, args.image, count_vec, vocab, device,
        use_beam_search=args.beam_search,
        beam_size=args.beam_size,
        max_len=args.max_len,
        model_type=args.model_type
    )
    
    # Print results with better formatting
    print("\n" + "="*60)
    print("GENERATED CAPTION:")
    print("="*60)
    print(f"{caption}")
    print("="*60)
    
    # Extract features for count prediction
    print("\nExtracting features for count prediction...")
    extractor = FeatureExtractor(device=device, backbone_name='convnext_base')
    att_feat_np, fc_feat_np = extractor.extract(args.image)
    
    # Print count predictions if available
    if hasattr(model, 'predict_counts'):
        with torch.no_grad():
            att_feat_tensor = torch.tensor(att_feat_np, dtype=torch.float32).to(device)
            fc_feat_tensor = torch.tensor(fc_feat_np, dtype=torch.float32).to(device)
            pred_counts = model.predict_counts(att_feat_tensor, fc_feat_tensor)
            pred_counts = pred_counts.cpu().numpy()[0]
            
            print("\nPREDICTED COUNTS:")
            print("-"*60)
            for cat, true_val, pred_val in zip(categories, count_vec, pred_counts):
                diff = abs(pred_val - true_val)
                status = "✓" if diff < 0.5 else "✗"
                print(f"  {cat:20s}: True={int(true_val):2d}, Pred={pred_val:5.1f} {status}")
            print("="*60)
    
    print(f"\nCaption Length: {len(caption.split())} words")
    print("="*60)
    
    # Visualize attention
    if args.visualize and alphas is not None:
        print("\nVisualizing attention...")
        
        # Load caption indices for visualization
        caption_indices = [vocab.stoi.get(word, vocab.stoi['<unk>']) 
                          for word in caption.split()]
        
        if args.save_viz:
            visualize_attention_grid(
                args.image, alphas, caption_indices, vocab.itos,
                save_path=args.save_viz
            )
        else:
            visualize_attention_grid(
                args.image, alphas, caption_indices, vocab.itos
            )
    
    # Display image with caption
    img = Image.open(args.image)
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Caption: {caption}', fontsize=14, fontweight='bold', wrap=True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
