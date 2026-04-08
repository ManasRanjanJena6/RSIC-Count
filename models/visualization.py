"""
Attention Visualization for Interpretability.
Generates heatmaps showing where the model focuses during caption generation.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def visualize_attention(image_path, alphas, caption_tokens, itos, save_path=None, figsize=(20, 4)):
    """
    Visualize attention weights over the image for each generated word.
    
    Args:
        image_path: Path to the original image
        alphas: List or array of attention weights [seq_len, 49]
        caption_tokens: List of token indices
        itos: Index to word mapping
        save_path: Optional path to save the visualization
        figsize: Figure size
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    
    # Convert alphas to numpy if tensor
    if isinstance(alphas, torch.Tensor):
        alphas = alphas.cpu().numpy()
    elif isinstance(alphas, list):
        alphas = np.array(alphas)
    
    # Get caption words (exclude special tokens)
    words = []
    valid_alphas = []
    
    for i, token_id in enumerate(caption_tokens):
        if i >= len(alphas):
            break
        word = itos.get(token_id, "<unk>")
        if word not in ["<start>", "<end>", "<pad>"]:
            words.append(word)
            valid_alphas.append(alphas[i])
    
    n_words = len(words)
    if n_words == 0:
        print("No valid words to visualize")
        return
    
    # Create subplot grid
    n_cols = min(5, n_words)
    n_rows = (n_words + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        if idx < n_words:
            # Get attention weights and reshape to 7x7 grid
            attn = valid_alphas[idx].reshape(7, 7)
            
            # Resize attention to image size
            attn_resized = resize_attention(attn, img_array.shape[:2])
            
            # Display image
            ax.imshow(img_array)
            
            # Overlay attention heatmap
            im = ax.imshow(attn_resized, cmap='jet', alpha=0.6, interpolation='bilinear')
            
            # Set title
            ax.set_title(words[idx], fontsize=12, fontweight='bold')
            ax.axis('off')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Attention visualization saved to {save_path}")
    
    plt.show()


def resize_attention(attention_map, target_size):
    """
    Resize attention map from 7x7 to target image size.
    
    Args:
        attention_map: [7, 7] attention weights
        target_size: (height, width) target size
    
    Returns:
        resized_map: [H, W] resized attention
    """
    # Convert to tensor
    attn_tensor = torch.tensor(attention_map).unsqueeze(0).unsqueeze(0)  # [1, 1, 7, 7]
    
    # Resize using bilinear interpolation
    resized = F.interpolate(attn_tensor, size=target_size, 
                            mode='bilinear', align_corners=False)
    
    return resized.squeeze().numpy()


def visualize_attention_grid(image_path, alphas, caption_tokens, itos, 
                             save_path=None, grid_size=(7, 7)):
    """
    Visualize attention as a grid overlay on the image.
    
    Args:
        image_path: Path to image
        alphas: Attention weights
        caption_tokens: Token indices
        itos: Index to word mapping
        save_path: Save path
        grid_size: Size of attention grid (7, 7) for ResNet
    """
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    
    if isinstance(alphas, torch.Tensor):
        alphas = alphas.cpu().numpy()
    
    # Average attention across all time steps
    avg_attention = np.mean(alphas, axis=0).reshape(grid_size)
    
    # Normalize
    avg_attention = (avg_attention - avg_attention.min()) / (avg_attention.max() - avg_attention.min())
    
    # Resize to image size
    attn_resized = resize_attention(avg_attention, img_array.shape[:2])
    
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    ax1.imshow(img_array)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Attention heatmap
    im2 = ax2.imshow(attn_resized, cmap='jet')
    ax2.set_title('Average Attention Heatmap')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # Overlay
    ax3.imshow(img_array)
    ax3.imshow(attn_resized, cmap='jet', alpha=0.5)
    
    # Generate caption text
    caption = " ".join([itos.get(t, "<unk>") for t in caption_tokens 
                       if itos.get(t, "") not in ["<start>", "<end>", "<pad>"]])
    ax3.set_title(f'Attention Overlay\n"{caption}"', fontsize=10)
    ax3.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Attention grid saved to {save_path}")
    
    plt.show()


def save_attention_video(image_path, alphas, caption_tokens, itos, output_path, fps=2):
    """
    Create a video/GIF showing attention evolution during caption generation.
    
    Args:
        image_path: Path to image
        alphas: Attention weights [seq_len, 49]
        caption_tokens: Token indices
        itos: Index to word mapping
        output_path: Output video/gif path
        fps: Frames per second
    """
    try:
        import imageio
    except ImportError:
        print("imageio not installed. Install with: pip install imageio")
        return
    
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    
    if isinstance(alphas, torch.Tensor):
        alphas = alphas.cpu().numpy()
    
    frames = []
    
    for i, token_id in enumerate(caption_tokens):
        if i >= len(alphas):
            break
        
        word = itos.get(token_id, "<unk>")
        if word in ["<start>", "<end>", "<pad>"]:
            continue
        
        # Create frame
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Get attention
        attn = alphas[i].reshape(7, 7)
        attn_resized = resize_attention(attn, img_array.shape[:2])
        
        # Plot
        ax.imshow(img_array)
        ax.imshow(attn_resized, cmap='jet', alpha=0.6)
        ax.set_title(f'Word: "{word}"', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        # Convert to image
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        
        plt.close(fig)
    
    # Save as GIF
    imageio.mimsave(output_path, frames, fps=fps, loop=0)
    print(f"Attention video saved to {output_path}")
