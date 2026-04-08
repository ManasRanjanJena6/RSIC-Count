"""
Optimized Multi-Task Training for RTX 3050 6GB
Jointly learns caption generation + object count prediction
"""

import sys
sys.path.append('.')

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
from tqdm import tqdm
import yaml

from preprocess.build_vocab import Vocabulary
from dataset.rsic_dataset import RSICDataset, collate_fn
from models.multitask_count_caption import MultiTaskCaptioner, CountAwareLoss

# Mixed precision for RTX 3050
from torch.cuda.amp import autocast, GradScaler

def train_epoch(model, dataloader, criterion, optimizer, device, scaler):
    model.train()
    total_loss = 0
    total_caption_loss = 0
    total_count_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        fc_feats = batch['fc'].to(device)
        att_feats = batch['att'].to(device)
        captions = batch['captions'].to(device)
        count_vecs = batch['count_vecs'].to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        with autocast():
            caption_outputs, alphas, count_preds = model(
                att_feats, fc_feats, captions[:, :-1], count_vecs
            )
            
            # Compute multi-task loss
            loss, caption_loss, count_loss = criterion(
                caption_outputs, count_preds, captions, count_vecs
            )
        
        # Mixed precision backward
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        total_caption_loss += caption_loss.item()
        total_count_loss += count_loss.item()
    
    n = len(dataloader)
    return total_loss / n, total_caption_loss / n, total_count_loss / n

@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_caption_loss = 0
    total_count_loss = 0
    
    for batch in tqdm(dataloader, desc="Validation"):
        fc_feats = batch['fc'].to(device)
        att_feats = batch['att'].to(device)
        captions = batch['captions'].to(device)
        count_vecs = batch['count_vecs'].to(device)
        
        with autocast():
            caption_outputs, alphas, count_preds = model(
                att_feats, fc_feats, captions[:, :-1], count_vecs
            )
            
            loss, caption_loss, count_loss = criterion(
                caption_outputs, count_preds, captions, count_vecs
            )
        
        total_loss += loss.item()
        total_caption_loss += caption_loss.item()
        total_count_loss += count_loss.item()
    
    n = len(dataloader)
    return total_loss / n, total_caption_loss / n, total_count_loss / n

def main():
    parser = argparse.ArgumentParser(description='Multi-Task Training (RTX 3050 Optimized)')
    
    # Config file
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    
    # Override options (optional)
    parser.add_argument('--captions', default=None)
    parser.add_argument('--counts', default=None)
    parser.add_argument('--fc_dir', default=None)
    parser.add_argument('--att_dir', default=None)
    parser.add_argument('--vocab', default=None)
    
    # Training overrides
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--patience', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    
    # Model overrides
    parser.add_argument('--embed_dim', type=int, default=None)
    parser.add_argument('--hidden_dim', type=int, default=None)
    parser.add_argument('--att_dim', type=int, default=None)
    parser.add_argument('--count_embed_dim', type=int, default=None)
    parser.add_argument('--dropout', type=float, default=None)
    
    # Multi-task specific overrides
    parser.add_argument('--caption_weight', type=float, default=None)
    parser.add_argument('--count_weight', type=float, default=None)
    parser.add_argument('--count_loss_type', default=None)
    
    # Checkpoint override
    parser.add_argument('--save_dir', default=None)
    parser.add_argument('--resume', default=None)
    parser.add_argument('--device', default=None)
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    def get_config_value(section, key, default=None, arg_value=None):
        if arg_value is not None:
            return arg_value
        elif section in config and key in config[section]:
            return config[section][key]
        return default
    
    # Data paths
    captions = get_config_value('data', 'captions', 'data/captions.json', args.captions)
    counts = get_config_value('data', 'counts', 'data/counts.json', args.counts)
    fc_dir = get_config_value('data', 'fc_features', 'data/fc_features', args.fc_dir)
    att_dir = get_config_value('data', 'att_features', 'data/att_features', args.att_dir)
    vocab_path = get_config_value('data', 'vocab', 'data/vocab.pkl', args.vocab)
    
    # Training parameters
    batch_size = get_config_value('training', 'batch_size', 12, args.batch_size)
    epochs = get_config_value('training', 'epochs', 130, args.epochs)
    learning_rate = get_config_value('training', 'learning_rate', 0.0001, args.lr)
    scheduler_config = get_config_value('training', 'scheduler', {})
    patience = get_config_value('training', 'patience', 8, args.patience) if scheduler_config else 8
    num_workers = get_config_value('system', 'num_workers', 2, args.num_workers)
    
    # Model parameters
    embed_dim = get_config_value('model', 'embed_dim', 512, args.embed_dim)
    hidden_dim = get_config_value('model', 'hidden_dim', 512, args.hidden_dim)
    att_dim = get_config_value('model', 'att_dim', 512, args.att_dim)
    count_embed_dim = get_config_value('model', 'count_embed_dim', 128, args.count_embed_dim)
    count_vec_size = get_config_value('model', 'count_vec_size', 8)
    dropout = get_config_value('model', 'dropout', 0.5, args.dropout)
    
    # Backbone feature dimension and name
    feat_dim = get_config_value('backbone', 'feature_dim', 2048, None)
    backbone_name = get_config_value('backbone', 'name', 'resnet50', None)
    
    # Multi-task parameters
    multitask_config = get_config_value('training', 'multitask', {})
    if multitask_config:
        caption_weight = multitask_config.get('caption_weight', 1.0)
        count_weight = multitask_config.get('count_weight', 1.0)
        count_loss_type = multitask_config.get('count_loss_type', 'mse')
    else:
        caption_weight = 1.0
        count_weight = 1.0
        count_loss_type = 'mse'
    
    # Augmentation config
    augmentation_config = get_config_value('training', 'augmentation', {})
    
    # Checkpoint and device
    save_dir = get_config_value('training', 'save_dir', 'checkpoints', args.save_dir) or 'checkpoints'
    device = torch.device(get_config_value('system', 'device', 'cuda', args.device) if torch.cuda.is_available() else 'cpu')
    print("="*60)
    print("MULTI-TASK TRAINING (RTX 3050 OPTIMIZED)")
    print("="*60)
    print(f"Device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    print(f"\nMulti-Task Configuration:")
    print(f"  Backbone: {backbone_name}")
    print(f"  Feature Dim: {feat_dim}")
    print(f"  Caption Weight: {caption_weight}")
    print(f"  Count Weight: {count_weight}")
    print(f"  Count Loss: {count_loss_type}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Epochs: {epochs}")
    print(f"  Augmentation: {augmentation_config.get('enabled', False)}")
    print(f"  Mixed Precision: Enabled")
    print("="*60)
    
    # Create save directory
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Load vocabulary
    print("\nLoading vocabulary...")
    vocab = Vocabulary.load(vocab_path)
    print(f"Vocabulary: {vocab.vocab_size} tokens")
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = RSICDataset(
        captions, counts,
        fc_dir, att_dir,
        vocab, split='train',
        augmentation_config=augmentation_config if augmentation_config.get('enabled', False) else None
    )
    val_dataset = RSICDataset(
        captions, counts,
        fc_dir, att_dir,
        vocab, split='val'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Create multi-task model
    print("Creating multi-task model...")
    model = MultiTaskCaptioner(
        vocab_size=vocab.vocab_size,
        feat_dim=feat_dim,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        att_dim=att_dim,
        count_embed_dim=count_embed_dim,
        count_vec_size=count_vec_size,
        dropout=dropout
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Multi-task loss
    criterion = CountAwareLoss(
        caption_weight=caption_weight,
        count_weight=count_weight,
        count_loss_type=count_loss_type
    )
    
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience)
    scaler = GradScaler()
    
    # Resume
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    print(f"\nStarting training...")
    print("="*60)
    
    patience_counter = 0
    
    for epoch in range(start_epoch, epochs):
        print(f"\n📊 Epoch {epoch+1}/{epochs}")
        
        # Train
        train_loss, train_cap_loss, train_cnt_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        
        # Validate
        val_loss, val_cap_loss, val_cnt_loss = validate(
            model, val_loader, criterion, device
        )
        
        # Print metrics
        print(f"\n  Train:")
        print(f"    Total Loss:   {train_loss:.4f}")
        print(f"    Caption Loss: {train_cap_loss:.4f}")
        print(f"    Count Loss:   {train_cnt_loss:.4f}")
        print(f"  Validation:")
        print(f"    Total Loss:   {val_loss:.4f}")
        print(f"    Caption Loss: {val_cap_loss:.4f}")
        print(f"    Count Loss:   {val_cnt_loss:.4f}")
        
        # Learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'caption_loss': val_cap_loss,
                'count_loss': val_cnt_loss,
                'vocab_size': vocab.vocab_size,
                'feat_dim': feat_dim,
                'backbone': backbone_name,
                'embed_dim': embed_dim,
                'hidden_dim': hidden_dim,
                'att_dim': att_dim,
                'count_embed_dim': count_embed_dim,
                'dropout': dropout,
                'caption_weight': caption_weight,
                'count_weight': count_weight
            }, Path(save_dir) / 'best.pth')
            print("  ✅ Saved best model")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{patience}")
        
        # Save every N epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'vocab_size': vocab.vocab_size,
                'feat_dim': feat_dim,
                'backbone': backbone_name,
                'embed_dim': embed_dim,
                'hidden_dim': hidden_dim,
                'att_dim': att_dim,
                'count_embed_dim': count_embed_dim,
                'dropout': dropout,
                'caption_weight': caption_weight,
                'count_weight': count_weight
            }, Path(save_dir) / f'epoch_{epoch+1}.pth')
            print(f"  💾 Saved checkpoint epoch_{epoch+1}")
        
        # Save last
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'vocab_size': vocab.vocab_size,
            'feat_dim': feat_dim,
            'backbone': backbone_name,
            'embed_dim': embed_dim,
            'hidden_dim': hidden_dim,
            'att_dim': att_dim,
            'count_embed_dim': count_embed_dim,
            'dropout': dropout,
            'caption_weight': caption_weight,
            'count_weight': count_weight
        }, Path(save_dir) / 'last.pth')
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n⚠️ Early stopping at epoch {epoch+1}")
            break
        
        # Cleanup
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"Best model: {Path(save_dir) / 'best.pth'}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print("="*60)

if __name__ == '__main__':
    main()
