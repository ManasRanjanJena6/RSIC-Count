"""
Multi-Task Learning: Joint Count Prediction + Caption Generation.
Learns to predict object counts and generate captions simultaneously,
improving quantitative grounding and caption quality.
"""

import torch
import torch.nn as nn
from models.att_lstm_count import AttLSTMCount, SoftAttention


class MultiTaskCaptioner(nn.Module):
    """
    Multi-task model that jointly predicts:
    1. Captions (with attention and count embeddings)
    2. Object counts (from visual features)
    """
    def __init__(self, vocab_size, count_vec_size, feat_dim=2048,
                 embed_dim=512, hidden_dim=512, att_dim=512,
                 count_embed_dim=128, dropout=0.5):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.count_vec_size = count_vec_size
        self.hidden_dim = hidden_dim
        self.feat_dim = feat_dim
        
        # Caption generation model (baseline LSTM with attention)
        self.captioner = AttLSTMCount(
            vocab_size=vocab_size,
            count_vec_size=count_vec_size,
            feat_dim=feat_dim,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            att_dim=att_dim,
            count_embed_dim=count_embed_dim,
            dropout=dropout
        )
        
        # Enhanced count prediction head with attention mechanism
        # This forces the model to actually LOOK at visual features
        self.count_attention = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 7*7),  # Spatial attention weights
            nn.Softmax(dim=-1)
        )
        
        # Deeper count predictor with residual connections
        self.count_predictor = nn.Sequential(
            nn.Linear(feat_dim + 512, 1024),  # Concatenate attended spatial features
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout/2),
            nn.Linear(256, count_vec_size)
        )
        
        # Spatial feature compressor
        self.spatial_compress = nn.Sequential(
            nn.Conv2d(feat_dim, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(7),  # Keep 7x7 spatial
        )
        
    def forward(self, att_feats, fc_feats, captions, count_vecs, 
                teacher_forcing=True, use_spatial_count=False):
        """
        Multi-task forward pass.
        
        Args:
            att_feats: [B, 2048, 7, 7] - attention features
            fc_feats: [B, 2048] - FC features
            captions: [B, max_len] - caption tokens
            count_vecs: [B, count_vec_size] - ground truth counts (for training)
            teacher_forcing: Use ground truth tokens
            use_spatial_count: Use spatial features for count prediction
        
        Returns:
            caption_outputs: [B, max_len-1, vocab_size] - caption logits
            alphas: [B, max_len-1, 49] - attention weights
            predicted_counts: [B, count_vec_size] - predicted counts
        """
        # Caption generation
        caption_outputs, alphas = self.captioner(
            att_feats, fc_feats, captions, count_vecs, teacher_forcing
        )
        
        # Enhanced count prediction with spatial attention
        batch_size = fc_feats.size(0)
        
        # Get attention weights from FC features
        att_weights = self.count_attention(fc_feats)  # [B, 49]
        
        # Compress spatial features
        spatial_compressed = self.spatial_compress(att_feats)  # [B, 512, 7, 7]
        spatial_flat = spatial_compressed.view(batch_size, 512, -1)  # [B, 512, 49]
        
        # Apply attention to spatial features
        att_weights_expanded = att_weights.unsqueeze(1)  # [B, 1, 49]
        attended_spatial = torch.bmm(att_weights_expanded, spatial_flat.transpose(1, 2))  # [B, 1, 512]
        attended_spatial = attended_spatial.squeeze(1)  # [B, 512]
        
        # Combine FC features with attended spatial features
        combined_feats = torch.cat([fc_feats, attended_spatial], dim=1)  # [B, feat_dim + 512]
        
        # Predict counts
        predicted_counts = self.count_predictor(combined_feats)
        
        return caption_outputs, alphas, predicted_counts
    
    def generate_with_predicted_counts(self, att_feats, fc_feats, stoi, max_len=50,
                                       no_repeat_ngram_size=3, repetition_penalty=1.2, temperature=1.0):
        """
        Generate caption using predicted counts (zero-shot counting).
        
        Args:
            att_feats: [1, 2048, 7, 7]
            fc_feats: [1, 2048]
            stoi: word to index dict
            max_len: max length (increased to 50 for complete captions)
        
        Returns:
            caption: List of token indices
            predicted_counts: Predicted count vector
            alphas: Attention weights
        """
        self.eval()
        
        with torch.no_grad():
            batch_size = fc_feats.size(0)
            
            # Enhanced count prediction with spatial attention
            att_weights = self.count_attention(fc_feats)  # [B, 49]
            spatial_compressed = self.spatial_compress(att_feats)  # [B, 512, 7, 7]
            spatial_flat = spatial_compressed.view(batch_size, 512, -1)  # [B, 512, 49]
            
            att_weights_expanded = att_weights.unsqueeze(1)  # [B, 1, 49]
            attended_spatial = torch.bmm(att_weights_expanded, spatial_flat.transpose(1, 2))
            attended_spatial = attended_spatial.squeeze(1)  # [B, 512]
            
            combined_feats = torch.cat([fc_feats, attended_spatial], dim=1)
            predicted_counts = self.count_predictor(combined_feats)  # [1, count_vec_size]
            
            # Generate caption using predicted counts
            caption, alphas = self.captioner.generate(
                att_feats, fc_feats, predicted_counts, stoi, max_len,
                no_repeat_ngram_size=no_repeat_ngram_size,
                repetition_penalty=repetition_penalty,
                temperature=temperature
            )
            
            return caption, predicted_counts, alphas


class MultiTaskTransformerCaptioner(nn.Module):
    """
    Multi-task Transformer variant.
    """
    def __init__(self, vocab_size, count_vec_size, feat_dim=2048,
                 d_model=512, nhead=8, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.1,
                 count_embed_dim=128, max_len=100):
        super().__init__()
        
        from models.transformer_count import TransformerCountCaptioner
        
        self.feat_dim = feat_dim
        
        # Caption generation model
        self.captioner = TransformerCountCaptioner(
            vocab_size=vocab_size,
            count_vec_size=count_vec_size,
            feat_dim=feat_dim,
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            count_embed_dim=count_embed_dim,
            max_len=max_len
        )
        
        # Count prediction head
        self.count_predictor = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, count_vec_size)
        )
    
    def forward(self, att_feats, fc_feats, captions, count_vecs, teacher_forcing=True):
        """Forward pass for Transformer multi-task model"""
        # Caption generation
        caption_outputs = self.captioner(att_feats, fc_feats, captions, count_vecs, teacher_forcing)
        
        # Count prediction
        predicted_counts = self.count_predictor(fc_feats)
        
        return caption_outputs, predicted_counts
    
    def generate_with_predicted_counts(self, att_feats, fc_feats, stoi, max_len=25):
        """Generate with predicted counts"""
        self.eval()
        
        with torch.no_grad():
            predicted_counts = self.count_predictor(fc_feats)
            caption = self.captioner.generate(att_feats, fc_feats, predicted_counts, stoi, max_len)
            
            return caption, predicted_counts


class CountAwareLoss(nn.Module):
    """
    Joint loss for multi-task learning.
    Combines caption generation loss with count prediction loss.
    """
    def __init__(self, caption_weight=1.0, count_weight=0.5, count_loss_type='mse'):
        super().__init__()
        
        self.caption_weight = caption_weight
        self.count_weight = count_weight
        self.count_loss_type = count_loss_type
        
        # Caption loss (cross-entropy)
        self.caption_criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        
        # Count loss
        if count_loss_type == 'mse':
            self.count_criterion = nn.MSELoss()
        elif count_loss_type == 'mae':
            self.count_criterion = nn.L1Loss()
        elif count_loss_type == 'smooth_l1':
            self.count_criterion = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown count loss type: {count_loss_type}")
    
    def forward(self, caption_outputs, predicted_counts, target_captions, target_counts):
        """
        Compute joint loss.
        
        Args:
            caption_outputs: [B, seq_len, vocab_size]
            predicted_counts: [B, count_vec_size]
            target_captions: [B, full_seq_len] - full caption with <start> and <end>
            target_counts: [B, count_vec_size]
        
        Returns:
            total_loss: Weighted sum of caption and count losses
            caption_loss: Caption generation loss
            count_loss: Count prediction loss
        """
        # Caption loss
        # caption_outputs predicts tokens 1 to end (excluding <start>)
        # So we compare with target_captions[:, 1:] (excluding <start>)
        caption_outputs_flat = caption_outputs.reshape(-1, caption_outputs.size(-1))
        # Slice target to match output length
        target_slice = target_captions[:, 1:1+caption_outputs.size(1)]
        target_captions_flat = target_slice.reshape(-1)
        caption_loss = self.caption_criterion(caption_outputs_flat, target_captions_flat)
        
        # Count loss
        count_loss = self.count_criterion(predicted_counts, target_counts)
        
        # Total loss
        total_loss = self.caption_weight * caption_loss + self.count_weight * count_loss
        
        return total_loss, caption_loss, count_loss
