"""
Baseline Attention LSTM with Count Embeddings for RSIC-Count.
Integrates visual features, spatial attention, and object count information.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftAttention(nn.Module):
    """Soft attention mechanism over spatial features"""
    def __init__(self, feat_dim, hidden_dim, att_dim):
        super().__init__()
        self.hidden_proj = nn.Linear(hidden_dim, att_dim)
        self.att_proj = nn.Linear(feat_dim, att_dim)  # Dynamic feature dim
        self.alpha_proj = nn.Linear(att_dim, 1)
    
    def forward(self, att_feats, hidden):
        """
        Args:
            att_feats: [B, 49, 2048] - spatial features
            hidden: [B, hidden_dim] - LSTM hidden state
        Returns:
            context: [B, 2048] - attended features
            alpha: [B, 49] - attention weights
        """
        # Project features
        att_proj = self.att_proj(att_feats)  # [B, 49, att_dim]
        hidden_proj = self.hidden_proj(hidden).unsqueeze(1)  # [B, 1, att_dim]
        
        # Compute attention scores
        combined = torch.tanh(att_proj + hidden_proj)  # [B, 49, att_dim]
        scores = self.alpha_proj(combined).squeeze(-1)  # [B, 49]
        
        # Attention weights
        alpha = F.softmax(scores, dim=1)  # [B, 49]
        
        # Weighted sum
        context = torch.bmm(alpha.unsqueeze(1), att_feats).squeeze(1)  # [B, 2048]
        
        return context, alpha


class AttLSTMCount(nn.Module):
    """Attention LSTM Captioner with Count Embeddings"""
    def __init__(self, vocab_size, count_vec_size, feat_dim=2048, embed_dim=512, hidden_dim=512, 
                 att_dim=512, count_embed_dim=128, dropout=0.5):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.count_vec_size = count_vec_size
        self.feat_dim = feat_dim
        
        # Word embeddings
        self.embed = nn.Embedding(vocab_size, embed_dim)
        
        # Count embedding MLP
        self.count_mlp = nn.Sequential(
            nn.Linear(count_vec_size, count_embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(count_embed_dim, count_embed_dim)
        )
        
        # Attention mechanism
        self.att = SoftAttention(feat_dim, hidden_dim, att_dim)
        
        # FC feature projection
        self.fc_proj = nn.Linear(feat_dim, hidden_dim)
        
        # LSTM decoder
        lstm_input_dim = embed_dim + feat_dim + count_embed_dim  # word + visual context + count
        self.lstm = nn.LSTMCell(lstm_input_dim, hidden_dim)
        
        # Output projection
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights"""
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)
    
    def init_hidden(self, fc_feats):
        """Initialize LSTM hidden state from FC features"""
        h = torch.tanh(self.fc_proj(fc_feats))
        c = torch.zeros_like(h)
        return h, c
    
    def forward(self, att_feats, fc_feats, captions, count_vecs, teacher_forcing=True):
        """
        Args:
            att_feats: [B, 2048, 7, 7] - attention features
            fc_feats: [B, 2048] - FC features
            captions: [B, max_len] - caption tokens
            count_vecs: [B, count_vec_size] - count embeddings
            teacher_forcing: Use ground truth tokens during training
        Returns:
            outputs: [B, max_len-1, vocab_size] - predicted logits
            alphas: [B, max_len-1, 49] - attention weights
        """
        batch_size = att_feats.size(0)
        
        # Reshape attention features: [B, feat_dim, 7, 7] -> [B, 49, feat_dim]
        att_feats = att_feats.view(batch_size, self.feat_dim, -1).permute(0, 2, 1)
        
        # Embed count vector
        count_embed = self.count_mlp(count_vecs)  # [B, count_embed_dim]
        
        # Initialize hidden state
        h, c = self.init_hidden(fc_feats)
        
        # Prepare for decoding
        max_len = captions.size(1) - 1  # Exclude last token
        outputs = []
        alphas = []
        
        for t in range(max_len):
            # Get input token
            if teacher_forcing:
                token = captions[:, t]
            else:
                if t == 0:
                    token = captions[:, 0]  # <start> token
                else:
                    token = torch.argmax(output, dim=1)
            
            # Embed word
            word_embed = self.embed(token)  # [B, embed_dim]
            
            # Attention over spatial features
            context, alpha = self.att(att_feats, h)  # [B, 2048], [B, 49]
            
            # Concatenate word + visual context + count
            lstm_input = torch.cat([word_embed, context, count_embed], dim=1)
            lstm_input = self.dropout(lstm_input)
            
            # LSTM step
            h, c = self.lstm(lstm_input, (h, c))
            
            # Predict next word
            output = self.fc(self.dropout(h))  # [B, vocab_size]
            
            outputs.append(output)
            alphas.append(alpha)
        
        # Stack outputs
        outputs = torch.stack(outputs, dim=1)  # [B, max_len-1, vocab_size]
        alphas = torch.stack(alphas, dim=1)    # [B, max_len-1, 49]
        
        return outputs, alphas
    
    def generate(self, att_feats, fc_feats, count_vec, stoi, max_len=50,
                 no_repeat_ngram_size=3, repetition_penalty=1.2, temperature=1.0):
        """
        Greedy generation for inference
        Args:
            att_feats: [1, 2048, 7, 7]
            fc_feats: [1, 2048]
            count_vec: [1, count_vec_size]
            stoi: word to index dict
            max_len: maximum generation length
        Returns:
            caption: List of token indices
            alphas: [max_len, 49] attention weights
        """
        self.eval()
        with torch.no_grad():
            # Reshape features
            att_feats = att_feats.view(1, self.feat_dim, -1).permute(0, 2, 1)  # [1, 49, feat_dim]
            count_embed = self.count_mlp(count_vec)  # [1, count_embed_dim]
            
            # Initialize
            h, c = self.init_hidden(fc_feats)
            
            caption = [stoi['<start>']]
            alphas_list = []

            no_repeat_ngram_size = int(no_repeat_ngram_size) if no_repeat_ngram_size is not None else 0
            repetition_penalty = float(repetition_penalty) if repetition_penalty is not None else 1.0
            temperature = float(temperature) if temperature is not None else 1.0

            ngram_bans = {}

            def _update_ngram_bans(seq):
                if no_repeat_ngram_size <= 1:
                    return
                if len(seq) < no_repeat_ngram_size:
                    return
                ngram = tuple(seq[-no_repeat_ngram_size:])
                prefix = ngram[:-1]
                nxt = ngram[-1]
                if prefix in ngram_bans:
                    ngram_bans[prefix].add(nxt)
                else:
                    ngram_bans[prefix] = {nxt}
            
            for _ in range(max_len):
                token = torch.tensor([caption[-1]], device=att_feats.device)
                word_embed = self.embed(token)
                
                context, alpha = self.att(att_feats, h)
                lstm_input = torch.cat([word_embed, context, count_embed], dim=1)
                
                h, c = self.lstm(lstm_input, (h, c))

                logits = self.fc(h).squeeze(0)
                if temperature != 1.0:
                    logits = logits / temperature

                if len(caption) > 1:
                    start_token = stoi.get('<start>')
                    if start_token is not None:
                        logits[start_token] = float('-inf')

                unk_token = stoi.get('<unk>')
                if unk_token is not None and len(caption) > 0 and caption[-1] == unk_token:
                    logits[unk_token] = float('-inf')

                if repetition_penalty > 1.0 and len(caption) > 0:
                    used = set(caption)
                    used_idx = torch.tensor(list(used), device=logits.device, dtype=torch.long)
                    used_vals = logits.index_select(0, used_idx)
                    used_vals = torch.where(used_vals > 0, used_vals / repetition_penalty, used_vals * repetition_penalty)
                    logits = logits.scatter(0, used_idx, used_vals)

                if no_repeat_ngram_size > 1 and len(caption) >= (no_repeat_ngram_size - 1):
                    prefix = tuple(caption[-(no_repeat_ngram_size - 1):])
                    banned = ngram_bans.get(prefix)
                    if banned:
                        banned_idx = torch.tensor(list(banned), device=logits.device, dtype=torch.long)
                        logits.index_fill_(0, banned_idx, float('-inf'))

                predicted = torch.argmax(logits, dim=0).item()
                caption.append(predicted)
                alphas_list.append(alpha.squeeze(0).cpu().numpy())

                _update_ngram_bans(caption)
                
                if predicted == stoi['<end>']:
                    break

            return caption, alphas_list
