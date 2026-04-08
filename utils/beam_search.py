"""
Beam Search Decoder for generating fluent captions.
Explores multiple hypotheses in parallel for better generation quality.
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple


def beam_search(model, att_feats, fc_feats, count_vec, stoi, itos,
                beam_size=3, max_len=50, device="cuda", length_penalty=0.7,
                repetition_penalty=1.2, no_repeat_ngram_size=3):
    """
    Beam search decoding for caption generation.
    
    Args:
        model: Trained captioning model (AttLSTMCount)
        att_feats: [1, 2048, 7, 7] - attention features
        fc_feats: [1, 2048] - FC features
        count_vec: [1, count_vec_size] - count vector
        stoi: word to index dict
        itos: index to word dict
        beam_size: Number of beams to maintain
        max_len: Maximum caption length
        device: 'cuda' or 'cpu'
        length_penalty: Penalty for shorter sequences (0.7 is common)
    
    Returns:
        caption: Generated caption string
        score: Log probability score
    """
    model.eval()
    
    start_token = stoi["<start>"]
    end_token = stoi["<end>"]
    pad_token = stoi["<pad>"]
    
    # Move to device
    att_feats = att_feats.to(device)
    fc_feats = fc_feats.to(device)
    count_vec = count_vec.to(device)
    
    with torch.no_grad():
        repetition_penalty = float(repetition_penalty) if repetition_penalty is not None else 1.0
        no_repeat_ngram_size = int(no_repeat_ngram_size) if no_repeat_ngram_size is not None else 0

        def _get_banned_tokens(seq_list):
            if no_repeat_ngram_size <= 1:
                return set()
            if len(seq_list) < no_repeat_ngram_size:
                return set()

            bans = {}
            for j in range(len(seq_list) - no_repeat_ngram_size + 1):
                ngram = tuple(seq_list[j:j + no_repeat_ngram_size])
                prefix = ngram[:-1]
                nxt = ngram[-1]
                if prefix in bans:
                    bans[prefix].add(nxt)
                else:
                    bans[prefix] = {nxt}

            prefix = tuple(seq_list[-(no_repeat_ngram_size - 1):])
            return bans.get(prefix, set())
        # Reshape features
        batch_size = att_feats.size(0)
        att_feats_reshaped = att_feats.view(batch_size, 2048, -1).permute(0, 2, 1)  # [1, 49, 2048]
        
        # Embed count
        count_embed = model.count_mlp(count_vec)  # [1, count_embed_dim]
        
        # Initialize hidden state
        h, c = model.init_hidden(fc_feats)
        
        # Initialize beams: (sequence, hidden, cell, log_prob, completed)
        beams = [(torch.tensor([start_token], device=device), h, c, 0.0, False)]
        completed_beams = []
        
        for step in range(max_len):
            candidates = []
            
            for seq, h_state, c_state, log_prob, completed in beams:
                if completed:
                    candidates.append((seq, h_state, c_state, log_prob, True))
                    continue
                
                # Last token
                token = seq[-1].unsqueeze(0)
                
                # Embed word
                word_embed = model.embed(token)  # [1, embed_dim]
                
                # Attention
                context, alpha = model.att(att_feats_reshaped, h_state)  # [1, 2048]
                
                # LSTM input
                lstm_input = torch.cat([word_embed, context, count_embed], dim=1)
                
                # LSTM step
                h_new, c_new = model.lstm(lstm_input, (h_state, c_state))
                
                # Predict next word
                logits = model.fc(h_new).squeeze(0)  # [vocab_size]

                # Do not generate <start> again
                if seq.numel() > 1:
                    start_token = stoi.get('<start>')
                    if start_token is not None:
                        logits[start_token] = float('-inf')

                # Apply repetition penalty
                if repetition_penalty > 1.0 and seq.numel() > 0:
                    used = set(seq.tolist())
                    used_idx = torch.tensor(list(used), device=logits.device, dtype=torch.long)
                    used_vals = logits.index_select(0, used_idx)
                    used_vals = torch.where(used_vals > 0, used_vals / repetition_penalty, used_vals * repetition_penalty)
                    logits = logits.scatter(0, used_idx, used_vals)

                # Apply no-repeat ngram blocking
                banned = _get_banned_tokens(seq.tolist())
                if banned:
                    banned_idx = torch.tensor(list(banned), device=logits.device, dtype=torch.long)
                    logits.index_fill_(0, banned_idx, float('-inf'))

                log_probs = F.log_softmax(logits, dim=-1)  # [vocab_size]
                
                # Get top-k candidates
                topk_log_probs, topk_indices = torch.topk(log_probs, beam_size)
                
                for i in range(beam_size):
                    next_token = topk_indices[i]
                    next_log_prob = log_prob + topk_log_probs[i].item()
                    next_seq = torch.cat([seq, next_token.unsqueeze(0)])
                    
                    # Check if sequence is complete
                    is_complete = (next_token.item() == end_token)
                    
                    candidates.append((next_seq, h_new, c_new, next_log_prob, is_complete))
            
            # Sort by score (with length penalty for completed sequences)
            def score_fn(candidate):
                seq, _, _, log_prob, completed = candidate
                if completed:
                    # Apply length penalty: score = log_prob / (length ^ length_penalty)
                    length = len(seq)
                    return log_prob / (length ** length_penalty)
                else:
                    return log_prob
            
            candidates.sort(key=score_fn, reverse=True)
            
            # Select top beam_size beams
            beams = candidates[:beam_size]
            
            # Check if all beams are completed
            if all(completed for _, _, _, _, completed in beams):
                break
        
        # Get best beam
        best_beam = max(beams, key=lambda x: x[3] / (len(x[0]) ** length_penalty))
        best_seq, _, _, best_score, _ = best_beam
        
        # Convert to text
        caption_tokens = best_seq.tolist()
        caption_words = []
        
        for token_id in caption_tokens:
            word = itos.get(token_id, "<unk>")
            if word not in ["<start>", "<end>", "<pad>"]:
                caption_words.append(word)
        
        caption = " ".join(caption_words)
        
        return caption, best_score


def beam_search_batch(model, att_feats, fc_feats, count_vecs, stoi, itos,
                      beam_size=3, max_len=25, device="cuda"):
    """
    Batch beam search for multiple images.
    
    Args:
        model: Trained captioning model
        att_feats: [B, 2048, 7, 7]
        fc_feats: [B, 2048]
        count_vecs: [B, count_vec_size]
        stoi, itos: Vocabulary mappings
        beam_size: Beam size
        max_len: Max length
        device: Device
    
    Returns:
        captions: List of caption strings
    """
    batch_size = att_feats.size(0)
    captions = []
    
    for i in range(batch_size):
        att = att_feats[i:i+1]
        fc = fc_feats[i:i+1]
        count = count_vecs[i:i+1]
        
        caption, _ = beam_search(model, att, fc, count, stoi, itos,
                                 beam_size, max_len, device)
        captions.append(caption)
    
    return captions
