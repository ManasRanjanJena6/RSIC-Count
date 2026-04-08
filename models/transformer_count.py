"""

Transformer-based Decoder for RSIC-Count.

Uses ViT encoder + Transformer decoder with count embeddings.

Modern architecture alternative to LSTM.

"""



import torch

import torch.nn as nn

import math





class PositionalEncoding(nn.Module):

    """Sinusoidal positional encoding for Transformer"""

    def __init__(self, d_model, max_len=100):

        super().__init__()

        

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        

        pe[:, 0::2] = torch.sin(position * div_term)

        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        

        self.register_buffer('pe', pe)

    

    def forward(self, x):

        """

        Args:

            x: [B, seq_len, d_model]

        Returns:

            x + positional encoding

        """

        return x + self.pe[:, :x.size(1), :]





class TransformerCountCaptioner(nn.Module):

    """

    Transformer-based caption generator with count embeddings.

    Uses pre-extracted CNN features or can use ViT.

    """

    def __init__(self, vocab_size, count_vec_size, feat_dim=2048,

                 d_model=512, nhead=8, num_decoder_layers=6, 

                 dim_feedforward=2048, dropout=0.1, 

                 count_embed_dim=128, max_len=100):

        super().__init__()

        

        self.vocab_size = vocab_size

        self.d_model = d_model

        self.count_vec_size = count_vec_size

        self.feat_dim = feat_dim

        

        # Visual feature projection (feat_dim -> d_model)

        self.visual_proj = nn.Linear(feat_dim, d_model)

        

        # Count embedding

        self.count_embed = nn.Sequential(

            nn.Linear(count_vec_size, count_embed_dim),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(count_embed_dim, d_model)

        )

        

        # Word embeddings

        self.word_embed = nn.Embedding(vocab_size, d_model)

        self.pos_encoding = PositionalEncoding(d_model, max_len)

        

        # Transformer decoder

        decoder_layer = nn.TransformerDecoderLayer(

            d_model=d_model,

            nhead=nhead,

            dim_feedforward=dim_feedforward,

            dropout=dropout,

            activation='relu',

            batch_first=True

        )

        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        

        # Output projection

        self.fc_out = nn.Linear(d_model, vocab_size)

        

        # Dropout

        self.dropout = nn.Dropout(dropout)

        

        # Initialize weights

        self.init_weights()

    

    def init_weights(self):

        """Initialize weights"""

        nn.init.xavier_uniform_(self.word_embed.weight)

        nn.init.xavier_uniform_(self.fc_out.weight)

        nn.init.constant_(self.fc_out.bias, 0)

    

    def generate_square_subsequent_mask(self, sz):

        """Generate causal mask for decoder"""

        mask = torch.triu(torch.ones(sz, sz), diagonal=1)

        mask = mask.masked_fill(mask == 1, float('-inf'))

        return mask

    

    def forward(self, att_feats, fc_feats, captions, count_vecs, teacher_forcing=True):

        """

        Args:

            att_feats: [B, 2048, 7, 7] - spatial features

            fc_feats: [B, 2048] - global features

            captions: [B, seq_len] - target captions

            count_vecs: [B, count_vec_size]

        Returns:

            outputs: [B, seq_len-1, vocab_size]

        """

        batch_size = att_feats.size(0)

        seq_len = captions.size(1) - 1  # Exclude last token

        

        # Process visual features

        # Flatten spatial features: [B, feat_dim, 7, 7] -> [B, 49, feat_dim]

        att_feats = att_feats.view(batch_size, self.feat_dim, -1).permute(0, 2, 1)

        att_feats = self.visual_proj(att_feats)  # [B, 49, d_model]

        

        # Process FC features

        fc_feats = self.visual_proj(fc_feats).unsqueeze(1)  # [B, 1, d_model]

        

        # Process count embeddings

        count_feats = self.count_embed(count_vecs).unsqueeze(1)  # [B, 1, d_model]

        

        # Concatenate memory: [spatial features | global feature | count]

        memory = torch.cat([att_feats, fc_feats, count_feats], dim=1)  # [B, 51, d_model]

        

        # Embed captions (exclude last token for target)

        tgt_input = captions[:, :-1]  # [B, seq_len-1]

        tgt_embed = self.word_embed(tgt_input) * math.sqrt(self.d_model)

        tgt_embed = self.pos_encoding(tgt_embed)  # [B, seq_len-1, d_model]

        tgt_embed = self.dropout(tgt_embed)

        

        # Create causal mask

        tgt_mask = self.generate_square_subsequent_mask(seq_len).to(att_feats.device)

        

        # Transformer decoder

        output = self.transformer_decoder(

            tgt=tgt_embed,

            memory=memory,

            tgt_mask=tgt_mask

        )  # [B, seq_len-1, d_model]

        

        # Project to vocabulary

        logits = self.fc_out(output)  # [B, seq_len-1, vocab_size]

        

        return logits

    

    def generate(self, att_feats, fc_feats, count_vec, stoi, max_len=25, temperature=1.0):

        """

        Greedy generation for inference.

        

        Args:

            att_feats: [1, 2048, 7, 7]

            fc_feats: [1, 2048]

            count_vec: [1, count_vec_size]

            stoi: word to index dict

            max_len: maximum length

            temperature: sampling temperature

        Returns:

            caption: List of token indices

        """

        self.eval()

        device = att_feats.device

        

        with torch.no_grad():

            # Process features

            att_feats = att_feats.view(1, 2048, -1).permute(0, 2, 1)

            att_feats = self.visual_proj(att_feats)  # [1, 49, d_model]

            

            fc_feats = self.visual_proj(fc_feats).unsqueeze(1)  # [1, 1, d_model]

            count_feats = self.count_embed(count_vec).unsqueeze(1)  # [1, 1, d_model]

            

            memory = torch.cat([att_feats, fc_feats, count_feats], dim=1)  # [1, 51, d_model]

            

            # Start with <start> token

            caption = [stoi['<start>']]

            

            for _ in range(max_len):

                # Prepare input

                tgt_input = torch.tensor([caption], dtype=torch.long, device=device)  # [1, current_len]

                tgt_embed = self.word_embed(tgt_input) * math.sqrt(self.d_model)

                tgt_embed = self.pos_encoding(tgt_embed)

                

                # Create mask

                current_len = len(caption)

                tgt_mask = self.generate_square_subsequent_mask(current_len).to(device)

                

                # Decode

                output = self.transformer_decoder(

                    tgt=tgt_embed,

                    memory=memory,

                    tgt_mask=tgt_mask

                )  # [1, current_len, d_model]

                

                # Get last token prediction

                logits = self.fc_out(output[:, -1, :])  # [1, vocab_size]

                logits = logits / temperature

                

                # Greedy selection

                predicted = torch.argmax(logits, dim=-1).item()

                caption.append(predicted)

                

                if predicted == stoi['<end>']:

                    break

            

            return caption

    

    def beam_search_generate(self, att_feats, fc_feats, count_vec, stoi, itos,

                             beam_size=3, max_len=25, length_penalty=0.7):

        """

        Beam search generation for Transformer model.

        

        Args:

            att_feats: [1, 2048, 7, 7]

            fc_feats: [1, 2048]

            count_vec: [1, count_vec_size]

            stoi, itos: Vocabulary mappings

            beam_size: Beam size

            max_len: Max length

            length_penalty: Length penalty

        Returns:

            caption: Best caption string

        """

        self.eval()

        device = att_feats.device

        

        with torch.no_grad():

            # Process features

            att_feats = att_feats.view(1, 2048, -1).permute(0, 2, 1)

            att_feats = self.visual_proj(att_feats)

            fc_feats = self.visual_proj(fc_feats).unsqueeze(1)

            count_feats = self.count_embed(count_vec).unsqueeze(1)

            memory = torch.cat([att_feats, fc_feats, count_feats], dim=1)

            

            # Initialize beams

            start_token = stoi['<start>']

            end_token = stoi['<end>']

            beams = [([start_token], 0.0)]  # (sequence, log_prob)

            

            for _ in range(max_len):

                candidates = []

                

                for seq, log_prob in beams:

                    if seq[-1] == end_token:

                        candidates.append((seq, log_prob))

                        continue

                    

                    # Prepare input

                    tgt_input = torch.tensor([seq], dtype=torch.long, device=device)

                    tgt_embed = self.word_embed(tgt_input) * math.sqrt(self.d_model)

                    tgt_embed = self.pos_encoding(tgt_embed)

                    

                    current_len = len(seq)

                    tgt_mask = self.generate_square_subsequent_mask(current_len).to(device)

                    

                    output = self.transformer_decoder(tgt=tgt_embed, memory=memory, tgt_mask=tgt_mask)

                    logits = self.fc_out(output[:, -1, :])

                    log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)

                    

                    # Get top-k

                    topk_log_probs, topk_indices = torch.topk(log_probs, beam_size)

                    

                    for i in range(beam_size):

                        next_token = topk_indices[i].item()

                        next_log_prob = log_prob + topk_log_probs[i].item()

                        next_seq = seq + [next_token]

                        candidates.append((next_seq, next_log_prob))

                

                # Sort and select top beams

                candidates.sort(key=lambda x: x[1] / (len(x[0]) ** length_penalty), reverse=True)

                beams = candidates[:beam_size]

                

                if all(seq[-1] == end_token for seq, _ in beams):

                    break

            

            # Get best beam

            best_seq, _ = beams[0]

            caption_words = [itos.get(token, "<unk>") for token in best_seq 

                           if itos.get(token, "") not in ["<start>", "<end>", "<pad>"]]

            

            return " ".join(caption_words)

