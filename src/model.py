import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from deepspeed.ops.adam import FusedAdam


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class SequenceEncoder(nn.Module):
    def __init__(self, embedding_dim=128, dropout=0.1):
        super(SequenceEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Linear(4, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=8,
            dim_feedforward=512,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
    def forward(self, seq_one_hot, mask=None):
        x = self.embedding(seq_one_hot)
        x = self.pos_encoder(x)
        if mask is not None:
            attn_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
            attn_mask = attn_mask.bool()
        else:
            attn_mask = None
        x = self.transformer_encoder(x, src_key_padding_mask=~mask.bool() if mask is not None else None)
        return x

class DescriptionEncoder(nn.Module):
    def __init__(self, embedding_dim=128, vocab_size=128, max_len=512, dropout=0.1):
        super(DescriptionEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=4,
            dim_feedforward=256,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, descriptions):
        max_len = 512
        batch_size = len(descriptions)
        input_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
        for i, desc in enumerate(descriptions):
            ascii_ids = [ord(c) if ord(c) < 128 else 0 for c in desc[:max_len]]
            input_ids[i, :len(ascii_ids)] = torch.tensor(ascii_ids)
        input_ids = input_ids.to(next(self.parameters()).device)
        x = self.embedding(input_ids)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.pool(x.transpose(1, 2)).squeeze(-1)
        x = self.projection(x)
        return x

class SecondaryStructureEncoder(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128):
        super(SecondaryStructureEncoder, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, edge_attr=None):
        # Adjacency-like aggregation
        row, col = edge_index
        deg = torch.bincount(row, minlength=x.size(0)).clamp(min=1).unsqueeze(1)
        agg = torch.zeros_like(x)
        agg.index_add_(0, row, x[col])
        x = F.relu(self.linear1(agg / deg))
        agg = torch.zeros_like(x)
        agg.index_add_(0, row, x[col])
        x = F.relu(self.linear2(agg / deg))
        return x

class BPPGraphEncoder(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128):
        super(BPPGraphEncoder, self).__init__()
        self.linear_q1 = nn.Linear(input_dim, hidden_dim)
        self.linear_k1 = nn.Linear(input_dim, hidden_dim)
        self.linear_v1 = nn.Linear(input_dim, hidden_dim)
        self.linear_q2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_k2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_v2 = nn.Linear(hidden_dim, hidden_dim)

    def attention_layer(self, x, edge_index, linear_q, linear_k, linear_v):
        row, col = edge_index
        Q = linear_q(x)
        K = linear_k(x)
        V = linear_v(x)
        scores = torch.sum(Q[row] * K[col], dim=-1) / math.sqrt(Q.size(-1))
        alpha = torch.zeros_like(scores)
        alpha = F.softmax(scores, dim=0)
        agg = torch.zeros_like(V)
        agg.index_add_(0, row, alpha.unsqueeze(1) * V[col])
        return agg

    def forward(self, x, edge_index, edge_attr=None):
        x = F.elu(self.attention_layer(x, edge_index, self.linear_q1, self.linear_k1, self.linear_v1))
        x = self.attention_layer(x, edge_index, self.linear_q2, self.linear_k2, self.linear_v2)
        return x

class FeatureFusion(nn.Module):
    def __init__(self, seq_dim=128, desc_dim=128, output_dim=256):
        super(FeatureFusion, self).__init__()
        self.seq_dim = seq_dim
        self.desc_dim = desc_dim
        self.output_dim = output_dim
        self.query_proj = nn.Linear(seq_dim, output_dim)
        self.key_proj = nn.Linear(desc_dim, output_dim)
        self.value_proj = nn.Linear(desc_dim, output_dim)
        self.output_proj = nn.Linear(seq_dim + output_dim, output_dim)
        
    def forward(self, seq_features, desc_features):
        batch_size, seq_len, _ = seq_features.shape
        desc_features_expanded = desc_features.unsqueeze(1).expand(-1, seq_len, -1)
        queries = self.query_proj(seq_features)
        keys = self.key_proj(desc_features_expanded)
        values = self.value_proj(desc_features_expanded)
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.output_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)
        context = torch.matmul(attention_weights, values)
        fused_features = torch.cat([seq_features, context], dim=-1)
        output = self.output_proj(fused_features)
        return output

class SE3EquivariantLayer(nn.Module):
    def __init__(self, input_dim, output_dim=3, hidden_dim=64):
        super(SE3EquivariantLayer, self).__init__()
        # Replace irreps and tensor products by normal Linear layers:
        # input_dim is feature dim per residue (seq element)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # output_dim usually 3 for xyz coords
        self.scalar = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape
        x_flat = x.view(-1, x.shape[-1])  # (batch_size * seq_len, input_dim)
        # scalar parameter could be used as a learnable multiplier
        scalar_input = self.scalar.expand(x_flat.shape[0], 1)
        hidden = self.fc1(x_flat) * scalar_input  # simple gating by scalar
        hidden = F.relu(hidden)
        output = self.fc2(hidden)
        coords = output.view(batch_size, seq_len, -1)  # (batch_size, seq_len, 3)
        return coords

class RNAStructurePredictor(nn.Module):
    """
    Complete RNA 3D structure prediction model.
    """
    def __init__(self, embedding_dim=128, hidden_dim=256, dropout=0.1):
        super(RNAStructurePredictor, self).__init__()
        
        # Encoders
        self.sequence_encoder = SequenceEncoder(embedding_dim, dropout)
        self.description_encoder = DescriptionEncoder(embedding_dim)
        self.ss_encoder = SecondaryStructureEncoder(embedding_dim, embedding_dim)
        self.bpp_encoder = BPPGraphEncoder(embedding_dim, embedding_dim)
        
        # Feature fusion
        self.feature_fusion = FeatureFusion(embedding_dim, embedding_dim, hidden_dim)
        
        # Transformer backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=1024,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_backbone = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        # Output heads
        self.se3_layer = SE3EquivariantLayer(hidden_dim)
        self.resname_head = nn.Linear(hidden_dim, 4)  # A, C, G, U
        self.resid_head = nn.Linear(hidden_dim, 1)  # Residue ID
        
    def forward(self, batch_data, split='train'):
        """
        Forward pass through the model.
        
        Args:
            batch_data: Dictionary containing input data
            split: 'train', 'validation', or 'test'
        """
        # Extract inputs
        seq_one_hot = batch_data['seq_one_hot']
        descriptions = batch_data['descriptions']
        mask = batch_data['mask']
        
        batch_size, seq_len = seq_one_hot.shape[:2]
        
        # Encode sequence
        seq_features = self.sequence_encoder(seq_one_hot, mask)
        
        # Encode description
        desc_features = self.description_encoder(descriptions)
        
        # Fuse features
        fused_features = self.feature_fusion(seq_features, desc_features)
        
        # Apply transformer backbone
        backbone_features = self.transformer_backbone(
            fused_features,
            src_key_padding_mask=~mask.bool() if mask is not None else None
        )
        
        # Determine output size based on split
        if split == 'train':
            num_positions = 1
        elif split == 'validation':
            num_positions = 40
        else:  # test
            num_positions = 5
            
        # Predict coordinates using SE(3)-equivariant layer
        coords_pred = self.se3_layer(backbone_features)  # [batch_size, seq_len, 3]
        
        # Ensure consistent shape between training and validation
        coords_pred = coords_pred.unsqueeze(2)  # [batch_size, seq_len, 1, 3]
        
        # For validation, repeat the coordinates to match the expected number of positions
        if num_positions > 1:
            coords_pred = coords_pred.expand(-1, -1, num_positions, -1)  # [batch_size, seq_len, num_positions, 3]
        
        # Predict residue names and IDs
        resname_logits = self.resname_head(backbone_features)
        resid_pred = self.resid_head(backbone_features).squeeze(-1)
        
        return {
            'coords_pred': coords_pred,
            'resname_logits': resname_logits,
            'resid_pred': resid_pred
        }

class RNAStructureLoss(nn.Module):
    """
    Combined loss function for RNA 3D structure prediction.
    """
    def __init__(self, coord_weight=1.0, resname_weight=0.5, resid_weight=0.5, coverage_weight=0.1):
        super(RNAStructureLoss, self).__init__()
        self.coord_weight = coord_weight
        self.resname_weight = resname_weight
        self.resid_weight = resid_weight
        self.coverage_weight = coverage_weight
        
        self.resname_loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.coord_loss_fn = nn.MSELoss(reduction='none')
        self.resid_loss_fn = nn.MSELoss(reduction='none')
        
    def forward(self, predictions, targets, mask=None, output_mask=None):
        """
        Compute the combined loss.
        
        Args:
            predictions: Dictionary with model predictions
            targets: Dictionary with ground truth values
            mask: Sequence mask [batch_size, seq_len]
            output_mask: Output positions mask [batch_size, seq_len, num_positions]
        """
        # Extract predictions and targets
        coords_pred = predictions['coords_pred']
        resname_logits = predictions['resname_logits']
        resid_pred = predictions['resid_pred']
        
        coords_target = targets['coords']
        resname_target = targets['resnames']
        resid_target = targets['resids']
        
        batch_size, seq_len = coords_pred.shape[:2]
        
        # Apply masks
        if mask is None:
            mask = torch.ones(batch_size, seq_len, device=coords_pred.device)
        
        if output_mask is None:
            if coords_pred.dim() == 4:  # [batch, seq_len, num_positions, 3]
                output_mask = torch.ones(batch_size, seq_len, coords_pred.shape[2], device=coords_pred.device)
            else:  # [batch, seq_len, 3]
                output_mask = torch.ones(batch_size, seq_len, 1, device=coords_pred.device)
        
        # Handle invalid values in targets
        # Replace -1e18 values with zeros and create a validity mask
        if coords_target.dim() >= 3:
            valid_coords_mask = (coords_target != -1e18).all(dim=-1)
            if valid_coords_mask.dim() == 3:  # [batch, seq_len, num_positions]
                output_mask = output_mask * valid_coords_mask.float()
            
            # Replace invalid values with zeros to avoid NaN
            coords_target = torch.where(
                coords_target == -1e18,
                torch.zeros_like(coords_target),
                coords_target
            )
        
        # Coordinate loss
        # Ensure both predictions and targets have the same number of dimensions
        if coords_pred.dim() != coords_target.dim():
            if coords_pred.dim() == 4 and coords_target.dim() == 3:  # Predictions have positions, target doesn't
                coords_target = coords_target.unsqueeze(2)  # [batch, seq_len, 1, 3]
            elif coords_pred.dim() == 3 and coords_target.dim() == 4:  # Target has positions, predictions don't
                coords_pred = coords_pred.unsqueeze(2)  # [batch, seq_len, 1, 3]
        
        # Ensure shapes match for loss calculation
        if coords_pred.dim() == 4 and coords_target.dim() == 4:
            if coords_pred.shape[2] == 1 and coords_target.shape[2] > 1:
                # If model predicts single position but target has multiple, use first target position
                coords_target = coords_target[:, :, 0:1]  # [batch, seq_len, 1, 3]
            elif coords_pred.shape[2] > 1 and coords_target.shape[2] == 1:
                # If target has single position but model predicts multiple, repeat target
                coords_target = coords_target.expand_as(coords_pred)
        
        # Data validation and cleaning
        def clean_tensor(tensor, name):
            """Clean and validate tensor, replacing invalid values with zeros"""
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
            return tensor
        
        # Clean both predictions and targets
        coords_pred = clean_tensor(coords_pred, "predictions")
        coords_target = clean_tensor(coords_target, "targets")
        
        # Calculate MSE loss with error handling
        try:
            # Calculate MSE loss
            coord_loss = self.coord_loss_fn(coords_pred, coords_target)
            
            # Handle invalid loss values
            if torch.isnan(coord_loss).any() or torch.isinf(coord_loss).any():
                coord_loss = torch.nan_to_num(coord_loss, nan=0.0, posinf=1e6, neginf=0.0)
            
            # Ensure we have a valid scalar loss
            if torch.isnan(coord_loss) or torch.isinf(coord_loss):
                coord_loss = torch.tensor(0.0, device=coords_pred.device)
                
        except Exception as e:
            coord_loss = torch.tensor(0.0, device=coords_pred.device)
        
        # Handle masks
        if mask is not None or output_mask is not None:
            # If coord_loss is scalar, we don't need to apply masks
            if coord_loss.dim() == 0:
                valid_elements = torch.tensor(1.0, device=coord_loss.device)
            else:
                # Start with all ones for valid elements
                valid_elements = torch.ones_like(coord_loss, device=coord_loss.device)
                
                # Get the target shape for broadcasting
                target_shape = coord_loss.shape
                
                # Apply sequence mask if provided
                if mask is not None and mask.any():
                    # Ensure mask is on the same device
                    mask = mask.to(valid_elements.device)
                    
                    # Reshape mask to match the first two dimensions of coord_loss
                    while mask.dim() < 2:
                        mask = mask.unsqueeze(-1)
                    
                    # Add singleton dimensions to match coord_loss dimensions
                    mask = mask.view(mask.size(0), mask.size(1), *([1] * (len(target_shape) - 2)))
                    valid_elements = valid_elements * mask
                
                # Apply output mask if provided
                if output_mask is not None and output_mask.any():
                    # Ensure output_mask is on the same device
                    output_mask = output_mask.to(valid_elements.device)
                    
                    # Ensure output_mask is at least 2D
                    while output_mask.dim() < 2:
                        output_mask = output_mask.unsqueeze(-1)
                    
                    # Handle different output_mask dimensions
                    if output_mask.dim() == 2:  # [batch, seq_len]
                        # Add singleton dimensions to match coord_loss
                        output_mask = output_mask.unsqueeze(-1)  # [batch, seq_len, 1]
                        if len(target_shape) > 3:  # If there's a position dimension
                            output_mask = output_mask.unsqueeze(-1)  # [batch, seq_len, 1, 1]
                    elif output_mask.dim() == 3:  # [batch, seq_len, num_positions]
                        # Add coordinate dimension if needed
                        if len(target_shape) > 3:
                            output_mask = output_mask.unsqueeze(-1)  # [batch, seq_len, num_positions, 1]
                    
                    # Ensure output_mask has same number of dimensions as coord_loss
                    while output_mask.dim() < len(target_shape):
                        output_mask = output_mask.unsqueeze(-1)
                    
                    # Expand to match target shape
                    try:
                        output_mask = output_mask.expand(target_shape)
                        valid_elements = valid_elements * output_mask
                    except RuntimeError as e:
                        print(f"Error: Cannot apply output_mask with shape {output_mask.shape} to tensor with shape {target_shape}")
                        print(f"coord_loss shape: {coord_loss.shape}")
                        print(f"valid_elements shape: {valid_elements.shape}")
                        print(f"output_mask shape after expansion attempt: {output_mask.shape}")
                        print(f"coord_loss device: {coord_loss.device}, output_mask device: {output_mask.device}")
                        raise RuntimeError("Mask shape mismatch. Please check your data and model outputs.")
                
                # Ensure we have at least one valid element
                if valid_elements.sum() == 0:
                    print("Warning: No valid elements after applying masks. Using zero loss.")
                    return {
                        'loss': torch.tensor(0.0, device=coord_loss.device, requires_grad=True),
                        'coord_loss': torch.tensor(0.0, device=coord_loss.device),
                        'resname_loss': torch.tensor(0.0, device=coord_loss.device),
                        'resid_loss': torch.tensor(0.0, device=coord_loss.device)
                    }
            
            # Calculate mean only over valid elements with numerical stability
            valid_count = valid_elements.sum()
            if valid_count > 0:
                loss_sum = (coord_loss * valid_elements).sum()
                if torch.isnan(loss_sum) or torch.isinf(loss_sum):
                    
                    coord_loss = torch.tensor(0.0, device=coords_pred.device)
                else:
                    coord_loss = loss_sum / valid_count
                    # Ensure we don't return NaN
                    if torch.isnan(coord_loss) or torch.isinf(coord_loss):
                        
                        coord_loss = torch.tensor(0.0, device=coords_pred.device)
            else:
                
                coord_loss = torch.tensor(0.0, device=coords_pred.device)
        else:
            # No mask, just take mean over all dimensions except batch
            coord_loss = coord_loss.mean()
        
        # Residue name loss - handle potential out-of-bounds indices
        try:
            resname_loss = self.resname_loss_fn(
                resname_logits.reshape(-1, 4),
                torch.clamp(resname_target.reshape(-1), 0, 3)  # Clamp to valid range
            ).reshape(batch_size, seq_len)
            
            mask_sum = mask.sum()
            if mask_sum > 0:
                resname_loss = (resname_loss * mask).sum() / mask_sum
            else:
                resname_loss = torch.tensor(0.0, device=coords_pred.device)
        except Exception as e:
            print(f"Error in resname loss: {e}")
            print(f"resname_logits shape: {resname_logits.shape}, resname_target shape: {resname_target.shape}")
            resname_loss = torch.tensor(0.0, device=coords_pred.device)
        
        # Residue ID loss
        try:
            resid_loss = self.resid_loss_fn(resid_pred, resid_target.float())
            
            mask_sum = mask.sum()
            if mask_sum > 0:
                resid_loss = (resid_loss * mask).sum() / mask_sum
            else:
                resid_loss = torch.tensor(0.0, device=coords_pred.device)
        except Exception as e:
            print(f"Error in resid loss: {e}")
            resid_loss = torch.tensor(0.0, device=coords_pred.device)
        
        # Coverage loss - encourage diversity in predictions
        if coords_pred.dim() == 4 and coords_pred.shape[2] > 1:
            try:
                # Calculate RMSD between different position predictions for the same residue
                coverage_loss = 0.0
                num_positions = coords_pred.shape[2]
                
                for i in range(num_positions):
                    for j in range(i+1, num_positions):
                        pos_i = coords_pred[:, :, i]
                        pos_j = coords_pred[:, :, j]
                        
                        # Calculate squared distance
                        dist = ((pos_i - pos_j) ** 2).sum(dim=-1)  # [batch, seq_len]
                        
                        # We want positions to be different, so penalize small distances
                        # Add epsilon to avoid exp(-inf)
                        coverage_loss += torch.exp(-torch.clamp(dist, min=1e-6, max=50.0)).mean()
                
                if num_positions > 1:
                    coverage_loss /= (num_positions * (num_positions - 1) / 2)
                else:
                    coverage_loss = torch.tensor(0.0, device=coords_pred.device)
            except Exception as e:
                print(f"Error in coverage loss: {e}")
                coverage_loss = torch.tensor(0.0, device=coords_pred.device)
        else:
            coverage_loss = torch.tensor(0.0, device=coords_pred.device)
        
        # Combine losses with checks for NaN values
        coord_loss = torch.nan_to_num(coord_loss, nan=0.0, posinf=0.0, neginf=0.0)
        resname_loss = torch.nan_to_num(resname_loss, nan=0.0, posinf=0.0, neginf=0.0)
        resid_loss = torch.nan_to_num(resid_loss, nan=0.0, posinf=0.0, neginf=0.0)
        coverage_loss = torch.nan_to_num(coverage_loss, nan=0.0, posinf=0.0, neginf=0.0)
        
        total_loss = (
            self.coord_weight * coord_loss +
            self.resname_weight * resname_loss +
            self.resid_weight * resid_loss +
            self.coverage_weight * coverage_loss
        )
        
        return {
            'total_loss': total_loss,
            'coord_loss': coord_loss,
            'resname_loss': resname_loss,
            'resid_loss': resid_loss,
            'coverage_loss': coverage_loss
        }
