#data_processing_script
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import random
import re

# Constants
NUCLEOTIDES = 'ACGU'
NUCLEOTIDE_TO_IDX = {nuc: i for i, nuc in enumerate(NUCLEOTIDES)}
RESIDUE_TYPES = ['A', 'C', 'G', 'U']
RESIDUE_TO_IDX = {res: i for i, res in enumerate(RESIDUE_TYPES)}
MAX_SEQ_LENGTH = 1024  # Maximum sequence length for padding/truncation

class RNADataset(Dataset):
    """
    Dataset class for RNA 3D structure prediction.
    Handles loading and preprocessing of RNA sequence and structure data.
    """
    
    def __init__(self, 
                 sequences_df: pd.DataFrame, 
                 labels_df: Optional[pd.DataFrame] = None,
                 is_training: bool = True,
                 max_length: int = MAX_SEQ_LENGTH):
        """
        Initialize the dataset.
        
        Args:
            sequences_df: DataFrame containing sequence information with columns:
                        ['target_id', 'sequence', 'temporal_cutoff', 'description', 'all_sequences']
            labels_df: DataFrame containing 3D structure labels with columns:
                     ['target_id', 'resname', 'resid', 'x_1', 'y_1', 'z_1']
            is_training: Whether this is a training dataset (affects data augmentation)
            max_length: Maximum sequence length (longer sequences will be truncated)
        """
        self.sequences_df = sequences_df
        self.labels_df = labels_df
        self.is_training = is_training
        self.max_length = max_length
        self.target_ids = sequences_df['target_id'].unique()
        
        # Create mapping from target_id to sequence data
        self.sequence_data = {}
        for _, row in sequences_df.iterrows():
            self.sequence_data[row['target_id']] = {
                'sequence': row['sequence'],
                'description': row['description'],
                'all_sequences': row['all_sequences']
            }
        
        # Create mapping from target_id to structure data if labels are provided
        self.structure_data = {}
        if labels_df is not None:
            for target_id in self.target_ids:
                target_data = labels_df[labels_df['target_id'] == target_id]
                if not target_data.empty:
                    self.structure_data[target_id] = {
                        'resname': target_data['resname'].values,
                        'resid': target_data['resid'].values,
                        'coords': target_data[['x_1', 'y_1', 'z_1']].values.astype(np.float32)
                    }
    
    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        return len(self.target_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item from the dataset.
        
        Returns:
            Dictionary containing:
                - seq_one_hot: One-hot encoded sequence [seq_len, 4]
                - mask: Sequence mask [seq_len]
                - resnames: Residue types as indices [seq_len]
                - resids: Residue IDs [seq_len]
                - coords: 3D coordinates [seq_len, 3]
                - target_id: Original target ID
                
        """
        target_id = self.target_ids[idx]
        seq_data = self.sequence_data[target_id]
        sequence = seq_data['sequence']
        
        # Truncate sequence if too long
        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]
        
        # Convert sequence to one-hot encoding
        seq_len = len(sequence)
        seq_one_hot = torch.zeros((self.max_length, len(NUCLEOTIDES)), dtype=torch.float32)
        
        for i, nuc in enumerate(sequence):
            if nuc in NUCLEOTIDE_TO_IDX:
                seq_one_hot[i, NUCLEOTIDE_TO_IDX[nuc]] = 1.0
        
        # Create sequence mask (1 for real tokens, 0 for padding)
        sequence_mask = torch.zeros(self.max_length, dtype=torch.bool)
        sequence_mask[:seq_len] = 1
        
        # Initialize outputs
        resnames = torch.zeros(self.max_length, dtype=torch.long)
        resids = torch.zeros(self.max_length, dtype=torch.long)
        coords = torch.zeros((self.max_length, 3), dtype=torch.float32)
        
        # Add structure data if available
        if target_id in self.structure_data and self.structure_data[target_id] is not None:
            struct = self.structure_data[target_id]
            struct_len = min(len(struct['resname']), self.max_length)
            
            # Convert residue names to indices
            for i in range(struct_len):
                if i < len(struct['resname']):
                    resname = struct['resname'][i].upper()
                    if resname in RESIDUE_TO_IDX:
                        resnames[i] = RESIDUE_TO_IDX[resname]
                    
                    resids[i] = struct['resid'][i] if i < len(struct['resid']) else 0
                    
                    if i < len(struct['coords']):
                        coords[i] = torch.tensor(struct['coords'][i], dtype=torch.float32)
        
        # Create output dictionary with the expected keys
        output = {
            'seq_one_hot': seq_one_hot,  # This is the key the model expects
            'mask': sequence_mask,
            'resnames': resnames,
            'resids': resids,
            'coords': coords,
            'target_id': target_id,
            'seq_len': seq_len,
            'descriptions': seq_data['description']  # Add description for the description encoder
        }
        
        return output

def collate_rna_batch(batch):
    """Collate function for handling variable-length sequences."""
    if not batch:
        return {}
        
    # Initialize batch dictionary
    batch_dict = {}
    
    # Get all keys from the first sample
    keys = batch[0].keys()
    
    # Process each key in the batch
    for key in keys:
        # Handle string fields (descriptions, target_id, etc.)
        if key in ['descriptions', 'target_id']:
            batch_dict[key] = [item[key] for item in batch]
        # Handle sequence length
        elif key == 'seq_len':
            batch_dict[key] = torch.tensor([item[key] for item in batch], dtype=torch.long)
        # Handle tensor fields (seq_one_hot, mask, resnames, resids, coords)
        else:
            # Check if all items have the key and are tensors
            if all(key in item and torch.is_tensor(item[key]) for item in batch):
                try:
                    batch_dict[key] = torch.stack([item[key] for item in batch])
                except RuntimeError as e:
                    print(f"Error stacking tensors for key '{key}': {e}")
                    print(f"Shapes: {[item[key].shape for item in batch if key in item]}")
                    raise
            else:
                # For non-tensor fields, just collect them as lists
                batch_dict[key] = [item[key] for item in batch if key in item]
    
    return batch_dict

def get_dataloaders(train_sequences_path: str,
                   train_labels_path: str,
                   val_sequences_path: str,
                   val_labels_path: str,
                   batch_size: int = 32,
                   num_workers: int = 4,
                   max_length: int = MAX_SEQ_LENGTH) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.
    
    Args:
        train_sequences_path: Path to training sequences CSV
        train_labels_path: Path to training labels CSV
        val_sequences_path: Path to validation sequences CSV
        val_labels_path: Path to validation labels CSV
        batch_size: Batch size for data loaders
        num_workers: Number of workers for data loading
        max_length: Maximum sequence length
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Load data
    train_sequences = pd.read_csv(train_sequences_path)
    train_labels = pd.read_csv(train_labels_path)
    val_sequences = pd.read_csv(val_sequences_path)
    val_labels = pd.read_csv(val_labels_path)
    
    # Create datasets
    train_dataset = RNADataset(
        sequences_df=train_sequences,
        labels_df=train_labels,
        is_training=True,
        max_length=max_length
    )
    
    val_dataset = RNADataset(
        sequences_df=val_sequences,
        labels_df=val_labels,
        is_training=False,
        max_length=max_length
    )
    
    # Create data loaders
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_rna_batch,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_rna_batch,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader

def create_test_loader(test_sequences_path: str,
                      batch_size: int = 32,
                      num_workers: int = 4,
                      max_length: int = MAX_SEQ_LENGTH) -> DataLoader:
    """
    Create a test data loader.
    
    Args:
        test_sequences_path: Path to test sequences CSV
        batch_size: Batch size for data loader
        num_workers: Number of workers for data loading
        max_length: Maximum sequence length
        
    Returns:
        DataLoader for test data
    """
    test_sequences = pd.read_csv(test_sequences_path)
    
    # Create dataset without labels
    test_dataset = RNADataset(
        sequences_df=test_sequences,
        labels_df=None,
        is_training=False,
        max_length=max_length
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_rna_batch,
        pin_memory=torch.cuda.is_available()
    )
    
    return test_dataset, test_loader
