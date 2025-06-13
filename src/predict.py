import os
import torch
import numpy as np

from model import RNAStructurePredictor

def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint.get('model_config', {
        'embedding_dim': 128,
        'hidden_dim': 256,
        'dropout': 0.1
    })
    model = RNAStructurePredictor(**model_config)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model
def predict(model, test_loader, device):
    all_predictions_coords = []
    all_predictions_resname_logits = []
    all_predictions_resid_pred = []
    all_target_ids = [] # To store target_ids for saving

    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Prediction"): # Changed 'batch' to 'batch_data' for consistency
            # Move data to device
            for key in batch_data:
                if isinstance(batch_data[key], torch.Tensor):
                    batch_data[key] = batch_data[key].to(device)

            # Forward pass
            predictions = model(batch_data, split='test') # Pass entire batch_data and specify split

            # Collect predictions
            all_predictions_coords.append(predictions['coords_pred'].cpu().numpy())
            all_predictions_resname_logits.append(predictions['resname_logits'].cpu().numpy())
            all_predictions_resid_pred.append(predictions['resid_pred'].cpu().numpy())
            all_target_ids.extend(batch_data['target_id']) # Collect target_ids (singular)

    # Concatenate all collected predictions and target_ids
    predictions_dict = {
        'coords_pred': np.concatenate(all_predictions_coords, axis=0),
        'resname_logits': np.concatenate(all_predictions_resname_logits, axis=0),
        'resid_pred': np.concatenate(all_predictions_resid_pred, axis=0),
        'target_ids': all_target_ids # Store target_ids here
    }
    return predictions_dict

def predict_for_test(test_loader, model_path, test_sequences_path, output_file, batch_size=32):
    device = 'cpu'
    print(f"Using device: {device}")
    
    print(f"Loading model from {model_path}...")
    model = load_model(model_path, device)
    
    print(f"Loading test data from {test_sequences_path}...")
    
    print("Generating predictions...")
    predictions = predict(model, test_loader, device)
    
    print(f"Saving predictions to {output_file}...")
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    np.save(output_file, predictions)
    
    print("Prediction completed!")

import os
import numpy as np
import pandas as pd

def load_predictions(predictions_path: str):
    print(f"Loading predictions from {predictions_path}...")
    data = np.load(predictions_path, allow_pickle=True).item()
    return data

def format_predictions(predictions, target_ids):
    all_rows = []
    num_sequences = len(target_ids)
    
    for i in range(num_sequences):
        target_id = target_ids[i]
        coords = predictions['coords_pred'][i]
        resnames = predictions.get('resname_logits', None)
        resids = predictions.get('resid_pred', None)
        
        if coords.ndim == 2:
            coords = coords.reshape(coords.shape[0], 1, 3)
        
        seq_len = coords.shape[0]
        num_positions = coords.shape[1]
        
        if resnames is not None:
            resname_ids = np.argmax(resnames[i][:seq_len], axis=-1)
            resname_map = {0: 'A', 1: 'C', 2: 'G', 3: 'U'}
            resname_letters = [resname_map.get(idx, 'X') for idx in resname_ids]
        else:
            resname_letters = ['N'] * seq_len
        
        if resids is not None:
            resid_values = np.round(resids[i][:seq_len]).astype(int)
        else:
            resid_values = np.arange(1, seq_len + 1)
        
        for pos in range(seq_len):
            row = {
                'target_id': target_id,
                'resid': int(resid_values[pos]),
                'resname': resname_letters[pos]
            }
            for pos_idx in range(num_positions):
                row[f'x_{pos_idx+1}'] = float(coords[pos, pos_idx, 0])
                row[f'y_{pos_idx+1}'] = float(coords[pos, pos_idx, 1])
                row[f'z_{pos_idx+1}'] = float(coords[pos, pos_idx, 2])
            all_rows.append(row)
    return pd.DataFrame(all_rows)

def create_submission(predictions_path, output_file, target_ids_path, sample_submission_path=None):
    predictions = load_predictions(predictions_path)
    print(f"Loading target IDs from {target_ids_path}...")
    target_ids_df = pd.read_csv(target_ids_path)
    target_ids = target_ids_df['target_id'].tolist()
    
    if len(target_ids) != len(predictions['coords_pred']):
        raise ValueError(f"Number of target IDs ({len(target_ids)}) does not match number of predictions ({len(predictions['coords_pred'])})")
    
    print("Formatting predictions...")
    submission_df = format_predictions(predictions, target_ids)
    
    if sample_submission_path and os.path.exists(sample_submission_path):
        print(f"Validating against sample submission format from {sample_submission_path}...")
        sample_df = pd.read_csv(sample_submission_path)
        required_columns = set(sample_df.columns)
        missing_columns = required_columns - set(submission_df.columns)
        if missing_columns:
            print(f"Warning: Missing columns in submission: {missing_columns}")
            for col in missing_columns:
                submission_df[col] = 0.0
    
    os.makedirs(os.path.dirname(os.path.abspath(output_file)) or '.', exist_ok=True)
    submission_df.to_csv(output_file, index=False, float_format='%.6f')
    print(f"Submission file created at {output_file}")
    print(f"Submission shape: {submission_df.shape}")

# To run this cell in Colab, call:
# create_submission(predictions_path="predictions.npy", output_file="submission.csv", target_ids_path="target_ids.csv", sample_submission_path=None)
