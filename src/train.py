import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, accuracy_score
import pandas as pd
import time
import json

def train_epoch(model, dataloader, optimizer, criterion, device, split='train'):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    coord_loss_sum = 0
    resname_loss_sum = 0
    resid_loss_sum = 0
    coverage_loss_sum = 0
    
    progress_bar = tqdm(dataloader, desc=f"{split} Epoch")
    
    for batch_idx, batch_data in enumerate(progress_bar):
        # Move data to device
        for key in batch_data:
            if isinstance(batch_data[key], torch.Tensor):
                batch_data[key] = batch_data[key].to(device)
        
        # Forward pass
        predictions = model(batch_data, split=split)
        
        # Compute loss
        targets = {
            'coords': batch_data['coords'],
            'resnames': batch_data['resnames'],
            'resids': batch_data['resids']
        }
        
        loss_dict = criterion(
            predictions, 
            targets, 
            mask=batch_data['mask'],
            output_mask=batch_data.get('output_mask')
        )
        
        loss = loss_dict['total_loss']
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        coord_loss_sum += loss_dict['coord_loss'].item()
        resname_loss_sum += loss_dict['resname_loss'].item()
        resid_loss_sum += loss_dict['resid_loss'].item()
        coverage_loss_sum += loss_dict['coverage_loss'].item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': loss.item(),
            'coord_loss': loss_dict['coord_loss'].item(),
            'resname_loss': loss_dict['resname_loss'].item(),
            'resid_loss': loss_dict['resid_loss'].item()
        })
    
    # Calculate average losses
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_coord_loss = coord_loss_sum / num_batches
    avg_resname_loss = resname_loss_sum / num_batches
    avg_resid_loss = resid_loss_sum / num_batches
    avg_coverage_loss = coverage_loss_sum / num_batches
    
    return {
        'loss': avg_loss,
        'coord_loss': avg_coord_loss,
        'resname_loss': avg_resname_loss,
        'resid_loss': avg_resid_loss,
        'coverage_loss': avg_coverage_loss
    }

def validate(model, dataloader, criterion, device, split='validation'):
    """Validate the model."""
    model.eval()
    total_loss = 0
    coord_loss_sum = 0
    resname_loss_sum = 0
    resid_loss_sum = 0
    coverage_loss_sum = 0
    
    all_coords_pred = []
    all_coords_true = []
    all_resname_pred = []
    all_resname_true = []
    all_resid_pred = []
    all_resid_true = []
    
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc=f"{split} Evaluation"):
            # Move data to device
            for key in batch_data:
                if isinstance(batch_data[key], torch.Tensor):
                    batch_data[key] = batch_data[key].to(device)
            
            # Forward pass
            predictions = model(batch_data, split=split)
            
            # Compute loss
            targets = {
                'coords': batch_data['coords'],
                'resnames': batch_data['resnames'],
                'resids': batch_data['resids']
            }
            
            loss_dict = criterion(
                predictions, 
                targets, 
                mask=batch_data['mask'],
                output_mask=batch_data.get('output_mask')
            )
            
            # Update metrics
            total_loss += loss_dict['total_loss'].item()
            coord_loss_sum += loss_dict['coord_loss'].item()
            resname_loss_sum += loss_dict['resname_loss'].item()
            resid_loss_sum += loss_dict['resid_loss'].item()
            coverage_loss_sum += loss_dict['coverage_loss'].item()
            
            # Collect predictions and targets for metrics
            mask = batch_data['mask'].bool()
            
            # Coordinates
            coords_pred = predictions['coords_pred']
            coords_true = batch_data['coords']
            
            # Handle different output shapes based on split
            if split == 'train':
                # For training, we only predict the first position
                coords_pred_flat = coords_pred[mask]
                coords_true_flat = coords_true[mask, 0]
            else:
                # For validation/test, we predict multiple positions
                output_mask = batch_data.get('output_mask', None)
                if output_mask is not None:
                    # Handle case where we have multiple positions
                    valid_mask = (output_mask > 0) & mask.unsqueeze(-1)
                    if valid_mask.dim() == coords_pred.dim() - 1:  # Add dimension if needed
                        valid_mask = valid_mask.unsqueeze(-1)
                    coords_pred_flat = coords_pred[valid_mask.expand_as(coords_pred)].view(-1, 3)
                    coords_true_flat = coords_true[valid_mask.expand_as(coords_true)].view(-1, 3)
                else:
                    # Handle single position case
                    if coords_pred.dim() > coords_true.dim():
                        # If predictions have extra dimension, take first position
                        coords_pred = coords_pred[..., 0, :]  # [batch, seq_len, 3]
                    coords_pred_flat = coords_pred[mask]
                    coords_true_flat = coords_true[mask]
            
            all_coords_pred.append(coords_pred_flat.cpu().numpy())
            all_coords_true.append(coords_true_flat.cpu().numpy())
            
            # Residue names
            resname_logits = predictions['resname_logits']
            resname_pred = torch.argmax(resname_logits, dim=-1)
            resname_true = batch_data['resnames']
            
            all_resname_pred.append(resname_pred[mask].cpu().numpy())
            all_resname_true.append(resname_true[mask].cpu().numpy())
            
            # Residue IDs
            resid_pred = torch.round(predictions['resid_pred'])
            resid_true = batch_data['resids']
            
            all_resid_pred.append(resid_pred[mask].cpu().numpy())
            all_resid_true.append(resid_true[mask].cpu().numpy())
    
    # Calculate average losses
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_coord_loss = coord_loss_sum / num_batches
    avg_resname_loss = resname_loss_sum / num_batches
    avg_resid_loss = resid_loss_sum / num_batches
    avg_coverage_loss = coverage_loss_sum / num_batches
    
    # Concatenate all predictions and targets
    all_coords_pred = np.concatenate(all_coords_pred) if all_coords_pred else np.array([])
    all_coords_true = np.concatenate(all_coords_true) if all_coords_true else np.array([])
    all_resname_pred = np.concatenate(all_resname_pred) if all_resname_pred else np.array([])
    all_resname_true = np.concatenate(all_resname_true) if all_resname_true else np.array([])
    all_resid_pred = np.concatenate(all_resid_pred) if all_resid_pred else np.array([])
    all_resid_true = np.concatenate(all_resid_true) if all_resid_true else np.array([])
    
    # Calculate metrics
    coord_rmse = float(np.sqrt(np.mean(np.sum((all_coords_pred - all_coords_true) ** 2, axis=-1)))) \
        if len(all_coords_pred) > 0 else 0.0
    resname_acc = float(np.mean(all_resname_pred == all_resname_true)) if len(all_resname_pred) > 0 else 0.0
    resid_mae = float(np.mean(np.abs(all_resid_pred - all_resid_true))) if len(all_resid_pred) > 0 else 0.0
    
    # Convert all metrics to native Python types
    metrics = {
        'loss': float(avg_loss),
        'coord_loss': float(avg_coord_loss),
        'resname_loss': float(avg_resname_loss),
        'resid_loss': float(avg_resid_loss),
        'coverage_loss': float(avg_coverage_loss),
        'coord_rmse': coord_rmse,
        'resname_acc': resname_acc,
        'resid_mae': resid_mae
    }
    
    return metrics

def save_checkpoint(model, optimizer, epoch, metrics, filename):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(model, optimizer, filename):
    """Load model checkpoint."""
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    metrics = checkpoint.get('metrics', {})
    print(f"Loaded checkpoint from epoch {epoch}")
    return epoch, metrics

def plot_learning_curves(train_metrics, val_metrics, save_path):
    """Plot and save learning curves."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot loss
    axes[0, 0].plot(train_metrics['loss'], label='Train')
    axes[0, 0].plot(val_metrics['loss'], label='Validation')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    
    # Plot coordinate loss
    axes[0, 1].plot(train_metrics['coord_loss'], label='Train')
    axes[0, 1].plot(val_metrics['coord_loss'], label='Validation')
    axes[0, 1].set_title('Coordinate Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    
    # Plot resname accuracy
    axes[1, 0].plot(val_metrics['resname_acc'], label='Validation')
    axes[1, 0].set_title('Residue Name Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    
    # Plot resid MAE
    axes[1, 1].plot(val_metrics['resid_mae'], label='Validation')
    axes[1, 1].set_title('Residue ID MAE')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('MAE')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def predict_and_save(model, dataloader, device, output_file, split='test'):
    """Generate predictions and save to a CSV file."""
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc=f"{split} Prediction"):
            # Move data to device
            for key in batch_data:
                if isinstance(batch_data[key], torch.Tensor):
                    batch_data[key] = batch_data[key].to(device)
            
            # Forward pass
            predictions = model(batch_data, split=split)
            
            # Extract predictions
            coords_pred = predictions['coords_pred']
            resname_logits = predictions['resname_logits']
            resname_pred = torch.argmax(resname_logits, dim=-1)
            resid_pred = torch.round(predictions['resid_pred'])
            
            # Convert to numpy
            coords_pred_np = coords_pred.cpu().numpy()
            resname_pred_np = resname_pred.cpu().numpy()
            resid_pred_np = resid_pred.cpu().numpy()
            
            # Get target IDs and masks
            target_ids = batch_data['target_id']
            mask = batch_data['mask'].cpu().numpy()
            
            # Create predictions for each sequence
            for i in range(len(target_ids)):
                target_id = target_ids[i]
                seq_len = int(mask[i].sum())
                
                # Extract valid predictions for this sequence
                if split == 'train':
                    # For training, we only predict the first position
                    seq_coords = coords_pred_np[i, :seq_len]
                    num_positions = 1
                else:
                    # For validation/test, we predict multiple positions
                    seq_coords = coords_pred_np[i, :seq_len]
                    num_positions = seq_coords.shape[1] if seq_coords.ndim > 2 else 1
                
                seq_resnames = resname_pred_np[i, :seq_len]
                seq_resids = resid_pred_np[i, :seq_len]
                
                # Convert resname IDs back to letters
                resname_letters = []
                for rid in seq_resnames:
                    if rid == 0:
                        resname_letters.append('A')
                    elif rid == 1:
                        resname_letters.append('C')
                    elif rid == 2:
                        resname_letters.append('G')
                    elif rid == 3:
                        resname_letters.append('U')
                    else:
                        resname_letters.append('X')
                
                # Create rows for this sequence
                for j in range(seq_len):
                    row = {'target_id': target_id, 'resname': resname_letters[j], 'resid': int(seq_resids[j])}
                    
                    # Add coordinates for each position
                    if split == 'train':
                        row['x_1'] = seq_coords[j, 0]
                        row['y_1'] = seq_coords[j, 1]
                        row['z_1'] = seq_coords[j, 2]
                    else:
                        for pos in range(num_positions):
                            row[f'x_{pos+1}'] = seq_coords[j, pos, 0] if seq_coords.ndim > 2 else seq_coords[j, 0]
                            row[f'y_{pos+1}'] = seq_coords[j, pos, 1] if seq_coords.ndim > 2 else seq_coords[j, 1]
                            row[f'z_{pos+1}'] = seq_coords[j, pos, 2] if seq_coords.ndim > 2 else seq_coords[j, 2]
                    
                    all_predictions.append(row)
    
    # Create DataFrame and save to CSV
    predictions_df = pd.DataFrame(all_predictions)
    predictions_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

def train(args):
    # Set device
    
    print(f"Using device: {device}")

    
    # Create output directory
    
    # Get dataloaders
    train_loader, val_loader = get_dataloaders(
        args.train_sequences,
        args.train_labels,
        args.val_sequences,
        args.val_labels,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create model
    model = RNAStructurePredictor(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    ).to(device)
    
    # Create loss function
    criterion = RNAStructureLoss(
        coord_weight=args.coord_weight,
        resname_weight=args.resname_weight,
        resid_weight=args.resid_weight,
        coverage_weight=args.coverage_weight
    ).to(device)
    
    # Create optimizer
    optimizer = FusedAdam(model.parameters(), lr=args.learning_rate)

    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )
    
    # Initialize tracking variables
    start_epoch = 0
    best_val_loss = float('inf')
    train_metrics_history = {
        'loss': [],
        'coord_loss': [],
        'resname_loss': [],
        'resid_loss': [],
        'coverage_loss': []
    }
    val_metrics_history = {
        'loss': [],
        'coord_loss': [],
        'resname_loss': [],
        'resid_loss': [],
        'coverage_loss': [],
        'coord_rmse': [],
        'resname_acc': [],
        'resid_mae': []
    }
    
    # Load checkpoint if provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        start_epoch, metrics = load_checkpoint(model, optimizer, args.checkpoint)
        if metrics:
            train_metrics_history = metrics.get('train', train_metrics_history)
            val_metrics_history = metrics.get('val', val_metrics_history)
            best_val_loss = min(val_metrics_history.get('loss', [float('inf')]))
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, split='train')
        print(f"Train Loss: {train_metrics['loss']:.4f}, Coord Loss: {train_metrics['coord_loss']:.4f}, "
              f"Resname Loss: {train_metrics['resname_loss']:.4f}, Resid Loss: {train_metrics['resid_loss']:.4f}")
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, split='validation')
        print(f"Val Loss: {val_metrics['loss']:.4f}, Coord RMSE: {val_metrics['coord_rmse']:.4f}, "
              f"Resname Acc: {val_metrics['resname_acc']:.4f}, Resid MAE: {val_metrics['resid_mae']:.4f}")
        
        # Update learning rate
        scheduler.step(val_metrics['loss'])
        
        # Update metrics history
        for key in train_metrics:
            train_metrics_history[key].append(train_metrics[key])
        
        for key in val_metrics:
            val_metrics_history[key].append(val_metrics[key])
        
        # Save checkpoint
        checkpoint_path = f"checkpoint_epoch_{epoch+1}.pt"
        metrics = {'train': train_metrics_history, 'val': val_metrics_history}
        save_checkpoint(model, optimizer, epoch+1, metrics, checkpoint_path)
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_model_path = "/kaggle/working/best_model.pt"
            save_checkpoint(model, optimizer, epoch+1, metrics, best_model_path)
            print(f"New best model saved with validation loss: {best_val_loss:.4f}")
        
        # Plot learning curves
        plot_path = "/kaggle/working/learning_curves.png"
        plot_learning_curves(train_metrics_history, val_metrics_history, plot_path)
        
        # Convert numpy numeric types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, (np.integer, np.floating, np.bool_)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy(x) for x in obj]
            return obj
        
        # Save metrics to JSON
        metrics_path = "/kaggle/working/metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(convert_numpy(metrics), f, indent=4)
    
    # Load best model for final prediction
    best_model_path = "/kaggle/working/best_model.pt"
    if os.path.exists(best_model_path):
        _, _ = load_checkpoint(model, optimizer, best_model_path)
    
    # Generate predictions for validation set
    val_predictions_path = "/kaggle/working/validation_predictions.csv"
    predict_and_save(model, val_loader, device, val_predictions_path, split='validation')
    
    # If test data is provided, generate predictions for test set
    if args.test_sequences:
    # Load test data
        test_sequences = pd.read_csv(args.test_sequences)
    
    # Create test dataset and dataloader
        test_dataset = RNADataset(
            sequences_df=test_sequences,
            labels_df=None,
            is_training=False,
            max_length=MAX_SEQ_LENGTH
        )
    
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_rna_batch,
            pin_memory=torch.cuda.is_available()
        )
            # Generate predictions
        test_predictions_path = "/kaggle/working/test_predictions.csv"
        predict_and_save(model, test_loader, device, test_predictions_path, split='test')
    return test_dataset, test_loader
