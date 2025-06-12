import os
import pandas as pd
import numpy as np
import subprocess
import time

def sample_training_data(input_file, output_file, num_samples=500):
    print(f"Sampling {num_samples} unique sequences from {input_file}...")
    df = pd.read_csv(input_file)
    unique_targets = df['target_id'].unique()
    if len(unique_targets) <= num_samples:
        sampled_targets = unique_targets
    else:
        sampled_targets = np.random.choice(unique_targets, size=num_samples, replace=False)
    sampled_df = df[df['target_id'].isin(sampled_targets)]
    sampled_df.to_csv(output_file, index=False)
    print(f"Sampled data saved to {output_file}")
    print(f"Original data had {len(unique_targets)} unique targets")
    print(f"Sampled data has {len(sampled_targets)} unique targets")
    return output_file

def run_training(num_sequences=5135, sample_data=True):
    train_sequences = "/kaggle/input/modified-stanford-rna-3d-folding/train_sequences.modified.csv"
    train_labels = "/kaggle/input/modified-stanford-rna-3d-folding/train_labels.modified.csv"
    
    if sample_data:
        sampled_sequences = f"/kaggle/working/sampled_train_sequences_{num_sequences}.csv"
        if not os.path.exists(sampled_sequences):
            train_sequences = sample_training_data(train_sequences, sampled_sequences, num_sequences)
        else:
            train_sequences = sampled_sequences
            print(f"Using existing sampled data: {sampled_sequences}")



        
    start_time = time.time()
    args = Args()
    train(args)
    total_time = time.time() - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")

print(f"Is CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
run_training(sample_data=False)

our_test_sequences_path = '/kaggle/input/modified-stanford-rna-3d-folding/validation_sequences.modified.csv'
test_sequences = pd.read_csv(our_test_sequences_path)
max_length = test_sequences['target_id'].astype(str).map(len).max()
args = Args()
batch_size = args.batch_size
num_workers = 4
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
predict_for_test(test_loader, model_path="/kaggle/working/best_model.pt", test_sequences_path="/kaggle/input/stanford-rna-3d-folding/test_sequences.csv", output_file="/kaggle/working/predictions.npy", batch_size=16)


create_submission(
    predictions_path="/kaggle/working/predictions.npy",
    output_file="/kaggle/working/submission.csv",
    target_ids_path="/kaggle/input/stanford-rna-3d-folding/test_sequences.csv", # This is the path to the original test sequences for target IDs
    sample_submission_path='/kaggle/input/stanford-rna-3d-folding/sample_submission.csv'
)
submission_df = pd.read_csv('submission.csv')
tm_score = score(validation_labels, submission_df.copy(), row_id_column_name='ID')
print(f"Average TM-score: {tm_score:.4f}")
