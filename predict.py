predict_for_test(test_loader, model_path="/kaggle/working/best_model.pt", test_sequences_path="/kaggle/input/stanford-rna-3d-folding/test_sequences.csv", output_file="/kaggle/working/predictions.npy", batch_size=16)


create_submission(
    predictions_path="/kaggle/working/predictions.npy",
    output_file="/kaggle/working/submission.csv",
    target_ids_path="/kaggle/input/stanford-rna-3d-folding/test_sequences.csv", # This is the path to the original test sequences for target IDs
    sample_submission_path='/kaggle/input/stanford-rna-3d-folding/sample_submission.csv'
)
