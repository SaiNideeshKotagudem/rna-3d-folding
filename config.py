class Args:
    pass
train_sequences = "/kaggle/input/modified-stanford-rna-3d-folding/train_sequences.modified.csv"
train_labels= "/kaggle/input/modified-stanford-rna-3d-folding/train_labels.modified.csv"
Args.train_sequences = train_sequences
Args.train_labels = train_labels
Args.val_sequences = "/kaggle/input/modified-stanford-rna-3d-folding/validation_sequences.modified.csv"
Args.val_labels = "/kaggle/input/modified-stanford-rna-3d-folding/validation_labels.modified.csv"
Args.test_sequences = "/kaggle/input/modified-stanford-rna-3d-folding/test_sequences.csv"

    # Model parameters
Args.embedding_dim = 128
Args.hidden_dim = 256
Args.dropout = 0.1

    # Training parameters
Args.batch_size = 16
Args.epochs = 100
Args.learning_rate = 0.001
Args.num_workers = 4

    # Loss weights
Args.coord_weight = 1.0
Args.resname_weight = 0.5
Args.resid_weight = 0.5
Args.coverage_weight = 1.0

    # Output and checkpointing
Args.checkpoint = ""
