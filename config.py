from dataclasses import dataclass


@dataclass
class Config:
    seq_len = 100
    stride = 1
    cond_mode = "bucket"
    n_buckets = 3

    z_dim = 32
    hidden_dim = 64
    n_layers = 2
    dropout = 0.1

    batch_size = 256
    lr_g = 2e-4
    lr_d = 2e-4
    epochs = 50

    label_smooth_real = 0.9
    grad_clip = 1.0
    feature_matching = True
    fm_weight = 5.0

    seed = 42
    ckpt_dir = "checkpoints"
