import torch
from torch.optim import Adam, AdamW


def get_optimizer(model, args):
    if args.optimizer == "adam":
        optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=args.wd)
        
    elif args.optimizer == "adamW":
        optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=args.wd)
    optimizer.zero_grad()
    return optimizer