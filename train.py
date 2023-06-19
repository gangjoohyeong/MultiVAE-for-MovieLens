import os
import wandb
import numpy as np
import pandas as pd
import torch
import time
from datetime import datetime, timezone, timedelta

from src.utils import set_seeds, get_logger, logging_conf
from src.args import parse_args
from src.wandb import wandb_settings
from src.dataloader import Preprocess, Dataloader
from src.model import MultiDAE, MultiVAE
from src.optimizer import get_optimizer
from src.criterion import loss_function_dae, loss_function_vae
from src.trainer import run, train, evaluate, submission

logger = get_logger(logger_conf=logging_conf)

def main(args):
    set_seeds(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("🐰 TEAM NewRecs")
    if args.mode == 'tuning':
        logger.info(f"Mode: {args.mode} - Not create submission file | Traning Dataset: Split")
    elif args.mode == 'submission':
        logger.info(f"Mode: {args.mode} - Create submission file | Traning Dataset: Full")
    
    # Weights & Biases Settings
    logger.info("1. Weights & Biases Settings ...")
    wandb.login()
    wandb_config, model_name, author, project, entity = wandb_settings(args)
    wandb.init(project=project, entity=entity, config=wandb_config)
    
    now = datetime.now(timezone(timedelta(hours=9)))
    wandb.run.name = f"{model_name}_{author}_{args.mode} {now.strftime('%m/%d %H:%M')}"
    
    # DATA Preprocessing
    logger.info("2. Data Preprocessing ...")
    raw_data, unique_uid, tr_users, vd_users, te_users, n_users = Preprocess(args).load_data_from_file(args)
    unique_sid, show2id, profile2id, id2show, id2profile = Preprocess(args).data_split(args, raw_data, unique_uid, tr_users, vd_users, te_users)
    
    # Data Loading
    logger.info("3. Data Loading ...")
    loader = Dataloader(args.data)
    n_items = loader.load_n_items()
    train_data = loader.load_data('train')
    vad_data_tr, vad_data_te = loader.load_data('validation')
    test_data_tr, test_data_te = loader.load_data('test')

    N = train_data.shape[0]
    idxlist = list(range(N))

    # Build Model
    logger.info("4. Model Buliding ...")
    p_dims = [200, 600, n_items]
    if args.model == 'MultiVAE':
        model = MultiVAE
    elif args.model == 'MultiDAE':
        model = MultiDAE
    
    model = model(p_dims).to(args.device)
    optimizer = get_optimizer(model, args)
    if args.model == 'MultiVAE':
        criterion = loss_function_vae
    elif args.model == 'MultiDAE':
        criterion = loss_function_dae
    
    # Training
    logger.info("5. Training ...")
    run(args, model, criterion, optimizer, train_data, vad_data_tr, vad_data_te, test_data_tr, test_data_te, N, idxlist)
    
    if args.mode == 'submission':
        # Create Submission File
        logger.info("6. Creating Submission File ...")
        top_items = submission(args, model, train_data)
        result = pd.DataFrame(top_items, columns=['user', 'item'])

        result['user'] = result['user'].apply(lambda x : id2profile[x])
        result['item'] = result['item'].apply(lambda x : id2show[x])
        result = result.sort_values(by='user')
        
        KST = timezone(timedelta(hours=9))
        record_time = datetime.now(KST)
        write_path = os.path.join(f"./submit/{args.model}_submission_{record_time.strftime('%Y-%m-%d_%H-%M-%S')}.csv")
        result.to_csv(write_path, index=False)
    
    logger.info("💫 Complete!")
if __name__ == "__main__":
    args = parse_args()
    main(args)