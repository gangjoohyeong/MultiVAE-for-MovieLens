import torch
import numpy as np
import pandas as pd
import time
import wandb
from .utils import naive_sparse2tensor, get_logger, logging_conf
from .metric import NDCG_binary_at_k_batch, Recall_at_k_batch

logger = get_logger(logger_conf=logging_conf)

def run(args, model, criterion, optimizer, train_data, vad_data_tr, vad_data_te, test_data_tr, test_data_te, N, idxlist):
    global update_count
    best_r10 = -np.inf
    update_count = 0

    for epoch in range(1, args.epochs + 1):
        if args.model == 'MultiVAE':
            is_VAE = True
        elif args.model == 'MultiDAE':
            is_VAE = False
        train(args, model, criterion, optimizer, is_VAE, N, idxlist, train_data, epoch)
        val_loss, n100, r10, r20, r50 = evaluate(args, model, criterion, vad_data_tr, vad_data_te, is_VAE, N)
        logger.info('🚀 | Epoch: {:2d} | valid loss {:4.2f} | '
                'n100 {:5.3f} | r10 {:5.3f} | r20 {:5.3f} | r50 {:5.3f}'.format(
                    epoch, val_loss, n100, r10, r20, r50))
        wandb.log({"epoch": epoch, "val_loss": val_loss, "val_n100": n100, "val_r10": r10,"val_r20": r20,"val_r50": r50})

        n_iter = epoch * len(range(0, N, args.batch_size))


        if r10 > best_r10:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_r10 = r10



    with open(args.save, 'rb') as f:
        model = torch.load(f)

    test_loss, n100, r10, r20, r50 = evaluate(args, model, criterion, test_data_tr, test_data_te, is_VAE, N)
    logger.info('🚀 | Finish | test loss {:4.2f} | n100 {:4.2f} | r10 {:4.2f} | r20 {:4.2f} | '
            'r50 {:4.2f}'.format(test_loss, n100, r10, r20, r50))
    wandb.log({"test_loss": test_loss, "test_n100": n100, "test_r10": r10,"test_r20": r20,"test_r50": r50})




def train(args, model, criterion, optimizer, is_VAE, N, idxlist, train_data, epoch):
    # Turn on training mode
    model.train()
    train_loss = 0.0
    start_time = time.time()
    global update_count

    np.random.shuffle(idxlist)
    
    for batch_idx, start_idx in enumerate(range(0, N, args.batch_size)):
        end_idx = min(start_idx + args.batch_size, N)
        data = train_data[idxlist[start_idx:end_idx]]
        data = naive_sparse2tensor(data).to(args.device)
        optimizer.zero_grad()

        if is_VAE:
            if args.total_anneal_steps > 0:
                anneal = min(args.anneal_cap, 
                                1. * update_count / args.total_anneal_steps)
            else:
                anneal = args.anneal_cap

            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            
            loss = criterion(recon_batch, data, mu, logvar, anneal)
        else:
            recon_batch = model(data)
            loss = criterion(recon_batch, data)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        update_count += 1

        if batch_idx % args.log_interval == 0 and batch_idx > 0:
            elapsed = time.time() - start_time
            logger.info('🚀 | Epoch {:3d} | {:4d}/{:4d} batches | ms/batch {:4.2f} | '
                    'loss {:4.2f}'.format(
                        epoch, batch_idx, len(range(0, N, args.batch_size)),
                        elapsed * 1000 / args.log_interval,
                        train_loss / args.log_interval))
            

            start_time = time.time()
            train_loss = 0.0


def evaluate(args, model, criterion, data_tr, data_te, is_VAE, N):
    # Turn on evaluation mode
    model.eval()
    total_loss = 0.0
    global update_count
    e_idxlist = list(range(data_tr.shape[0]))
    e_N = data_tr.shape[0]
    n100_list = []
    r10_list = []
    r20_list = []
    r50_list = []
    
    with torch.no_grad():
        for start_idx in range(0, e_N, args.batch_size):
            end_idx = min(start_idx + args.batch_size, N)
            data = data_tr[e_idxlist[start_idx:end_idx]]
            heldout_data = data_te[e_idxlist[start_idx:end_idx]]

            data_tensor = naive_sparse2tensor(data).to(args.device)
            if is_VAE :
                if args.total_anneal_steps > 0:
                    anneal = min(args.anneal_cap, 
                                  1. * update_count / args.total_anneal_steps)
                else:
                    anneal = args.anneal_cap
            
                recon_batch, mu, logvar = model(data_tensor)
                loss = criterion(recon_batch, data_tensor, mu, logvar, anneal)

                
            else :
              recon_batch = model(data_tensor)
              loss = criterion(recon_batch, data_tensor)

            total_loss += loss.item()

            # Exclude examples from training set
            recon_batch = recon_batch.cpu().numpy()
            recon_batch[data.nonzero()] = -np.inf

            n100 = NDCG_binary_at_k_batch(recon_batch, heldout_data, 100)
            r10 = Recall_at_k_batch(recon_batch, heldout_data, 10)
            r20 = Recall_at_k_batch(recon_batch, heldout_data, 20)
            r50 = Recall_at_k_batch(recon_batch, heldout_data, 50)

            n100_list.append(n100)
            r10_list.append(r10)
            r20_list.append(r20)
            r50_list.append(r50)
 
    total_loss /= len(range(0, e_N, args.batch_size))
    n100_list = np.concatenate(n100_list)
    r10_list = np.concatenate(r10_list)
    r20_list = np.concatenate(r20_list)
    r50_list = np.concatenate(r50_list)

    return total_loss, np.mean(n100_list), np.mean(r10_list), np.mean(r20_list), np.mean(r50_list)




def submission(args, model, data_tr):
    model.eval()
    
    data = data_tr
    top_items = []
    
    with torch.no_grad():
        data_tensor = naive_sparse2tensor(data).to(args.device)
        
        predictions = model(data_tensor)
        if args.model == 'MultiVAE':
            predictions = predictions[0].cpu().numpy()    
        elif args.model == 'MultiDAE':
            predictions = predictions.cpu().numpy()    
        predictions[data.nonzero()] = -np.inf
        
        _, top_indices = torch.topk(torch.from_numpy(predictions).float().to(args.device), k=10, dim=1)
        for user_idx, item_indices in enumerate(top_indices):
            user_id = user_idx
            for item_idx in item_indices:
                item_id = item_idx.item()
                top_items.append((user_id, item_id))

    return top_items