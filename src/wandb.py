def wandb_settings(args):
    wandb_config = {
        "epochs" : args.epochs,
        "learning_rate" : args.lr,
        "weight_decay_coefficient" : args.wd,
        "batch_size" : args.batch_size,
        "total_anneal_steps" : args.total_anneal_steps,
        "anneal_cap" : args.anneal_cap,
        "seed" : args.seed,
        "cuda" : args.cuda,
        "log_interval" : args.log_interval,
    }
    
    model_name = args.model
    author = 'bles'
    project = 'movierec'
    entity = 'new-recs'
    
    return wandb_config, model_name, author, project, entity