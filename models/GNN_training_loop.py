import os
from GNN_dataloader import BICIMADloader
from GNN_utils import StepLR,train,validate,save_checkpoint
from torch.utils.tensorboard import SummaryWriter
import torch
from torch import nn
import logging
logging.basicConfig()
logging.root.setLevel(logging.NOTSET)

def train_loop(args):

    best_val_loss = 1000


    #load datasets
    loader = BICIMADloader(raw_data_dir=args.root, raw_data_name=args.training_file,adj_mat_name="adj_mat_inverted_dist.npy")
    train_dataset = loader.get_dataset(target_variable=args.target_variable,slide=168,num_timesteps_in=168, num_timesteps_out=168)


    loader = BICIMADloader(raw_data_dir=args.root, raw_data_name=args.validation_file,adj_mat_name="adj_mat_inverted_dist.npy")
    val_dataset = loader.get_dataset(target_variable=args.target_variable,slide=168,num_timesteps_in=168, num_timesteps_out=168)

    writer = SummaryWriter(log_dir=os.path.join("tensorboard", args.experiment_name))

    #calling device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
            logging.info("Use GPU for training : CUDA version {}, device:{}".format(torch.version.cuda,device))

    if not torch.cuda.is_available():
            logging.info('using CPU, this will be slow')

    # create model
    model = args.model.to(device)

    logging.info("=> Loading Teacher model '{}'".format(model._get_name()))

    # define loss function (criterion), optimizer, and learning rate scheduler

    class RMSELoss(torch.nn.Module):
        def __init__(self):
            super(RMSELoss,self).__init__()

        def forward(self,x,y):
            criterion = nn.MSELoss(reduction="mean")
            loss = torch.sqrt(criterion(x, y))
            return loss

    criterion = RMSELoss().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)#,weight_decay=args.weight_decay)

    scheduler = StepLR(optimizer, step_size=args.LR_step_size, gamma=args.LR_step_gamma)#0.1)
    #scheduler = CosineAnnealingWarmRestarts(optimizer, T_0 = args.epochs + 1, T_mult=1, eta_min=args.min_lr, verbose=False)

    for epoch in range(args.start_epoch, args.epochs):

            # train for one epoch
            writer = train(train_dataset, model, criterion, optimizer, epoch, device, writer, args)

            # evaluate on validation set # change this as this is the measure of whether the model is good or not versus the training/ validation set
            
            if epoch % 3==0:
                val_loss, writer = validate(val_dataset, model, criterion,device, writer, epoch, args)
            
            scheduler.step()
            
            # remember best acc@1 and save checkpoint
            is_best = val_loss < best_val_loss
            best_val_loss = min(val_loss, best_val_loss)


            save_checkpoint(
                    state={'epoch': epoch + 1,
                    'model': args.model,
                    'best_val_loss': best_val_loss,
                    'optimizer' : optimizer.state_dict(),
                    'scheduler' : scheduler.state_dict()
                    }, 
                    is_best=is_best,
                    args=args,
                    filename=os.path.join("checkpoints",args.experiment_name +'_checkpoint.pth.tar'))
    writer.flush()


