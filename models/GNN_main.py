#args
import os
import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import A3TGCN
from GNN_training_loop import train_loop
import os

import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import os
from GNN_dataloader import BICIMADloader
from GNN_utils import StepLR,train,validate,save_checkpoint
from torch.utils.tensorboard import SummaryWriter
import torch
from torch import nn
import logging
logging.basicConfig()
logging.root.setLevel(logging.NOTSET)
import datetime

import joblib

class argclass(object):
    def __init__(self, *initial_data,**kwargs):
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self,node_features, periods):
        super().__init__()
        self.conv1 = GCNConv(node_features, 16)
        self.conv2 = GCNConv(16, periods)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

class TemporalGNN(torch.nn.Module):
    def __init__(self, node_features, periods):
        super(TemporalGNN, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        self.tgnn = A3TGCN(in_channels=node_features,
                           out_channels=32,
                           periods=periods)
        # Equals single-shot prediction
        self.linear = torch.nn.Linear(32, periods)

    def forward(self, x, edge_index):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        h = self.tgnn(x, edge_index)
        h = F.relu(h)
        h = self.linear(h)
        return h

class RMSELoss(torch.nn.Module):
        def __init__(self):
            super(RMSELoss,self).__init__()

        def forward(self,x,y):
            criterion = nn.MSELoss(reduction="mean")
            loss = torch.sqrt(criterion(x, y))
            return loss


def objective(trial, args):
    """This function is only used for Optuna hyperparameter search"""
    #calling device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
            logging.info("Use GPU for training : CUDA version {}, device:{}".format(torch.version.cuda,device))

    if not torch.cuda.is_available():
            logging.info('using CPU, this will be slow')
    
    hours = {"168":168,
             "336":336}

    ### setting parameters for search ###
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])    
    
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-1)
    LR_step_size = 10#trial.suggest_int("lr_step_size",1,5)
    LR_step_gamma = 0.7#trial.suggest_float("lr_step_gamma",0.3,0.9)
    in_hours = trial.suggest_categorical("in_hours", ["168","336"])
    in_hours = hours[in_hours]

    #out_hours = trial.suggest_categorical("out_hours", ["168","336"])
    out_hours = 168#hours[out_hours]
    #weight_decay = trial.suggest_loguniform("weight_decay", 1e-4, 1e-1)

     # create model
    model = TemporalGNN(node_features=32, periods=out_hours).to(device)
    logging.info("=> Loading Teacher model '{}'".format(model._get_name()))

    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)#, weight_decay=weight_decay)

    time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    experiment_name = args.experiment_name + time_stamp
    model_name = experiment_name + '_checkpoint.pth.tar'

    # Training of the model.
    best_val_loss = 1000

    #load datasets
    loader = BICIMADloader(raw_data_dir=args.root, raw_data_name=args.training_file,adj_mat_name="adj_mat_inverted_dist.npy")
    train_dataset = loader.get_dataset(target_variable=args.target_variable,slide=168,num_timesteps_in=in_hours, num_timesteps_out=out_hours)


    loader = BICIMADloader(raw_data_dir=args.root, raw_data_name=args.validation_file,adj_mat_name="adj_mat_inverted_dist.npy")
    val_dataset = loader.get_dataset(target_variable=args.target_variable,slide=168,num_timesteps_in=in_hours, num_timesteps_out=out_hours)

    writer = SummaryWriter(log_dir=os.path.join("tensorboard", experiment_name))


    # define loss function (criterion), optimizer, and learning rate scheduler

    criterion = RMSELoss().to(device)

    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    scheduler = StepLR(optimizer, step_size=LR_step_size, gamma=LR_step_gamma)#0.1)
    #scheduler = CosineAnnealingWarmRestarts(optimizer, T_0 = args.epochs + 1, T_mult=1, eta_min=args.min_lr, verbose=False)

    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        writer = train(train_dataset, model, criterion, optimizer, epoch, device, writer, args)

        # evaluate on validation set # change this as this is the measure of whether the model is good or not versus the training/ validation set
        val_loss, writer = validate(val_dataset, model, criterion,device, writer, epoch, args)
        
        scheduler.step()
        
        # remember best acc@1 and save checkpoint
        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)

        if is_best:
             torch.save(model.state_dict(),os.path.join("checkpoints",model_name +'_checkpoint.pth.tar'))

        """save_checkpoint(
                state={'epoch': epoch + 1,
                'model': args.model,
                'best_val_loss': best_val_loss,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()
                }, 
                is_best=is_best,
                args=args,
                filename=os.path.join("checkpoints",model_name +'_checkpoint.pth'))"""
    
        #PRUNING
        trial.report(val_loss, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            writer.flush()
            raise optuna.exceptions.TrialPruned()
        
    writer.flush()

    return val_loss



args_2022_plugs_params={"root":os.path.join("GNN_data"),
              "training_file":"train_GNN_2022_std.npy",
              "validation_file":"test_GNN_2022_std.npy",
              "lr":0.001,
              "LR_step_size":10,
              "LR_step_gamma":0.5,
              "target_variable":"plugs",
             'epochs':30,
             'start_epoch':0,
             'print_freq':1,
             'model':TemporalGNN(node_features=32, periods=168),
             "experiment_name":"2022_plugs",
            }

args_2022_unplugs_params={"root":os.path.join("GNN_data"),
              "training_file":"train_GNN_2022_std.npy",
              "validation_file":"test_GNN_2022_std.npy",
              "lr":0.001,
              "LR_step_size":10,
              "LR_step_gamma":0.5,
              "target_variable":"unplugs",
             'epochs':30,
             'start_epoch':0,
             'print_freq':1,
             'model':TemporalGNN(node_features=32, periods=168),
             "experiment_name":"2022_unplugs",
            }

args_2022_plugs_lagged_weather_params={"root":os.path.join("GNN_data"),
              "training_file":"train_GNN_2022_weather_lagged_std.npy",
              "validation_file":"test_GNN_2022_weather_lagged_std.npy",
              "lr":0.001,
              "LR_step_size":10,
              "LR_step_gamma":0.5,
              "target_variable":"plugs",
             'epochs':30,
             'start_epoch':0,
             'print_freq':1,
             'model':TemporalGNN(node_features=32, periods=168),
             "experiment_name":"2022_plugs_weather_lagged",
            }

args_2022_unplugs_lagged_weather_params={"root":os.path.join("GNN_data"),
              "training_file":"train_GNN_2022_weather_lagged_std.npy",
              "validation_file":"test_GNN_2022_weather_lagged_std.npy",
              "lr":0.001,
              "LR_step_size":10,
              "LR_step_gamma":0.5,
              "target_variable":"unplugs",
             'epochs':30,
             'start_epoch':0,
             'print_freq':1,
             'model':TemporalGNN(node_features=32, periods=168),
             "experiment_name":"2022_unplugs_weather_lagged",
            }

args_innova_plugs_params={"root":os.path.join("GNN_data"),
              "training_file":"train_GNN_innova_std.npy",
              "validation_file":"test_GNN_innova_std.npy",
              "lr":0.001,
              "LR_step_size":10,
              "LR_step_gamma":0.5,
              "target_variable":"plugs",
             'epochs':30,
             'start_epoch':0,
             'print_freq':1,
             'model':TemporalGNN(node_features=32, periods=168),
             "experiment_name":"innova_plugs",
            }

args_innova_unplugs_params={"root":os.path.join("GNN_data"),
              "training_file":"train_GNN_innova_std.npy",
              "validation_file":"test_GNN_innova_std.npy",
              "lr":0.001,
              "LR_step_size":10,
              "LR_step_gamma":0.5,
              "target_variable":"unplugs",
             'epochs':30,
             'start_epoch':0,
             'print_freq':1,
             'model':TemporalGNN(node_features=32, periods=168),
             "experiment_name":"innova_unplugs",
            }

args_innova_plugs_lagged_weather_params={"root":os.path.join("GNN_data"),
              "training_file":"train_GNN_innova_weather_lagged_std.npy",
              "validation_file":"test_GNN_innova_weather_lagged_std.npy",
              "lr":0.001,
              "LR_step_size":10,
              "LR_step_gamma":0.5,
              "target_variable":"plugs",
             'epochs':30,
             'start_epoch':0,
             'print_freq':1,
             'model':TemporalGNN(node_features=32, periods=168),
             "experiment_name":"innova_plugs_weather_lagged",
            }

args_innova_unplugs_lagged_weather_params={"root":os.path.join("GNN_data"),
              "training_file":"train_GNN_innova_weather_lagged_std.npy",
              "validation_file":"test_GNN_innova_weather_lagged_std.npy",
              "lr":0.001,
              "LR_step_size":10,
              "LR_step_gamma":0.5,
              "target_variable":"unplugs",
             'epochs':30,
             'start_epoch':0,
             'print_freq':1,
             'model':TemporalGNN(node_features=32, periods=168),
             "experiment_name":"innova_unplugs_weather_lagged",
            }

args_2022_plugs=argclass(args_2022_plugs_params)
args_2022_unplugs=argclass(args_2022_unplugs_params)
args_2022_plugs_lagged_weather=argclass(args_2022_plugs_lagged_weather_params)
args_2022_unplugs_lagged_weather=argclass(args_2022_unplugs_lagged_weather_params)
args_innova_plugs=argclass(args_innova_plugs_params)
args_innova_unplugs=argclass(args_innova_unplugs_params)
args_innova_plugs_lagged_weather=argclass(args_innova_plugs_lagged_weather_params)
args_innova_unplugs_lagged_weather=argclass(args_innova_unplugs_lagged_weather_params)


if __name__ == "__main__":
    train_loop(args_2022_plugs_lagged_weather)
    #train_loop(args_2022_plugs)
    
    #train_loop(args_2022_unplugs)
    train_loop(args_2022_unplugs_lagged_weather)
    #train_loop(args_innova_plugs)
    #train_loop(args_innova_unplugs)
    

"""
studies = [args1,args2]

if __name__ == "__main__":
    for args in studies:
        study = optuna.create_study(study_name=args.experiment_name,direction="minimize")
        study.optimize(lambda trial:objective(trial=trial,args=args), n_trials=100, timeout=5*3600)#7 hours

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        joblib.dump(study, os.path.join("studies",args.experiment_name+".pkl"))




"""
