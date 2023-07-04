import os
import zipfile
import numpy as np
import torch
from torch_geometric.utils import dense_to_sparse
from torch_geometric_temporal.signal import StaticGraphTemporalSignal
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

class BICIMADloader(object):
    """A traffic forecasting dataset based on Los Angeles
    Metropolitan traffic conditions. The dataset contains traffic
    readings collected from 207 loop detectors on highways in Los Angeles
    County in aggregated 5 minute intervals for 4 months between March 2012
    to June 2012.

    For further details on the version of the sensor network and
    discretization see: `"Diffusion Convolutional Recurrent Neural Network:
    Data-Driven Traffic Forecasting" <https://arxiv.org/abs/1707.01926>`_
    """

    def __init__(self, raw_data_dir, raw_data_name, adj_mat_name):
        self.raw_data_dir = raw_data_dir
        self.raw_data_name = raw_data_name
        self.adj_mat_name = adj_mat_name
        self._load_data()

    
    def _load_data(self):
        #loads data, already preprocessed and normalized!
        A = np.load(os.path.join(self.raw_data_dir, self.adj_mat_name))
        X = np.load(os.path.join(self.raw_data_dir, self.raw_data_name)).transpose(
            (1, 2, 0)
        ).astype(np.float32)

        self.A = torch.from_numpy(A)
        self.X = torch.from_numpy(X)
        

    def _get_edges_and_weights(self):
        #gets adjacency matrix and transforms it to correct format
        edge_indices, values = dense_to_sparse(self.A)
        edge_indices = edge_indices.numpy()
        values = values.numpy()
        self.edges = edge_indices
        self.edge_weights = values

    def _generate_task(self,target_variable,slide = 1, num_timesteps_in: int = 24, num_timesteps_out: int = 24):
        """Uses the node features of the graph and generates a feature/target
        relationship of the shape
        (num_nodes, num_node_features, num_timesteps_in) -> (num_nodes, num_timesteps_out)
        predicting the average traffic speed using num_timesteps_in to predict the
        traffic conditions in the next num_timesteps_out

        Args:
            num_timesteps_in (int): number of timesteps the sequence model sees
            num_timesteps_out (int): number of timesteps the sequence model has to predict
        """
        indices = [
            (i*slide, i*slide + (num_timesteps_in + num_timesteps_out))
            for i in range(int((self.X.shape[2] - (num_timesteps_in + num_timesteps_out))/slide+1))
        ]

        # Generate observations
        features, target = [], []
        if target_variable == "plugs":
            target_index = 0
        if target_variable == "unplugs":
            target_index = 1
        for i, j in indices:
            #it assumes that first column will always be the target
            features.append((self.X[:, :, i : i + num_timesteps_in]).numpy())
            target.append((self.X[:, target_index, i + num_timesteps_in : j]).numpy())

        self.features = features
        self.targets = target

    def get_dataset(
        self, target_variable, slide = 1,num_timesteps_in: int = 24, num_timesteps_out: int = 24
    ) -> StaticGraphTemporalSignal:
        """Returns data iterator for METR-LA dataset as an instance of the
        static graph temporal signal class.

        Return types:
            * **dataset** *(StaticGraphTemporalSignal)* - The METR-LA traffic
                forecasting dataset.
        """
        self._get_edges_and_weights()
        self._generate_task(target_variable,slide,num_timesteps_in, num_timesteps_out)
        dataset = StaticGraphTemporalSignal(
            self.edges, self.edge_weights, self.features, self.targets
        )

        return dataset


