import os
import os.path as osp
from typing import Optional, Callable, Tuple, Dict
import warnings
import numpy as np
# PyTorch
import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import ModuleList, Sequential, Embedding, Linear
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
# PyG
from torch_geometric.io import fs
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, SumAggregation
from torch_geometric.nn.resolver import aggregation_resolver as aggr_resolver
from torch_geometric.nn import GCNConv, GINConv, GINEConv, GATv2Conv, MLP, SAGEConv
from torch_geometric.data import Dataset, download_url, extract_zip
from torch_geometric.typing import OptTensor
# Lightning
import lightning as L
from utils.functions import RSE_loss
from utils.schnet import RadiusInteractionGraph, InteractionBlock, CFConv, GaussianSmearing, ShiftedSoftplus

class GNN_model(L.LightningModule):
    '''
        Graph neural network class to run various types of GNNs
    '''
    def __init__(self, num_tasks, num_layers=4, in_channels=[33,100,100,100], out_channels=[100,100,100,100],
                 gnn_type='gcn', heads=None, drop_ratio=0.5, graph_pooling="sum", learning_rate=0.01):
        '''
            Args:
                num_tasks (int): number of labels to be predicted
                num_layers (int): number of layers in the NN
                in_channels (list): size of each input layer
                out_channels (list): size of each output layer
                gnn_type (str): the specific GNN to use
                heads (int): number of heads
                drop_ratio (float): the drop_ratio 
                graph_pooling (str): the pooling function for the GNN
                learning_rate (float): the learning rate for model
        '''

        super(GNN_model, self).__init__()

        self.num_tasks = num_tasks
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gnn_type = gnn_type
        self.heads = heads
        self.drop_ratio = drop_ratio
        self.graph_pooling = graph_pooling
        self.lr = learning_rate

        self._initialize_weights()
        self.save_hyperparameters()

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        # Sanity check number of layers
        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        
        # Choose the selected pooling function
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        else:
            raise ValueError("Invalid graph pooling type")

        # Set up GNN layers
        # Select the messgage passing layer
        for i, in_c, out_c in zip(range(self.num_layers), self.in_channels, self.out_channels):
            if gnn_type == "gin":
                mlp = MLP([in_c, in_c, out_c])
                self.convs.append(GINConv(nn=mlp, train_eps=False))
            elif gnn_type == "gine":
                mlp = MLP([in_c, out_c, out_c])
                self.convs.append(GINEConv(nn=mlp, eps=0.5, train_eps=True, edge_dim=5))
            elif gnn_type == "gcn":
                self.convs.append(GCNConv(in_c, out_c))
            elif gnn_type == "gat":
                if i == 1:
                    self.convs.append(GATv2Conv(int(in_c), out_c, heads=int(heads), edge_dim=5))
                else:
                    self.convs.append(GATv2Conv(int(in_c*heads), out_c, heads=int(heads), edge_dim=5))
            elif gnn_type == "sage":
                self.convs.append(SAGEConv(in_c, out_c, aggr='mean'))
            else:
                ValueError("Undefiend GNN type called {gnn_type}")

            self.batch_norms.append(torch.nn.BatchNorm1d(out_c))

        # Final linear layer
        self.graph_pred_linear = torch.nn.Linear(self.out_channels[-1], self.num_tasks)      	


    def forward(self, batched_data):

        x = batched_data.x
        edge_index = batched_data.edge_index
        edge_attr = batched_data.edge_attr

        # Create list of features
        h_list = [x]

        # Pass through GNN layers
        for layer in range(self.num_layers):
            # Use edge_attr for required gnn types
            if self.gnn_type == 'gine' or self.gnn_type == 'gat':
                h = self.convs[layer](h_list[layer], edge_index, edge_attr=edge_attr)
            else:
                h = self.convs[layer](h_list[layer], edge_index)
            # Batch normalization
            h = self.batch_norms[layer](h)
            # Dropout with/without relu
            if layer == self.num_layers - 1:
                # Remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            h_list.append(h)
        # Final node representation from GNN
        node_rep = h_list[-1]
        # Pool all atom node representation for grap rep
        graph_embedding = self.pool(node_rep, batched_data.batch)
        # Activation function
        p = torch.nn.LeakyReLU(0.1)
        # Pass through linear layer
        out = p(self.graph_pred_linear(graph_embedding))

        return out
    
    def training_step(self, train_batch, batch_idx):
        data = train_batch
        pred = self(data)
        loss = nn.MSELoss()(pred.flatten(), data.spectrum)
        self.log('train_loss', loss, on_epoch=True, batch_size=data.num_graphs)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        data = val_batch
        pred = self(data)
        loss = nn.MSELoss()(pred.flatten(), data.spectrum)
        self.log('val_loss', loss, on_epoch=True, batch_size=data.num_graphs)
        return loss
    
    def test_step(self, test_batch, batch_idx):
        data = test_batch
        pred = self(data)
        loss = F.mse_loss(pred.flatten(), data.spectrum)
        rse_loss = RSE_loss(pred.flatten(), data.spectrum)
        self.log('test_MSE', loss, batch_size=data.num_graphs)
        self.log('test_RSE', rse_loss, batch_size=data.num_graphs)

    def predict_step(self, test_batch):
        data = test_batch
        pred = self(data)

        data_size = data.spectrum.shape[0] // 200
        data.spectrum = data.spectrum.view(data_size, 200)

        return pred.detach().cpu().numpy(), data.spectrum.detach().cpu().numpy()
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(),
                         lr=self.lr,
                         betas=(0.9, 0.999), 
                         eps=1e-08, 
                         amsgrad=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer, 
                    mode='min',
                    factor=0.5,
                    patience=10,
                    min_lr=0.0000001
                ),
                "monitor": "val_loss",
                "frequency": 1
            }
        }

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

qm9_target_dict: Dict[int, str] = {
    0: 'dipole_moment',
    1: 'isotropic_polarizability',
    2: 'homo',
    3: 'lumo',
    4: 'gap',
    5: 'electronic_spatial_extent',
    6: 'zpve',
    7: 'energy_U0',
    8: 'energy_U',
    9: 'enthalpy_H',
    10: 'free_energy',
    11: 'heat_capacity',
}

class SchNet(L.LightningModule):
    r"""The continuous-filter convolutional neural network SchNet from the
    `"SchNet: A Continuous-filter Convolutional Neural Network for Modeling
    Quantum Interactions" <https://arxiv.org/abs/1706.08566>`_ paper that uses
    the interactions blocks of the form.

    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \odot
        h_{\mathbf{\Theta}} ( \exp(-\gamma(\mathbf{e}_{j,i} - \mathbf{\mu}))),

    here :math:`h_{\mathbf{\Theta}}` denotes an MLP and
    :math:`\mathbf{e}_{j,i}` denotes the interatomic distances between atoms.

    .. note::

        For an example of using a pretrained SchNet variant, see
        `examples/qm9_pretrained_schnet.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        qm9_pretrained_schnet.py>`_.

    Args:
        hidden_channels (int, optional): Hidden embedding size.
            (default: :obj:`128`)
        num_filters (int, optional): The number of filters to use.
            (default: :obj:`128`)
        num_interactions (int, optional): The number of interaction blocks.
            (default: :obj:`6`)
        num_gaussians (int, optional): The number of gaussians :math:`\mu`.
            (default: :obj:`50`)
        interaction_graph (callable, optional): The function used to compute
            the pairwise interaction graph and interatomic distances. If set to
            :obj:`None`, will construct a graph based on :obj:`cutoff` and
            :obj:`max_num_neighbors` properties.
            If provided, this method takes in :obj:`pos` and :obj:`batch`
            tensors and should return :obj:`(edge_index, edge_weight)` tensors.
            (default :obj:`None`)
        cutoff (float, optional): Cutoff distance for interatomic interactions.
            (default: :obj:`10.0`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            collect for each node within the :attr:`cutoff` distance.
            (default: :obj:`32`)
        readout (str, optional): Whether to apply :obj:`"add"` or :obj:`"mean"`
            global aggregation. (default: :obj:`"add"`)
        dipole (bool, optional): If set to :obj:`True`, will use the magnitude
            of the dipole moment to make the final prediction, *e.g.*, for
            target 0 of :class:`torch_geometric.datasets.QM9`.
            (default: :obj:`False`)
        mean (float, optional): The mean of the property to predict.
            (default: :obj:`None`)
        std (float, optional): The standard deviation of the property to
            predict. (default: :obj:`None`)
        atomref (torch.Tensor, optional): The reference of single-atom
            properties.
            Expects a vector of shape :obj:`(max_atomic_number, )`.
    """

    url = 'http://www.quantum-machine.org/datasets/trained_schnet_models.zip'

    def __init__(
        self,
        hidden_channels: int = 128,
        num_filters: int = 128,
        num_interactions: int = 6,
        num_gaussians: int = 50,
        cutoff: float = 10.0,
        a_emb: int = 20,
        interaction_graph: Optional[Callable] = None,
        max_num_neighbors: int = 32,
        readout: str = 'add',
        dipole: bool = False,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        atomref: OptTensor = None,
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        self.dipole = dipole
        self.sum_aggr = SumAggregation()
        self.readout = aggr_resolver('sum' if self.dipole else readout)
        self.mean = mean
        self.std = std
        self.scale = None
        self.a_emb = a_emb
        self._initialize_weights()
        self.save_hyperparameters()

        if self.dipole:
            import ase

            atomic_mass = torch.from_numpy(ase.data.atomic_masses)
            self.register_buffer('atomic_mass', atomic_mass)

        # Support z == 0 for padding atoms so that their embedding vectors
        # are zeroed and do not receive any gradients.
        self.embedding = Embedding(100, hidden_channels, padding_idx=0)

        if interaction_graph is not None:
            self.interaction_graph = interaction_graph
        else:
            self.interaction_graph = RadiusInteractionGraph(
                cutoff, max_num_neighbors)

        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels, num_gaussians,
                                     num_filters, cutoff)
            self.interactions.append(block)

        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.act = ShiftedSoftplus()
        self.lin2 = Linear(hidden_channels, 200)

        self.register_buffer('initial_atomref', atomref)
        self.atomref = None
        if atomref is not None:
            self.atomref = Embedding(a_emb, 1)
            self.atomref.weight.data.copy_(atomref)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.embedding.reset_parameters()
        for interaction in self.interactions:
            interaction.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)
        if self.atomref is not None:
            self.atomref.weight.data.copy_(self.initial_atomref)

    @staticmethod
    def from_qm9_pretrained(
        root: str,
        dataset: Dataset,
        target: int,
    ) -> Tuple['SchNet', Dataset, Dataset, Dataset]:  # pragma: no cover
        r"""Returns a pre-trained :class:`SchNet` model on the
        :class:`~torch_geometric.datasets.QM9` dataset, trained on the
        specified target :obj:`target`.
        """
        import ase
        import schnetpack as spk  # noqa

        assert target >= 0 and target <= 12
        is_dipole = target == 0

        units = [1] * 12
        units[0] = ase.units.Debye
        units[1] = ase.units.Bohr**3
        units[5] = ase.units.Bohr**2

        root = osp.expanduser(osp.normpath(root))
        os.makedirs(root, exist_ok=True)
        folder = 'trained_schnet_models'
        if not osp.exists(osp.join(root, folder)):
            path = download_url(SchNet.url, root)
            extract_zip(path, root)
            os.unlink(path)

        name = f'qm9_{qm9_target_dict[target]}'
        path = osp.join(root, 'trained_schnet_models', name, 'split.npz')

        split = np.load(path)
        train_idx = split['train_idx']
        val_idx = split['val_idx']
        test_idx = split['test_idx']

        # Filter the splits to only contain characterized molecules.
        idx = dataset.data.idx
        assoc = idx.new_empty(idx.max().item() + 1)
        assoc[idx] = torch.arange(idx.size(0))

        train_idx = assoc[train_idx[np.isin(train_idx, idx)]]
        val_idx = assoc[val_idx[np.isin(val_idx, idx)]]
        test_idx = assoc[test_idx[np.isin(test_idx, idx)]]

        path = osp.join(root, 'trained_schnet_models', name, 'best_model')

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            state = fs.torch_load(path, map_location='cpu')

        net = SchNet(
            hidden_channels=128,
            num_filters=128,
            num_interactions=6,
            num_gaussians=50,
            cutoff=10.0,
            dipole=is_dipole,
            atomref=dataset.atomref(target),
        )

        net.embedding.weight = state.representation.embedding.weight

        for int1, int2 in zip(state.representation.interactions,
                              net.interactions):
            int2.mlp[0].weight = int1.filter_network[0].weight
            int2.mlp[0].bias = int1.filter_network[0].bias
            int2.mlp[2].weight = int1.filter_network[1].weight
            int2.mlp[2].bias = int1.filter_network[1].bias
            int2.lin.weight = int1.dense.weight
            int2.lin.bias = int1.dense.bias

            int2.conv.lin1.weight = int1.cfconv.in2f.weight
            int2.conv.lin2.weight = int1.cfconv.f2out.weight
            int2.conv.lin2.bias = int1.cfconv.f2out.bias

        net.lin1.weight = state.output_modules[0].out_net[1].out_net[0].weight
        net.lin1.bias = state.output_modules[0].out_net[1].out_net[0].bias
        net.lin2.weight = state.output_modules[0].out_net[1].out_net[1].weight
        net.lin2.bias = state.output_modules[0].out_net[1].out_net[1].bias

        mean = state.output_modules[0].atom_pool.average
        net.readout = aggr_resolver('mean' if mean is True else 'add')

        dipole = state.output_modules[0].__class__.__name__ == 'DipoleMoment'
        net.dipole = dipole

        net.mean = state.output_modules[0].standardize.mean.item()
        net.std = state.output_modules[0].standardize.stddev.item()

        if state.output_modules[0].atomref is not None:
            net.atomref.weight = state.output_modules[0].atomref.weight
        else:
            net.atomref = None

        net.scale = 1.0 / units[target]

        return net, (dataset[train_idx], dataset[val_idx], dataset[test_idx])

    def forward(self, z: Tensor, pos: Tensor,
                batch: OptTensor = None) -> Tensor:
        r"""Forward pass.

        Args:
            z (torch.Tensor): Atomic number of each atom with shape
                :obj:`[num_atoms]`.
            pos (torch.Tensor): Coordinates of each atom with shape
                :obj:`[num_atoms, 3]`.
            batch (torch.Tensor, optional): Batch indices assigning each atom
                to a separate molecule with shape :obj:`[num_atoms]`.
                (default: :obj:`None`)
        """
        batch = torch.zeros_like(z) if batch is None else batch

        h = self.embedding(z)
        edge_index, edge_weight = self.interaction_graph(pos, batch)
        edge_attr = self.distance_expansion(edge_weight)

        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)

        if self.dipole:
            # Get center of mass.
            mass = self.atomic_mass[z].view(-1, 1)
            M = self.sum_aggr(mass, batch, dim=0)
            c = self.sum_aggr(mass * pos, batch, dim=0) / M
            h = h * (pos - c.index_select(0, batch))

        if not self.dipole and self.mean is not None and self.std is not None:
            h = h * self.std + self.mean

        if not self.dipole and self.atomref is not None:
            h = h + self.atomref(z)

        out = self.readout(h, batch, dim=0)

        if self.dipole:
            out = torch.norm(out, dim=-1, keepdim=True)

        if self.scale is not None:
            out = self.scale * out

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'hidden_channels={self.hidden_channels}, '
                f'num_filters={self.num_filters}, '
                f'num_interactions={self.num_interactions}, '
                f'num_gaussians={self.num_gaussians}, '
                f'cutoff={self.cutoff})')
    
    def training_step(self, train_batch, batch_idx):
        data = train_batch
        pred = self(data.z, data.pos)
        data_size = data.spectrum.shape[0] // 200
        data.spectrum = data.spectrum.view(data_size, 200)
        loss = nn.MSELoss()(pred, data.spectrum)
        self.log('train_loss', loss, on_epoch=True, batch_size=data.num_graphs)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        data = val_batch
        pred = self(data.z, data.pos)
        data_size = data.spectrum.shape[0] // 200
        data.spectrum = data.spectrum.view(data_size, 200)
        loss = nn.MSELoss()(pred, data.spectrum)
        self.log('val_loss', loss, on_epoch=True, batch_size=data.num_graphs)
        return loss
    
    def test_step(self, test_batch, batch_idx):
        data = test_batch
        pred = self(data.z, data.pos)
        data_size = data.spectrum.shape[0] // 200
        data.spectrum = data.spectrum.view(data_size, 200)
        loss = F.mse_loss(pred, data.spectrum)
        rse_loss = RSE_loss(pred, data.spectrum)
        self.log('test_MSE', loss, batch_size=data.num_graphs)
        self.log('test_RSE', rse_loss, batch_size=data.num_graphs)

    def predict_step(self, test_batch):
        data = test_batch
        pred = self(data.z, data.pos)

        data_size = data.spectrum.shape[0] // 200
        data.spectrum = data.spectrum.view(data_size, 200)

        return pred.detach().cpu().numpy(), data.spectrum.detach().cpu().numpy()
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(),
                         lr=0.001,
                         betas=(0.9, 0.999), 
                         eps=1e-08, 
                         amsgrad=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer, 
                    mode='min',
                    factor=0.5,
                    patience=10,
                    min_lr=0.0000001
                ),
                "monitor": "val_loss",
                "frequency": 1
            }
        }
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)