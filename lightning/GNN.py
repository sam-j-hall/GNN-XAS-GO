# --- PyTorch
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
# --- PyG
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn import GCNConv, GINConv, GINEConv, GATv2Conv, MLP, SAGEConv
# --- Lightning
import pytorch_lightning as pl

class GNN_model(pl.LightningModule):
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

        # --- Sanity check number of layers
        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        # --- Set up GNN layers
        # --- Select the messgage passing layer
        for i, in_c, out_c in zip(range(self.num_layers), self.in_channels, self.out_channels):
            self.convs.append(GCNConv(in_c, out_c))
            self.batch_norms.append(torch.nn.BatchNorm1d(out_c))
        # --- Select the pooling function
        self.pool = global_mean_pool
        # --- Final linear layer
        self.graph_pred_linear = torch.nn.Linear(self.out_channels[-1], self.num_tasks)      	


    def forward(self, batched_data):

        x = batched_data.x.float()
        edge_index = batched_data.edge_index
        edge_attr = batched_data.edge_attr.float()

        # --- Create list of features
        h_list = [x]

        # --- Pass through GNN layers
        for layer in range(self.num_layers):
            # --- Message passing layer
            h = self.convs[layer](h_list[layer], edge_index)
            # --- Batch normalization
            h = self.batch_norms[layer](h)
            # --- Dropout with/without relu
            if layer == self.num_layers - 1:
                # --- Remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            h_list.append(h)
        # --- Final node representation from GNN
        node_rep = h_list[-1]
        # --- Pool all atom node representation for grap rep
        graph_embedding = self.pool(node_rep, batched_data.batch)
        # --- Activation function
        p = torch.nn.LeakyReLU(0.1)
        # --- Pass through linear layer
        out = p(self.graph_pred_linear(graph_embedding))

        return out
    
    def training_step(self, train_batch, batch_idx):
        data = train_batch
        spec_hat = self(data)
        loss = F.mse_loss(spec_hat.view(-1, 1), data.spectrum.view(-1, 1))
        self.log('train_loss', loss, batch_size=len(data))
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        data = val_batch
        spec_hat = self(data)
        loss = F.mse_loss(spec_hat.view(-1, 1), data.spectrum.view(-1, 1))
        self.log('val_loss', loss, batch_size=len(data))
        return loss
    
    def test_step(self, test_batch, batch_idx):
        data = test_batch
        spec_hat = self(data)
        loss = F.mse_loss(spec_hat.view(-1, 1), data.spectrum.view(-1, 1))
        self.log('test_loss', loss)
    
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
                    patience=100,
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
