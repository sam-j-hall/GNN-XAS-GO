import torch
import torch.nn as nn
from torch_geometric.data import Batch

def train_model(model, loader, optimizer, device):
    '''
    
    '''

    model.train()

    total_loss = 0
    num_graphs = 0

    for data in loader:
        data = data.to(device)

        optimizer.zero_grad()

        pred = model(data)
        
        # data_size = data.spectrum.shape[0] // 200
        # data.spectrum = data.spectrum.view(data_size, 200)

        loss = nn.MSELoss()(pred.flatten(), data.spectrum)

        total_loss += loss.item()
        num_graphs += data.num_graphs

        loss.backward()

        optimizer.step()

        # out = pred[0].detach().cpu().numpy()
        # true = data.spectrum[0].detach().cpu().numpy()

    return total_loss #/ num_graphs #, embedding#, true, out


def val_test(model, loader, device):
    '''
    
    '''

    model.eval()

    total_loss = 0
    num_graphs = 0

    for data in loader:
        data = data.to(device)

        pred = model(data)

        # data_size = data.spectrum.shape[0] // 200
        # data.spectrum = data.spectrum.view(data_size, 200)
        
        loss = nn.MSELoss()(pred.flatten(), data.spectrum)

        total_loss += loss.item()
        num_graphs += data.num_graphs

        # out = pred[0].detach().cpu().numpy()
        # true = data.spectrum[0].detach().cpu().numpy()

    return total_loss #/ num_graphs #, out, true

def train_schnet(model, train_loader, optimizer, device):

    model.train()

    total_loss = 0
    num_graphs = 0

    for batch in train_loader:
        batch = batch.to(device)

        optimizer.zero_grad()

        pred = model(batch.z, batch.pos, batch.batch)

        data_size = batch.spectrum.shape[0] // 200
        batch.spectrum = batch.spectrum.view(data_size, 200)

        loss = nn.MSELoss()(pred, batch.spectrum)
        print(loss)

        total_loss += loss.item() #/ batch.num_graphs
        num_graphs += batch.num_graphs

        loss.backward()

        optimizer.step()

    return total_loss #/ num_graphs

def val_schnet(model, val_loader, device):
    '''
    '''

    model.eval()
   
    total_loss = 0
    num_graphs = 0

    for batch in val_loader:
        batch = batch.to(device)

        pred = model(batch.z, batch.pos, batch.batch)

        loss = nn.MSELoss()(pred.flatten(), batch.spectrum)

        total_loss = loss.item()
        num_graphs = batch.num_graphs

    return total_loss #/ num_graphs


def get_spec_prediction(model, index, dataset, device):
    '''
    
    '''
    # --- Set the model to evaluation mode
    model.eval()

    # --- Get a single graph from the test dataset
    graph_index = index
    graph_data = dataset[graph_index].to(device)
    data = Batch.from_data_list([graph_data])

    # --- Pass the graph through the model
    with torch.no_grad():
        pred = model(data)

    # ---
    true_spectrum = graph_data.spectrum.cpu().numpy()
    predicted_spectrum = pred.cpu().numpy()
    predicted_spectrum = predicted_spectrum.reshape(-1)

    return predicted_spectrum, true_spectrum
