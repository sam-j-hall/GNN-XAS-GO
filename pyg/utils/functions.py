import numpy as np
import torch
import torch.nn as nn
from bokeh.plotting import figure
from bokeh.models import SingleIntervalTicker, LinearAxis, NumeralTickFormatter, Span
from bokeh.palettes import HighContrast3

def train_model(model, loader, optimizer, device):
    model.train()

    total_loss = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        pred = model(batch)

        data_size = batch.spectrum.shape[0] // 200
        batch.spectrum = batch.spectrum.view(data_size, 200)


        loss = nn.MSELoss()(pred, batch.spectrum)

        loss.backward()

        total_loss += loss.item()

        optimizer.step()

        return total_loss
    
def val_model(model, loader, device):
    model.eval()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)

        data_size = batch.spectrum.shape[0] // 200
        batch.spectrum = batch.spectrum.view(data_size, 200)

        loss = nn.MSELoss()(pred, batch.spectrum)

        total_loss += loss.item()

    return total_loss
        

def count_funct_groups(data):
    '''
    '''

    # Empty list to store numbers
    numbers = []
    oh = 0
    cooh = 0
    keto = 0
    cho = 0
    epoxy = 0
    for x in data:
        numbers = []
        # Extract numbers from mol name string
        for char in x.mol:
            if char.isdigit():
                numbers.append(int(char))

        oh += numbers[0]
        cooh += numbers[1]
        epoxy += numbers[2]
        cho += numbers[3]
        keto += numbers[4]
        oh += numbers[5]
        epoxy += numbers[6]
        epoxy += numbers[7]

    tot_groups = oh + cooh + epoxy + cho + keto

    oh_perc = oh / tot_groups
    cooh_perc = cooh / tot_groups
    epoxy_perc = epoxy / tot_groups
    cho_perc = cho / tot_groups
    keto_perc = keto / tot_groups
    
    return oh_perc*100, cooh_perc*100, epoxy_perc*100, cho_perc*100, keto_perc*100   

def RSE_loss(prediction, target):
    '''
    '''

    dE = (300 - 280) / 200
    nom = torch.sum(dE*torch.pow((target-prediction), 2))
    denom = torch.sum(dE*target)
    return torch.sqrt(nom) / denom

def get_spec_prediction(model, data, device):
    model.to(device)
    model.eval()

    with torch.no_grad():
        pred = model(data)

    pred_spectrum = pred.flatten().detach().numpy()
    true_spectrum = data.spectrum.detach().numpy()

    return pred_spectrum, true_spectrum

def calculate_rse(target, prediction):
    del_E = (300 - 280) / len(prediction)
    numerator = np.sum(del_E * np.power((target - prediction),2))
    denominator = np.sum(del_E * target)
    return np.sqrt(numerator) / denominator

def bokeh_spectra(pred_spectra, true_spectra):
    p = figure(
    x_axis_label = 'Photon Energy (eV)', y_axis_label = 'arb. units',
    x_range = (280,300),
    width = 350, height = 350,
    outline_line_color = 'black', outline_line_width = 2
    )

    p.toolbar.logo = None
    p.toolbar_location = None
    p.min_border = 25

    # x-axis settings
    p.xaxis.ticker.desired_num_ticks = 3
    p.xaxis.axis_label_text_font_size = '24px'
    p.xaxis.major_label_text_font_size = '24px'
    p.xaxis.major_tick_in = 0
    p.xaxis.major_tick_out = 10
    p.xaxis.minor_tick_out = 6
    p.xaxis.major_tick_line_width = 2
    p.xaxis.minor_tick_line_width = 2
    p.xaxis.major_tick_line_color = 'black'
    p.xaxis.minor_tick_line_color = 'black'
    # y-axis settings
    p.yaxis.axis_label_text_font_size = '24px'
    p.yaxis.major_tick_line_color = None
    p.yaxis.minor_tick_line_color = None
    p.yaxis.major_label_text_color = None
    # grid settings
    p.grid.grid_line_color = 'grey'
    p.grid.grid_line_alpha = 0.3
    p.grid.grid_line_width = 1.5
    p.grid.grid_line_dash = "dashed"

    # plot data
    x = np.linspace(280,300,200)
    p.line(x, true_spectra, line_width=3, line_color=HighContrast3[0], legend_label='True')
    p.line(x, pred_spectra, line_width=3, line_color=HighContrast3[1], legend_label='ML Model')

    # legend settings
    p.legend.location = 'bottom_right'
    p.legend.label_text_font_size = '20px'

    p.output_backend = 'svg'

    return p

def bokeh_hist(dataframe, average):
    p = figure(
        x_axis_label = 'RSE value', y_axis_label = 'Frequency',
        # x_range = (edges[0], edges[-1]), y_range = (0, max(hist)+spacing),
        width = 500, height = 450,
        outline_line_color = 'black', outline_line_width = 2
    )

    p.toolbar.logo = None
    p.toolbar_location = None
    p.min_border = 25

    # --- x-axis settings
    p.xaxis.ticker.desired_num_ticks = 3
    p.xaxis.axis_label_text_font_size = '24px'
    p.xaxis.major_label_text_font_size = '24px'
    p.xaxis.major_tick_in = 0
    p.xaxis.major_tick_out = 10
    p.xaxis.minor_tick_out = 6
    p.xaxis.major_tick_line_width = 2
    p.xaxis.minor_tick_line_width = 2
    p.xaxis.major_tick_line_color = 'black'
    p.xaxis.minor_tick_line_color = 'black'
    p.xaxis[0].ticker.desired_num_ticks = 4
    # --- y-axis settings
    p.yaxis.axis_label_text_font_size = '24px'
    p.yaxis.major_label_text_font_size = '24px'
    p.yaxis.major_tick_in = 0
    p.yaxis.major_tick_out = 10
    p.yaxis.major_tick_line_width = 2
    p.yaxis.major_tick_line_color = 'black'
    p.yaxis.minor_tick_line_color = None
    p.yaxis.major_label_text_color = 'black'
    # --- grid settings
    p.grid.grid_line_color = 'grey'
    p.grid.grid_line_alpha = 0.3
    p.grid.grid_line_width = 1.5
    p.grid.grid_line_dash = "dashed"

    # --- Format x-axis
    ticker = SingleIntervalTicker(interval=20)
    xaxis = LinearAxis(ticker=ticker)
    p.add_layout(xaxis, 'below')

    # --- Plot data
    # --- Add histogram
    p.quad(bottom=0, top=dataframe['rse_value'], left=dataframe['left'], right=dataframe['right'],
           fill_color='skyblue', line_color='black')
    # --- Add average line
    vline = Span(location=average, dimension='height', line_color='black', line_width=3, line_dash='dashed')
    p.renderers.extend([vline])

    p.output_backend = 'svg'

    return(p)