import numpy as np
import torch
from bokeh.plotting import figure
from bokeh.models import Span, ContinuousTicker, LinearAxis
from bokeh.palettes import HighContrast3

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
    p.xaxis.axis_label_text_font_style = 'normal'
    p.xaxis.major_tick_in = 0
    p.xaxis.major_tick_out = 10
    p.xaxis.minor_tick_out = 6
    p.xaxis.major_tick_line_width = 2
    p.xaxis.minor_tick_line_width = 2
    p.xaxis.major_tick_line_color = 'black'
    p.xaxis.minor_tick_line_color = 'black'
    # y-axis settings
    p.yaxis.axis_label_text_font_size = '24px'
    p.yaxis.axis_label_text_font_style = 'normal'
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

def bokeh_hist(series, hist, edges, spacing):
    p = figure(
        x_axis_label = 'RSE value', y_axis_label = 'Frequency',
        x_range = (edges[0], edges[-1]), y_range = (0, max(hist)+spacing),
        width = 500, height = 450,
        outline_line_color = 'black', outline_line_width = 2
    )

    p.toolbar.logo = None
    p.toolbar_location = None
    p.min_border = 25

    # x-axis settings
    p.xaxis.ticker.desired_num_ticks = 3
    p.xaxis.axis_label_text_font_size = '24px'
    p.xaxis.major_label_text_font_size = '24px'
    p.xaxis.axis_label_text_font_style = 'normal'
    p.xaxis.major_tick_in = 0
    p.xaxis.major_tick_out = 10
    p.xaxis.minor_tick_out = 6
    p.xaxis.major_tick_line_width = 2
    p.xaxis.minor_tick_line_width = 2
    p.xaxis.major_tick_line_color = 'black'
    p.xaxis.minor_tick_line_color = 'black'
    p.xaxis[0].ticker.desired_num_ticks = 4
    # y-axis settings
    p.yaxis.axis_label_text_font_size = '24px'
    p.yaxis.major_label_text_font_size = '24px'
    p.yaxis.axis_label_text_font_style = 'normal'
    p.yaxis.major_tick_in = 0
    p.yaxis.major_tick_out = 10
    p.yaxis.major_tick_line_width = 2
    p.yaxis.major_tick_line_color = 'black'
    p.yaxis.minor_tick_line_color = None
    p.yaxis.major_label_text_color = 'black'
    # grid settings
    p.grid.grid_line_color = 'grey'
    p.grid.grid_line_alpha = 0.3
    p.grid.grid_line_width = 1.5
    p.grid.grid_line_dash = "dashed"

    # Plot data
    # Add histogram
    p.quad(bottom=0, top=hist, left=edges[:-1], right=edges[1:],
           fill_color='skyblue', line_color='black')    
    # Add average line
    vline = Span(location=series.mean(), dimension='height', line_color='dodgerblue', line_width=3, line_dash='dashed')
    p.renderers.extend([vline])

    p.output_backend = 'svg'

    return(p)