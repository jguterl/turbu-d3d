import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import PySimpleGUI as sg
from screeninfo import get_monitors
from get_DIIID_data import *
from plot_tools import *
from TS_data_fit import *
from find_plateaus_2 import *
from save_data import *
from scan_study import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

def draw_figure_w_toolbar(canvas, fig, canvas_toolbar):
    if canvas.children:
        for child in canvas.winfo_children():
            child.destroy()
    if canvas_toolbar.children:
        for child in canvas_toolbar.winfo_children():
            child.destroy()
    figure_canvas_agg = FigureCanvasTkAgg(fig, master=canvas)
    figure_canvas_agg.draw()
    toolbar = Toolbar(figure_canvas_agg, canvas_toolbar)
    toolbar.update()
    figure_canvas_agg.get_tk_widget().pack(side='right', expand=1)


class Toolbar(NavigationToolbar2Tk):
    # only display the buttons we need
    # toolitems = [t for t in NavigationToolbar2Tk.toolitems if
    #              t[0] in ('Home', 'Pan', 'Zoom')]
                # t[0] in ('Home', 'Pan', 'Zoom','Save')]
    toolitems = [t for t in NavigationToolbar2Tk.toolitems]
                # t[0] in ('Home', 'Pan', 'Zoom','Save')]
    def __init__(self, *args, **kwargs):
        super(Toolbar, self).__init__(*args, **kwargs)

win_height = get_monitors()[0].height
win_width = get_monitors()[0].width


def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

def delete_fig_agg(fig_agg):
    fig_agg.get_tk_widget().forget()
    plt.close('all')
    
def init_fig_maker():
    # plt.clf()
    # plt.close()
    fig = matplotlib.figure.Figure()
    dpi = fig.get_dpi()
    # ax = fig.add_subplot(111)
    # ax.plot([],[],'bo')
    # ax.grid(color='k', linestyle='-')       
    return fig

def fig_maker(database):
    # plt.clf()
    # plt.close()
    fig = matplotlib.figure.Figure()
    dpi = fig.get_dpi()
    try:
        ax = fig.add_subplot(231)
        ax.plot(database['BT']['time'],database['BT']['data'],'k', label = 'BT [T]')
        ax.plot(database['Ip']['time'],database['Ip']['data']*1e-6,'r', label = 'Ip [MA]')
        ax.plot(database['nl']['time'],database['nl']['data']*1e-13,'b', label = r'$n_{lid} [\times 10^{19}m^{-3}]$')    
        ax.plot(database['q95']['time'],database['q95']['data'],'g', label = r'$q_{95} [w/o units]$')    
        ax.plot(database['puff']['time'],database['puff']['data']*1e-1,'m', label = r'$Gas\, puff\, rate [\times 10 Torr.L/s]$')     
        ax.legend()
        ax.grid(color='k', linestyle='-')       
        ax.set_xlabel('time [ms]')
        ax.set_xlim((-100,7000))
    except:
        []
    try:
        ax1 = fig.add_subplot(232)
        ax1.plot(database['kappa']['time'],database['kappa']['data'],'k', label = r'$\kappa$')
        ax1.plot(database['tri_up']['time'],database['tri_up']['data'],'r', label = r'$\delta_{up}$')
        ax1.plot(database['tri_bot']['time'],database['tri_bot']['data'],'b', label = r'$\delta_{bot}$')
        ax1.plot(database['tri']['time'],database['tri']['data'],'g', label = r'$\delta$')
        ax1.plot(database['R0']['time'],database['R0']['data'],'m', label = r'$R_{0} [m]$')    
        ax1.plot(database['a']['time'],database['a']['data'],'c', label = r'$a [m]$')    
        ax1.legend()
        ax1.grid(color='k', linestyle='-')       
        ax1.set_xlabel('time [ms]')
        ax1.set_xlim((-100,7000))
    except:
        []
    
    try:
        ax2 = fig.add_subplot(233)
        ax2.plot(database['R_OSP']['time'],database['R_OSP']['data'],'k', label = r'$R_{OSP} [m]$')
        ax2.plot(database['Z_OSP']['time'],database['Z_OSP']['data'],'r', label = r'$Z_{OSP} [m]$')
        ax2.legend()
        ax2.grid(color='k', linestyle='-')       
        ax2.set_xlabel('time [ms]')
        ax2.set_xlim((-100,7000))
    except:
        []
    try:
        ax3 = fig.add_subplot(234)
        ax3.plot(database['Pohm']['time'],database['Pohm']['data']*1e-6,'k', label = r'$P_{ohm} [MW]$')
        ax3.plot(database['PECH']['time'],database['PECH']['data'],'r', label = r'$P_{ECH} [MW]$')
        ax3.plot(database['PICH']['time'],database['PICH']['data'],'b', label = r'$P_{ICH} [MW]$')
        ax3.plot(database['PNBI']['time'],database['PNBI']['data']*1e-3,'g', label = r'$P_{NBI} [MW]$')
        ax3.plot(database['Prad']['time'],database['Prad']['data']*1e3,'m', label = r'$P_{rad} [MW]$')
        ax3.plot(database['Prad_div']['time'],database['Prad_div']['data']*1e-6,'c', label = r'$P_{rad, div} [MW]$')
        ax3.legend()
        ax3.grid(color='k', linestyle='-')       
        ax3.set_xlabel('time [ms]')
        ax3.set_xlim((-100,7000))
    except:
        []
    try:
        ax4 = fig.add_subplot(235)
        ax4.plot(database['H98']['time'],database['H98']['data'],'k', label = 'H98 [w/o units]')
        ax4.plot(database['tauE']['time'],database['tauE']['data'],'r', label = r'$\tau_{E} [s]$')
        # ax4.plot(database['tauMHD']['time'],database['tauMHD']['data'],'b', label = r'$\tau_{MHD} [s]$')
        ax4.plot(database['tauth']['time'],database['tauth']['data'],'g', label = r'$\tau_{th} [s]$')
        ax4.legend()
        ax4.grid(color='k', linestyle='-')       
        ax4.set_xlabel('time [ms]')
        ax4.set_xlim((-100,7000))
    except:
        []
    try:
        
        ax5 = fig.add_subplot(236)
        ax5.plot([0],[0],'k', label = 'Need to add TeT')
        ax5.legend()
        ax5.grid(color='k', linestyle='-')       
        ax5.set_xlabel('time [ms]')
        ax5.set_xlim((-100,7000))
    except:
        []
    
    
    return fig

def open_window():
    layout = [[sg.Text("New Window", key="new")]]
    window = sg.Window("Second Window", layout, modal=True)
    choice = None
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        
    window.close()

matplotlib.use('TkAgg')
w, h = size = (np.floor(0.8*win_width), np.floor(0.7*win_height))     # figure size

fig = init_fig_maker()#matplotlib.figure.Figure(figsize=figsize)
# dpi = fig.get_dpi()
      # canvas size
# t = np.arange(0, 3, .01)
# fig.add_subplot(111).plot(t, 2 * np.sin(2 * np.pi * t))


layout = [ [sg.Canvas(size=size, key='-CANVAS-')],
          [sg.Canvas(key='controls_cv')],
          [sg.Text('Enter shot number'), sg.InputText(), sg.OK()],
          [sg.Cancel(), sg.Button('Get data'), sg.Button('Find plateaus'), sg.Button('Scan study'), sg.Button('Get Thomson profiles'), sg.Button('Save data')]]

window = sg.Window('OMFAT', layout, finalize=True, element_justification='center', font='Helvetica 14')
fig_agg = draw_figure(window['-CANVAS-'].TKCanvas, fig)

shotnumber = []
data = []
ts_data =[]
t0 =[]
t1 =[]
data_fit = []
scan = ''
while True:             
    event, values = window.read()
    if event in (sg.WIN_CLOSED, 'Cancel'):
        break
    elif event in ('OK'):
        try:
            shotnumber = float(values[0])
        except:
            print('Not a shot number!')
            shotnumber = []
    elif event in ('Get data'):
        t0 =[]
        t1 = []
        data = []
        ts_data = []
        data_fit = []
        if shotnumber==[]:
            print('Need a shot number')
        else:
            print('data fetching for shot number:'+str(int(shotnumber)))
            data = fetch_data(int(shotnumber))
            print('data fetched')
            # delete_fig_agg(fig_agg)
            fig = fig_maker(data)
            fig_agg = draw_figure_w_toolbar(window['-CANVAS-'].TKCanvas, fig, window['controls_cv'].TKCanvas)
            plt.close()
            window['-CANVAS-'].set_size((size))
    elif event in ('Find plateaus'):
        t0 = []
        t1 = []
        ts_data = []
        scan = ''
        if data==[]:
            print('No data fetched ---> No plateau to find')
        else:
            t0, t1 = plateau_bounds(data)
            print('Plateau boundaries found')
    elif event in ('Scan study'):
        t0 = []
        t1 = []
        ts_data = []
        if data==[]:
            print('No data fetched ---> No steps to find')
        else:
            t0, t1 = scan_bounds(data)
            scan = '_scan'
            print('Step boundaries found')           
            
            
    elif event in ('Get Thomson profiles'):
        ts_data = []
        data_fit = []
        if t0==[]:
            print('No plateau found -----> no Thomson data to get.')
        else:
            ts_data, eq = get_TS_data(shotnumber, t0, t1)
            print('Thomson data uploaded')
            data_fit = study_TS_data(ts_data, eq)
    elif event in ('Save data'):
        if data==[]:
            print('No data fetched ---> No plateau to find')
        else:
            if t0==[]:
                print('No plateau found -----> no Thomson data to get.')
            else:
                if ts_data==[]:
                    print('No Thomson data:')
                else:
                    if data_fit == []:
                        print('No fit applied')
                    else:
                        save_data_plateau(t0, t1, shotnumber, ts_data, eq, data_fit, data, scan)

            
            
    
            # fig_canvas_agg = draw_figure(window['-CANVAS-'].TKCanvas, fig)
window.close()