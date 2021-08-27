
import PySimpleGUI as sg
import pandas as pd
from datetime import datetime
from scipy import stats
from statannot import add_stat_annotation
import seaborn
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem
from mlxtend.evaluate import paired_ttest_5x2cv
from statsmodels.sandbox.stats.runs import mcnemar


def plot_growth_curve(data, figure_name, x_axis_label: str = "Time",
                      y_axis_label: str = "Y Axis", plot_title: str = "Growth Curve", with_std_mean: bool = False,
                      just_std_mean: bool = False):
    plt.figure(1, figsize=(9, 6))
    if not just_std_mean:
        plt.plot(data['time'], data.loc[:, data.columns != 'time'], label=data.loc[:, data.columns != 'time'].columns)
        plt.legend()

    # plot std/mean if desired
    if with_std_mean or just_std_mean:
        mean = data.loc[:, data.columns != 'time'].mean(axis=1)
        std = data.loc[:, data.columns != 'time'].std(axis=1)

        plt.errorbar(data['time'], mean,
                     yerr=std,
                     fmt='-o',
                     capsize=5,
                     label='Errorbar')

    plt.title(plot_title)
    plt.xlabel(x_axis_label) if x_axis_label else ''
    plt.ylabel(y_axis_label) if y_axis_label else ''
    plt.legend()
    plt.grid()
    plt.savefig(figure_name, dpi=400)
    plt.show()
    plt.close()


def plot_box_swarm(data, figure_name, x_axis_label: str = "",
                   y_axis_label: str = "", plot_title: str = "Boxplot", with_ttest: bool = False):
    plt.figure(1, figsize=(9, 6))

    # add title to plot
    plt.title(plot_title)

    # plot data on swarmplot and boxplot
    seaborn.swarmplot(data=data, color=".25")
    ax = seaborn.boxplot(data=data)

    # y-axis label
    ax.set(ylabel=y_axis_label) if y_axis_label else ''
    ax.set(xlabel=x_axis_label) if x_axis_label else ''

    if with_ttest:
        add_stat_annotation(ax, data=data,
                            box_pairs=get_ttest_variables(data),
                            test='t-test_ind', text_format='star',
                            loc='inside', verbose=1, comparisons_correction=None)[0]

    ax.grid()

    # write figure file with quality 400 dpi
    plt.savefig(figure_name, dpi=400)
    plt.show()
    plt.close()


def get_ttest_variables(data):
    all_cols = data.keys()

    combine = [(cola, colb) for cola in all_cols for colb in all_cols if cola != colb]

    for tuple in combine:
        if tuple[0] == tuple[1]:
            combine.remove(tuple)
        if (tuple[1], tuple[0]) in combine:
            combine.remove(tuple)

    return combine


def build_plottable_df(data):
    newdata = pd.DataFrame(columns=['Primer', 'NHDF DMSO', 'NHDF 1uM Elami', 'NHDF 10uM Elami',
                                    'PZ DMSO', 'PZ 1uM Elami', 'PZ 10uM Elami'])
    stddata = pd.DataFrame(columns=['Primer', 'NHDF DMSO', 'NHDF 1uM Elami', 'NHDF 10uM Elami',
                                    'PZ DMSO', 'PZ 1uM Elami', 'PZ 10uM Elami'])

    i = 0

    for primer in list(set(data['Primer'])):
        temp = data[data['Primer'] == primer]
        for cell in newdata.columns:
            if cell == 'Primer':
                newdata.loc[primer] = primer
                stddata.loc[primer] = primer
                continue

            newdata.iloc[i][cell] = temp[temp['Treatment'] == cell]['RQ value'].mean()
            stddata.iloc[i][cell] = temp[temp['Treatment'] == cell]['RQ value'].sem()

        i = i + 1

    newdata.reset_index(drop=True, inplace=True)
    stddata.reset_index(drop=True, inplace=True)

    return newdata, stddata


def plotti_barotti(data, title, xlabel, ylabel, figure_name):
    data, std = build_plottable_df(data)
    my_cmap = plt.get_cmap("viridis")

    x_label_bio = list(data.Primer)
    x_cats = list(data.columns)
    x_cats.remove('Primer')

    ind = np.arange(0, 12, 4)  # the x locations for the groups
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
    rects1 = ax.bar(ind - 2 * width - width / 2, data[x_cats[0]], width, yerr=std[x_cats[0]],
                    label=x_cats[0], color='#AB0AAB')
    rects2 = ax.bar(ind - width - width / 2, data[x_cats[1]], width, yerr=std[x_cats[1]],
                    label=x_cats[1], color='#F69AF6')
    rects3 = ax.bar(ind - width / 2, data[x_cats[2]], width, yerr=std[x_cats[2]],
                    label=x_cats[2], color='#F6CEF6')
    rects4 = ax.bar(ind + width / 2, data[x_cats[3]], width, yerr=std[x_cats[3]],
                    label=x_cats[3], color='#C7F9CE')
    rects5 = ax.bar(ind + width + width / 2, data[x_cats[4]], width, yerr=std[x_cats[4]],
                    label=x_cats[4], color='#78DC85')
    rects6 = ax.bar(ind + 2 * width + width / 2, data[x_cats[5]], width, yerr=std[x_cats[5]],
                    label=x_cats[5], color='#358740')

    # Add some text for labels, title and custom x-axis tick labels, etc.

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(ind)
    ax.set_xticklabels(x_label_bio)
    ax.legend()
    fig.savefig(figure_name + '.png')


sg.theme('DarkAmber')  # Add a touch of color
# All the stuff inside your window.
layout = [[sg.Text('--- No R supported here ---')],
          [sg.Text('Einmal Datensatz bitte:')],
          [sg.Input(), sg.FileBrowse()],
          [sg.Text()],
          [sg.Text('Welcher plot soll es sein?   ¯\_(ツ)_/¯')],
          [sg.Combo(['Boxplot', 'Boxplot with TTest', 'Growth Curve',
                     'Growth Curve mit Mean+Std', 'Growth Curve nur Mean+Std', 'Barchart mit std'])],
          # , 'Growth Curve'
          [sg.Text()],
          [sg.Text("X-Achsenbeschriftung:"), sg.InputText(),
           sg.Text("Y-Achsenbeschriftung:"), sg.InputText(),
           sg.Text("Titel:"), sg.InputText()],
          [sg.Button('Let\'s plot')],
          [sg.Text()],
          [sg.Text('T-Test nochmal als txt-Datei außerhalb des plots?')],
          [sg.Button('Ja bitte!'), sg.Button('Cancel')]
          ]

# Create the Window
window = sg.Window('Plottobot v.1.1', layout)
# Event Loop to process "events" and get the "values" of the inputs
to_plot = None
while True:
    event, values = window.read()

    if event == sg.WIN_CLOSED or event == 'Cancel':  # if user closes window or clicks cancel
        break

    if event == 'Let\'s plot':
        if values['Browse'] != '':
            to_plot = pd.read_excel(str(values['Browse']))

        if to_plot is not None and values[1] != '':
            x_axis_label = values[2] if values[2] else None
            y_axis_label = values[3] if values[3] else None
            title_label = values[4] if values[4] else None

            if values[1] == 'Boxplot':
                plot_box_swarm(data=to_plot,
                               y_axis_label=str(y_axis_label),
                               plot_title=str(title_label),
                               figure_name="Boxplot_Plottobot_" + datetime.now().strftime('%d%m%Y_%H-%M-%S') + '.png',
                               with_ttest=False)
            if values[1] == 'Boxplot with TTest':
                plot_box_swarm(data=to_plot,
                               y_axis_label=str(y_axis_label),
                               plot_title=str(title_label),
                               figure_name="Boxplot_with_ttest_Plottobot_" + datetime.now().strftime(
                                   '%d%m%Y_%H-%M-%S') + '.png',
                               with_ttest=True)
            if values[1] == 'Growth Curve':
                plot_growth_curve(data=to_plot,
                                  y_axis_label=str(y_axis_label),
                                  plot_title=str(title_label),
                                  figure_name="Growth_Curve_Plottobot_" + datetime.now().strftime(
                                      '%d%m%Y_%H-%M-%S') + '.png')
            if values[1] == 'Growth Curve mit Mean+Std':
                plot_growth_curve(data=to_plot,
                                  y_axis_label=str(y_axis_label),
                                  plot_title=str(title_label),
                                  figure_name="Growth_Curve_Plottobot_" + datetime.now().strftime(
                                      '%d%m%Y_%H-%M-%S') + '.png', with_std_mean=True)
            if values[1] == 'Growth Curve nur Mean+Std':
                plot_growth_curve(data=to_plot,
                                  y_axis_label=str(y_axis_label),
                                  plot_title=str(title_label),
                                  figure_name="Growth_Curve_Plottobot_" + datetime.now().strftime(
                                      '%d%m%Y_%H-%M-%S') + '.png', with_std_mean=True, just_std_mean=True)
            if values[1] == 'Barchart mit std':
                plotti_barotti(data=to_plot,
                               title=str(title_label),
                               xlabel=str(x_axis_label),
                               ylabel=str(y_axis_label),
                               figure_name="Barchart_mit_std_Plottobot_" + datetime.now().strftime(
                                   '%d%m%Y_%H-%M-%S'))

    if event == 'Ja bitte!':
        if values['Browse'] != '':
            to_plot = pd.read_excel(str(values['Browse']))
            to_plot = to_plot.dropna()

            txt_string = ""

            col_list = to_plot.columns.values.tolist()

            combined = get_ttest_variables(to_plot)
            for tuple in combined:
                txt_string = txt_string + "\nT-Test (" + tuple[0] + " & " + tuple[1] + "): " + \
                             str(mcnemar(to_plot[tuple[0]], to_plot[tuple[1]], exact=True))

            text_file = open("T_Test" + datetime.now().strftime('%d%m%Y_%H-%M-%S') + ".txt", "w")
            n = text_file.write(txt_string)
            text_file.close()

window.close()