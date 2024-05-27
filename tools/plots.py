import pickle
from math import exp
import matplotlib.pyplot as plt
from rpy2.robjects import pandas2ri

#https://clauswilke.com/dataviz/color-pitfalls.html
default_color_scheme = ['#E69F00', 
                        '#56B4E9', 
                        '#009E73', 
                        "#F0E442", 
                        "#0072B2", 
                        "#D55E00", 
                        "#CC79A7", 
                        "#000000"]

def samples_to_s(sample_rate, samples):
    ''' returns time * sample_rate'''
    return samples / sample_rate

def s_to_samples(sample_rate, time):
    ''' returns sample_rate * time'''
    return sample_rate * time

def plot_conditional_effects(r_link, 
                             filename, 
                             rdf, 
                             plot_points:bool=False, 
                             sec_ax:tuple=None,
                             colors:list=None,
                             unlog:bool=False):
    ''' 
        r_link: an instance of the r_link class
        filename: str or Path
        rdf: cross-effects plot data from brms conditional_effects
        plot_points: If true, plots sample values
        sec_ax: for adding a secondary axis to the plots. Format is (label, (forward_conversion, invers_conversion))
        colors: colour scheme for the plots. If none, the default colour scheme from the Plots class will be used.
        unlog: converts log link to linear
    '''
    context = r_link.context()
    # not generalisable
    effect_names = r_link.base.attr(rdf, 'effects')

    with context():
        df = pandas2ri.rpy2py(rdf)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    if colors is None:
        colors = default_color_scheme

    colour_map = {}
    for i, v in enumerate(df[effect_names[1]].unique()):
        colour_map[v] = colors[i]

    if plot_points:
        points = r_link.base.attr(rdf, 'points')
        with context():
            point_df = pandas2ri.rpy2py(points)

        for var_id, group in point_df.groupby(effect_names[1]):
            x = group[effect_names[0]] #-  int(var_id)
            y = group['resp__']
            if unlog:
                x = x.apply(lambda x: exp(x))

            ax.scatter(x, y, linestyle="None", marker='.', color=colour_map[var_id])

    xlabel = df.columns[0]
    ylabel = df.columns[2]
    legend_title = df.columns[1]

    for var_id, group in df.groupby(effect_names[1]):
        x = group[effect_names[0]]
        var_line = group["estimate__"]
        upperlimit = group["upper__"]
        lowerlimit = group["lower__"]
        if unlog:
            x = x.apply(lambda x: exp(x))

        color = colour_map[var_id]
        ax.plot(x, var_line, color=color, alpha=0.8, label=str(var_id))
        ax.fill_between(x, upperlimit, lowerlimit, color=color, alpha=0.2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if sec_ax:
            secax = ax.secondary_xaxis('top', functions=(sec_ax[1][0], sec_ax[1][1]))
            secax.set_xlabel(sec_ax[0])

    lgd = ax.legend(loc='center left', title=legend_title, bbox_to_anchor=(1,0.5))
    save_plt_fig(fig, filename, ext='png', bbox_extra_artists=(lgd,), tight=True)

    return fig

def save_plt_fig(fig, filename, bbox_extra_artists=None, ext="eps", tight=False):
    '''Save a plot figure to file with timestamp.

       fig: figure to save
       filename: output path
       bbox_extra_artists: additional plot elements for formatting
       ext: file type to save
       tight: tight layout

    '''
    pickle_path = f"{filename}.pkl"
    img_path = f"{filename}.{ext}"
    if bbox_extra_artists is not None and not tight:
        fig.savefig(img_path, bbox_extra_artists=bbox_extra_artists, bbox_inches='tight')
    elif tight:
        fig.savefig(img_path, format=ext, bbox_inches='tight')
    else:
        fig.savefig(img_path, format=ext)

    with open(pickle_path, 'wb') as f:
        pickle.dump(fig, f)

    plt.close(fig)