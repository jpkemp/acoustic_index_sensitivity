'''graph combination'''
import io
import pickle
from functools import partial
from itertools import product
from math import ceil
from pathlib import Path
import matplotlib.pyplot
from tqdm import tqdm
from PIL import Image
import matplotlib
import numpy as np

class RenamingUnpickler(pickle.Unpickler):
    def find_class(cls, module, name):
        if module == 'study_settings.carara':
            module = 'study_settings.santa_rosa'
        if name == "CararaSettings":
            name = 'SantaRosaSetting'
        elif name == 'CararaToolbox':
            name = 'SantaRosaToolbox'
        return super().find_class(module, name)


class PlotCombiner:
    '''graph combining functions'''
    time_period_conversion = {'0': 'Midnight', '12': 'Midday'}
    def __init__(self) -> None:
        self.fig_bufs = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for buf in self.fig_bufs:
            buf.close()


    @classmethod
    def fig_changer(cls, fig, filename):
        ax = fig.gca()
        if "Value" in ax.get_ylabel():
            filename = str(filename)
            if "ACI" in filename:
                replacement = "ACI"
            if "ADI" in filename:
                replacement = "ADI"
            if "AEI" in filename:
                replacement = "AEI"
            if "BIO" in filename:
                replacement = "BIO"
            ax.set_ylabel(replacement)
            t = ax.yaxis.get_offset_text()
            text = t.get_text()
            if text:
                t.set_visible(False)

        if ax.get_xlabel() == "Window":
            ax.set_xlabel("Window length (samples)")
        cls.format_axis_numbers(ax)
        fig.set_dpi(300)
        fig.canvas.draw()
        fig.canvas.flush_events()

    def open_figure(self, filename, discard_lgd=False):
        fig = RenamingUnpickler(open(filename, 'rb')).load()
        self.fig_changer(fig, filename)
        lgd = fig.get_axes()[0].get_legend()
        if lgd is None:
            buf = None
        elif discard_lgd:
            lgd.remove()
            buf = None
        else:
            buf = io.BytesIO()
            self.fig_bufs.append(buf)
            self.export_legend(lgd, buf)
            lgd.remove()

        return fig, buf

    @classmethod
    def export_legend(cls, legend, buf, fontsize=16):
        cls.update_lgd_font_size(legend, fontsize)
        ttl = legend.get_title()
        if ttl.get_text() == 'Hour':
            ttl.set_text('Time Period')
            for text in legend.texts:
                text.set_text(cls.time_period_conversion[text.get_text()])

        fig  = legend.figure
        fig.canvas.draw()
        bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(buf, bbox_inches=bbox, dpi=300)
        matplotlib.pyplot.close(fig)

    @classmethod
    def convert_plots_to_images(cls, plots):
        bufs = [io.BytesIO() for _ in range(len(plots))]
        images=[]
        lgds = []
        for i, (fig, lgd_buf) in enumerate(plots):
            buf = bufs[i]
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
            matplotlib.pyplot.close(fig)
            buf.seek(0)
            images.append(Image.open(buf))
            if lgd_buf:
                lgd = Image.open(lgd_buf)
                lgds.append(lgd)

        return images, bufs, lgds

    @classmethod
    def combine_images(cls, images, rows=2, imgs_per_row=None, spacing=20, lgd=None):
        if imgs_per_row is None: imgs_per_row = ceil(len(images) / rows)
        mod = len(images) % imgs_per_row
        final = []
        subset = images
        if mod:
            subset = images[:-mod]
            final = images[-mod:]

        widths, heights = zip(*(i.size for i in subset))
        lgd_width = 0
        if lgd:
            lgd_width += lgd.size[0] + spacing

        max_width = 0
        max_col_width = [0] * imgs_per_row
        max_height = 0
        for i in range(0, len(subset), imgs_per_row):
            col_widths = widths[i:i+imgs_per_row]
            row_width = sum(col_widths) + imgs_per_row * spacing
            for j in range(imgs_per_row):
                max_col_width[j] = max(max_col_width[j], col_widths[j])

            max_width = max(max_width, row_width)
            max_height += max(heights[i:i+imgs_per_row])


        total_width = max_width + lgd_width
        total_height = ceil(max(heights) * rows) + (rows - 1) * (int(spacing))

        new_im = Image.new('RGB', (total_width, total_height), color="white")

        x_offset = 0
        y_offset = 0
        for i, im in enumerate(subset):
            if i and rows > 1 and not i % imgs_per_row:
                y_offset += im.size[1] + spacing
                x_offset = 0

            adjusted_pos = int((max_col_width[i%imgs_per_row] - im.size[0]) / 2)
            new_im.paste(im, (x_offset + adjusted_pos,y_offset))
            x_offset += max(im.size[0], max_col_width[i % imgs_per_row])

        filler = ceil(im.size[0] * (imgs_per_row - len(final)) / imgs_per_row)
        x_offset = filler
        y_offset += im.size[1] + spacing
        for i, im in enumerate(final):
            new_im.paste(im, (x_offset,y_offset))
            x_offset += filler

        if lgd:
            x_offset = total_width - lgd_width - spacing
            y_offset = ceil((total_height / 2) - (lgd.size[1] / 2)) + 10
            new_im.paste(lgd, (x_offset,y_offset))

        return new_im

    @classmethod
    def combine_figures(cls, plots, rows=2, imgs_per_row=None, spacing=20, include_lgd=True):
        '''combine graphs into one image. assumes all images same size if number of images does not equal rows * imgs_per_row'''
        bufs = []
        try:
            images, bufs, lgd = cls.convert_plots_to_images(plots)
            lgd = lgd[-1] if lgd and include_lgd else None
            new_im = cls.combine_images(images, rows, imgs_per_row, spacing, lgd)
        finally:
            for buf in bufs:
                buf.close()

        return new_im

    @classmethod
    def get_figures_from_folders(cls, folder_path, only_include=None, sort_by=None):
        '''get graphs from analysis output folders'''
        filenames = [x for x in Path(folder_path).iterdir() if (x.is_file() and x.suffix == '.pkl')]
        if only_include is not None:
            if isinstance(only_include, list):
                filenames = [x for x in filenames if all(y in x.name for y in only_include)]
            else:
                filenames = [x for x in filenames if only_include(x.name)]

        if sort_by is not None:
            filenames.sort(key=sort_by)

        return filenames

    @classmethod
    def format_numbers_for_one_axis(cls, ax, pos, tg, ts, lg, ls):
        labels = tg()
        # ts(labels)
        max_exponent = max([int("{:.2e}".format(x).split('e')[1]) for x in labels])
        if max_exponent > 4:
            new_labels = [int(x) / 10**max_exponent for x in labels]
            ax.text(*pos, f"$x10^{max_exponent}$", transform=ax.transAxes, fontsize=16)
        elif max_exponent > 2:
            new_labels = [f"{int(x):,}" for x in labels]
        else:
            new_labels = lg()

        ls(new_labels)

    @classmethod
    def format_axis_numbers(cls, ax):
        cls.format_numbers_for_one_axis(ax,
                                        (1.02,-0.01),
                                        ax.get_xticks,
                                        ax.set_xticks,
                                        ax.get_xticklabels,
                                        ax.set_xticklabels)
        cls.format_numbers_for_one_axis(ax,
                                        (0.01, 0.92),
                                        ax.get_yticks,
                                        ax.set_yticks,
                                        ax.get_yticklabels,
                                        ax.set_yticklabels)

    @classmethod
    def update_axis_font_size(cls, ax, size=16):
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
            ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(size)

    @classmethod
    def update_lgd_font_size(cls, lgd, size=16):
        for text in lgd.texts:
            text.set_fontsize(size)
        lgd.get_title().set_fontsize(size)

    @classmethod
    def update_line_size(cls, ax, size=2.2):
        lines = ax.get_lines()
        for line in lines:
            line.set_linewidth(size)

    @classmethod
    def combine_plots(cls, filenames, output_folder, output_file_notation, max_figs_per_row=2, increase_line_size=False):
        '''wrapper for graph combining process'''
        with PlotCombiner() as combiner:
            figs = [combiner.open_figure(x, bool(i)) for i, x in enumerate(filenames)]
            for fig in figs:
                ax = fig[0].axes[0]
                cls.update_axis_font_size(ax)
                if increase_line_size:
                    cls.update_line_size(ax)

            n_figs = len(figs)
            rows = ceil(n_figs / max_figs_per_row)
            final_fig = cls.combine_figures(figs, rows, max_figs_per_row)
            output_filename = str(output_folder) + f"/combined_{output_file_notation}.png"
            final_fig.save(output_filename, dpi=(300,300))

if __name__ == "__main__":
    def inc(tgt:str, band:str, flt:bool, x):
        if tgt not in x:
            return False
        if band not in x:
            return False
        if "filtered" in x and not flt:
            return False
        if flt and "filtered"not in x:
            return False

        return True

    combiner = PlotCombiner
    targets = ["Site", "Hour"]
    bands = ["broadband", "shrimp", "fish"]
    filtered = [True, False]
    sorting = lambda x: x.name.split('_')[6]
    for target, band, fltr in tqdm(product(targets, bands, filtered)):
        only_inc = partial(inc, target, band, fltr)
        filenames = combiner.get_figures_from_folders("output", only_inc, sorting)
        if not filenames: continue
        combiner.combine_plots(filenames, "output", f"{target}_{band}_{fltr}")


    matplotlib.rcParams.update({'font.size': 16, 'figure.dpi': 300})
    filenames = [f"output/{x}_call frequency effect_rate_plot.pkl" for x in ["ACI", "ADI", "AEI", "BIO"]]
    combiner.combine_plots(filenames, "output", f"frequency_all", increase_line_size=True)

    filenames = [f"output/{x}_call frequency effect_rate_plot.pkl" for x in ["ACI", "BIO"]]
    combiner.combine_plots(filenames, "output", f"frequency_ACI_BIO", increase_line_size=True)

    filenames = ["output/Hour_x_Window_conditional_effects_for_AEI_over_filtered_shrimp_frequencies_Python.pkl",
                "output/Hour_x_Window_conditional_effects_for_AEI_over_shrimp_frequencies_Python.pkl"]
    combiner.combine_plots(filenames, "output", f"shrimp_AEI")

    filenames = ["output/Hour_x_Window_conditional_effects_for_ADI_over_filtered_fish_frequencies_Python.pkl",
                "output/Hour_x_Window_conditional_effects_for_ADI_over_fish_frequencies_Python.pkl"]
    combiner.combine_plots(filenames, "output", f"fish_ADI")

    filenames = [f"output/{freq}_Hz_{window}_FFT_simulated_calls.pkl" for freq in (1200, 12000) for window in (256, 4096)]
    combiner.combine_plots(filenames, "output", f"simulated")

    filename = "output/ACI_2048.pkl"
    fig = pickle.load(open(filename, 'rb'))
    ax = fig.axes[0]
    PlotCombiner.update_axis_font_size(ax)
    lgd = ax.get_legend()
    PlotCombiner.update_lgd_font_size(lgd)
    PlotCombiner.update_line_size(ax)
    fig.savefig("output/ACI_2048_updated.png", bbox_inches="tight", dpi=300)

