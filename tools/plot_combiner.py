'''graph combination'''
import io
import pickle
from functools import partial
from itertools import product
from math import ceil
from pathlib import Path
from tqdm import tqdm
from PIL import Image

class PlotCombiner:
    '''graph combining functions'''
    def __init__(self) -> None:
        self.fig_bufs = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for buf in self.fig_bufs:
            buf.close()

    def open_figure(self, filename, discard_lgd=False):
        fig = pickle.load(open(filename, 'rb'))
        lgd = fig.get_axes()[0].get_legend()
        if discard_lgd:
            lgd.remove()
            buf = None
        else:
            buf = io.BytesIO()
            self.fig_bufs.append(buf)
            self.export_legend(lgd, buf)
            lgd.remove()

        ax = fig.gca()
        ax.ticklabel_format()
        # ax.ticklabel_format(style='sci', axis='y', scilimits=(-5,5))
        t = ax.yaxis.get_offset_text()
        text = t.get_text()
        if text:
            t.set_visible(False)
            text = f"Value ($x10^{text[2:]}$)"
            ax.set_ylabel(text)

        return fig, buf

    @classmethod
    def export_legend(cls, legend, buf):
        fig  = legend.figure
        fig.canvas.draw()
        bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(buf, dpi="figure", bbox_inches=bbox)

    @classmethod
    def convert_plots_to_images(cls, plots):
        bufs = [io.BytesIO() for _ in range(len(plots))]
        images=[]
        lgds = []
        for i, (fig, lgd_buf) in enumerate(plots):
            buf = bufs[i]
            fig.savefig(buf, format="png", bbox_inches="tight")
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
        max_height = 0
        for i in range(0, len(subset), imgs_per_row):
            row_width = sum(widths[i:i+imgs_per_row])
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

            new_im.paste(im, (x_offset,y_offset))
            x_offset += im.size[0]

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
    def combine_plots(cls, filenames, output_folder, output_file_notation, max_figs_per_row=2):
        '''wrapper for graph combining process'''
        with PlotCombiner() as combiner:
            figs = [combiner.open_figure(x, bool(i)) for i, x in enumerate(filenames)]
            n_figs = len(figs)
            rows = ceil(n_figs / max_figs_per_row)
            final_fig = cls.combine_figures(figs, rows, max_figs_per_row)
            output_filename = str(output_folder) + f"/combined_{output_file_notation}.png"
            final_fig.save(output_filename)

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


    filenames = [f"output/{x}_call frequency effect_rate_plot.pkl" for x in ["ACI", "ADI", "AEI", "BIO"]]
    combiner.combine_plots(filenames, "output", f"frequency_all")

    filenames = [f"output/{x}_call frequency effect_rate_plot.pkl" for x in ["ACI", "BIO"]]
    combiner.combine_plots(filenames, "output", f"frequency_ACI_BIO")

    filenames = ["output/Hour_x_Window_conditional_effects_for_AEI_over_filtered_shrimp_frequencies_Python.pkl",
                "output/Hour_x_Window_conditional_effects_for_AEI_over_shrimp_frequencies_Python.pkl"]
    combiner.combine_plots(filenames, "output", f"shrimp_AEI")
    
    filenames = ["output/Hour_x_Window_conditional_effects_for_ADI_over_filtered_fish_frequencies_Python.pkl",
                "output/Hour_x_Window_conditional_effects_for_ADI_over_fish_frequencies_Python.pkl"]
    combiner.combine_plots(filenames, "output", f"fish_ADI")

    
