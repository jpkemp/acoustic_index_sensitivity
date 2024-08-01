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
    @classmethod
    def open_figure(cls, filename, discard_lgd=False):
        fig = pickle.load(open(filename, 'rb'))
        lgd = fig.get_axes()[0].get_legend()
        if discard_lgd:
            lgd.remove()
            buf = None
        else:
            buf = io.BytesIO()
            cls.export_legend(lgd, buf)
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
    def combine_figures(cls, plots, rows=2, imgs_per_row=None, spacing=20, include_lgd=True):
        '''combine graphs into one image. assumes all images same size if number of images does not equal rows * imgs_per_row'''
        bufs = [io.BytesIO() for _ in range(len(plots))]
        if imgs_per_row is None: imgs_per_row = len(plots)
        try:
            images=[]
            for i, (fig, lgd_buf) in enumerate(plots):
                buf = bufs[i]
                fig.savefig(buf, format="png", bbox_inches="tight")
                buf.seek(0)
                images.append(Image.open(buf))
                if lgd_buf: lgd = Image.open(lgd_buf)

            mod = len(images) % imgs_per_row
            final = []
            subset = images
            if mod:
                subset = images[:-mod]
                final = images[-mod:]

            widths, heights = zip(*(i.size for i in subset))
            lgd_width = 0
            if include_lgd:
                lgd_width += lgd.size[0] + spacing

            total_width = ceil(sum(widths) / imgs_per_row) + lgd_width
            max_height = ceil(max(heights) * rows) + (rows - 1) * (int(spacing))

            new_im = Image.new('RGB', (total_width, max_height), color="white")

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

            if include_lgd:
                x_offset = total_width - lgd_width - spacing
                y_offset = ceil((max_height / 2) - (lgd.size[1] / 2)) + 10
                new_im.paste(lgd, (x_offset,y_offset))

            
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
    def combine_plots(cls, output_folder, output_file_notation, only_include=None, sorting=None, max_figs_per_row=2):
        '''wrapper for graph combining process'''
        filenames = cls.get_figures_from_folders(output_folder, only_include, sorting)
        if not filenames: return
        figs = [cls.open_figure(x, bool(i)) for i, x in enumerate(filenames)]
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

    combiner = PlotCombiner()
    targets = ["Site", "Hour"]
    bands = ["broadband", "shrimp", "fish"]
    filtered = [True, False]
    sorting = lambda x: x.name.split('_')[6]
    for target, band, fltr in tqdm(product(targets, bands, filtered)):
        only_inc = partial(inc, target, band, fltr)
        combiner.combine_plots("output", f"{target}_{band}_{fltr}", only_inc, sorting)

