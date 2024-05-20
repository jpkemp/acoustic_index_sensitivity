from rpy2 import rinterface_lib
from rpy2.robjects import pandas2ri, FactorVector
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects

Rplus = robjects.r['+']

class Rlink:
    brms = importr("brms")
    grDevices = importr('grDevices')
    base = importr('base')
    gg = importr('ggplot2')
    stats = importr('stats')

    def __init__(self) -> None:
        self.r_src = None
        self.null_value = robjects.rinterface.NULL

    def load_src(self, source):
        from rpy2.robjects.packages import STAP
        with open(source, 'r') as f:
            inpt = f.read()

        self.r_src = STAP(inpt, "str")

    def save_workspace(self, path):
        self.base.save_image(str(path))

    def convert_to_rdf(self, df):
        context = self.context()
        with context():
            return pandas2ri.py2rpy(df)

    @classmethod
    def change_col_to_factor(cls, r_df, col):
        col_index = list(r_df.colnames).index(col)
        col_vals = FactorVector(r_df.rx2(col))
        r_df[col_index] = col_vals

    @classmethod
    def gr_plot(cls, filename, plot_object, y_limit_change=None):
        cls.grDevices.png(filename, width=1600, height=1600) # pylint: disable=no-member)
        if y_limit_change:
            plot_object = Rplus(plot_object, cls.gg.ylim(y_limit_change))

        cls.base.plot(plot_object)
        cls.grDevices.dev_off() # pylint: disable=no-member

    @classmethod
    def get_conditional_effects(cls, model, title, file_path):
        fname = title.replace(' ', '_')
        filename = file_path / fname
        theme = cls.gg.theme_minimal(base_size=36, base_line_size=0.5, base_rect_size=0.5)
        # plt = cls.base.plot(model, ask=False, plot=False, theme=theme)
        # for i in range(len(plt)):
        #     cls.gr_plot(f"{filename}_base_{i}.png", plt[i])

        effects = {}
        for points in [True]:
            eff = cls.brms.conditional_effects(model)
            effects[points] = eff
            # obj = cls.base.plot(eff, ask=False, plot=False, points=points, theme=theme)
            # for i in range(len(obj)):
            #     cls.gr_plot(f"{filename}_cond_{points}_{i}.png", obj[i])

        return effects

    def context(self):
        return (robjects.default_converter + pandas2ri.converter).context

    @classmethod
    def capture_rpy2_output(cls, errorwarn_callback=None, print_callback=None):
        '''Prevent R output being written to console, for clean logging'''
        if not print_callback:
            print_callback = lambda x: None

        if not errorwarn_callback:
            errorwarn_callback = lambda x: None

        rinterface_lib.callbacks.consolewrite_print = print_callback
        rinterface_lib.callbacks.consolewrite_warnerror = errorwarn_callback
