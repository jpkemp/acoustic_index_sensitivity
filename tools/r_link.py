import pandas as pd
from rpy2 import rinterface_lib
from rpy2.robjects import pandas2ri, FactorVector
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects


class Rlink:
    ''' Contains wrapper functions for linking to R
    '''
    brms = importr("brms")
    grDevices = importr('grDevices')
    base = importr('base')
    gg = importr('ggplot2')
    stats = importr('stats')
    at = robjects.r['@']
    plus = robjects.r['+']

    def __init__(self) -> None:
        self.r_src = None
        self.null_value = robjects.rinterface.NULL

    def load_src(self, source:str):
        '''load functions from an R script

           source: path to the source file

           returns: None
        '''
        from rpy2.robjects.packages import STAP
        with open(source, 'r') as f:
            inpt = f.read()

        self.r_src = STAP(inpt, "str")

    def save_workspace(self, path:str):
        '''save the current R workspace to a file
        path: the file to save to
        
        returns: None'''
        self.base.save_image(str(path))

    def convert_to_rdf(self, df:pd.DataFrame):
        '''convert a pandas dataframe to an R dataframe'''
        context = self.context()
        with context():
            return pandas2ri.py2rpy(df)

    @classmethod
    def change_col_to_factor(cls, r_df, col):
        '''converts an R dataframe column to a FactorVector
        r_df: the R dataframe
        col: column to convert
        
        returns: None'''
        col_index = list(r_df.colnames).index(col)
        col_vals = FactorVector(r_df.rx2(col))
        r_df[col_index] = col_vals

    @classmethod
    def get_conditional_effects(cls, model):
        '''gets conditional effects value from a brms model
        model: the brms model

        returns: None
        '''
        effects = {}
        for points in [True]:
            eff = cls.brms.conditional_effects(model)
            effects[points] = eff

        return effects

    def context(self):
        '''gets the rpy2 pandas2ri context, for converting dataframes
        
        returns: rpy2 context'''
        return (robjects.default_converter + pandas2ri.converter).context

    @classmethod
    def capture_rpy2_output(cls, errorwarn_callback=None, print_callback=None):
        '''Prevent R output being written to console, for clean logging
        errorwarn_callback: function for handling errors and warnings
        print_callback: functions for handling print statements
        
        returns: None'''
        if not print_callback:
            print_callback = lambda x: None

        if not errorwarn_callback:
            errorwarn_callback = lambda x: None

        rinterface_lib.callbacks.consolewrite_print = print_callback
        rinterface_lib.callbacks.consolewrite_warnerror = errorwarn_callback
