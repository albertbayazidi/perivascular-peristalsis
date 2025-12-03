import numpy as np

COLOR_REM = '#1f77b4'  
COLOR_NON_REM = '#ff7f0e'  


STYLE_REM = {
    'color': COLOR_REM,
    'linestyle': '-',
    'label': 'REM'
}

STYLE_NON_REM = {
    'color': COLOR_NON_REM,
    'linestyle': '-',
    'label': 'Non-REM'
}

STYLE_MEAN_REM = {
    'color': COLOR_REM,  
    'linestyle': '--',
    'label': 'REM Mean'
}

STYLE_MEAN_NON_REM = {
    'color': COLOR_NON_REM,  
    'linestyle': '--',
    'label': 'Non-REM Mean'
}

FIG_SIZE = (12, 5) 

def get_time_indices(time_vec, t_start, t_end):
        # Start Index
        if t_start is None:
            idx_s = 0
        else:
            # Find the index where time_vec is closest to t_start
            idx_s = (np.abs(time_vec - t_start)).argmin()
            
        # End Index
        if t_end is None:
            idx_e = len(time_vec)
        else:
            idx_e = (np.abs(time_vec - t_end)).argmin()
            
        return idx_s, idx_e
