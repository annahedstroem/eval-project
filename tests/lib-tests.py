import os
import sys
import numpy as np

# Import library from src as fl (faithfulness library)
PROJ_DIR = os.path.realpath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.join(PROJ_DIR,'src'))
import xai_faithfulness_experiments_lib_edits as fl

# Test _attributions_to_ranking_row
np.testing.assert_array_equal(fl._attributions_to_ranking_row(np.array([0.55, 0.3, 1.4, -3.2, -4])), np.array([0.75, 0.5,  1.0, 0.25, 0.0]))