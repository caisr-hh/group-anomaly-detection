from collections import namedtuple
import numpy as np

# ===========================================
DeviationContext = namedtuple('DeviationContext', 'strangeness pvalue deviation is_deviating')

# ===========================================
class NotFittedError(Exception): pass
class InputValidationError(Exception): pass
class TestUnitError(Exception): pass
class NoRefGroupError(Exception): pass

# ===========================================
def validate_measure_str(string):
    strings = ["median", "knn", "lof"]
    if string not in strings:
        raise InputValidationError("measure should be one of {}. Given measure = '{}'".format(strings, string))
        
def validate_int_higher(k, low):
    if not k > low: raise InputValidationError("The specified integer {} should be higher than {}".format(k, low))

def validate_list_not_empty(lst):
    if len(lst) == 0:
        raise InputValidationError("The list should not be empty")

def validate_all_in_range(lst, bounds):
    low, up = bounds
    for v in lst:
        if not (low <= v <= up):
            raise InputValidationError("Values of the list should be in [{}, {}]".format(low, up))
            
def validate_is_fitted(is_fitted):
    if not is_fitted:
        raise NotFittedError("'fit' must be called before calling 'get'")
        
def validate_reference_group(Xref):
    if len(Xref) == 0:
        raise NoRefGroupError("Empty reference group data.")
    
def validate_reference_grouping_input(uid_test, dt, dffs):
    if not (0 <= uid_test < len(dffs)):
        raise InputValidationError("uid_test should be in range(nb_units). Given uid_test = {}, nb_units is {}".format(uid_test, len(dffs)))
        
    if len(dffs[uid_test]) == 0:
        raise TestUnitError("Test unit (uid={}) does not have data".format(uid_test))
    
    try:
        x = dffs[uid_test].loc[dt].values
    except KeyError:
        raise TestUnitError("Test unit (uid={}) does not have data at time {}".format(uid_test, dt))
        
def validate_individual_deviation_params(w_martingale, non_conformity, k, dev_threshold):
    if w_martingale < 1:
        raise InputValidationError("w_martingale should be an integer higher than 0. Given w_martingale = {}".format(w_martingale))
    
    strings = ["median", "knn", "lof"]
    if non_conformity not in strings:
        raise InputValidationError("non_conformity should be one of {}. Given non_conformity = '{}'".format(strings, non_conformity))
    
    if non_conformity == "knn":
        if k < 1:
            raise InputValidationError("k should be an integer higher than 0. Given k = {}".format(k))
    
    if not (0 <= dev_threshold <= 1):
        raise InputValidationError("dev_threshold should be in [0, 1]. Given dev_threshold = {}".format(dev_threshold))
        
def validate_fit_input(X):
    if len(X) == 0 or not isinstance(X,(list, np.ndarray)):
        raise InputValidationError("X must be a non empty array-like of shape (n_samples, n_features)")
        
def validate_get_input(x):
    if len(x) == 0 or not isinstance(x,(list, np.ndarray)):
        raise InputValidationError("x must be a non empty array-like of shape (n_features,)")
        
# ===========================================





