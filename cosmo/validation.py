from cosmo.exceptions import InputValidationError, NotFitted, TestUnitError

# ===========================================
def validate_str_in_list(string, strings):
    if string not in strings:
        raise InputValidationError("string '{}' should be one of {}".format(string, strings))
        
# ===========================================
def validate_int_higher(k, low):
    if not k > low: raise InputValidationError("The specified integer {} should be higher than {}".format(k, low))

# ===========================================
def validate_list_not_empty(lst):
    if len(lst) == 0:
        raise InputValidationError("The list should not be empty")

# ===========================================
def validate_all_in_range(lst, bounds):
    low, up = bounds
    for v in lst:
        if not (low <= v <= up):
            raise InputValidationError("Values of the list should be in [{}, {}]".format(low, up))
            
# ===========================================
def validate_is_fitted(is_fitted):
    if not is_fitted:
        raise NotFitted("'fit' must be called before calling 'get'")
        
# ===========================================
def validate_reference_grouping(uid_test, dt, dffs):
    if not (0 <= uid_test < len(dffs)):
        raise InputValidationError("uid_test should be in range(nb_units). Given uid_test is {}, nb_units is {}".format(uid_test, len(dffs)))
        
    if len(dffs[uid_test]) == 0:
        raise TestUnitError("Test unit (uid={}) does not have data".format(uid_test))
    
    try:
        x = dffs[uid_test].loc[dt].values
    except KeyError:
        raise TestUnitError("Test unit (uid={}) does not have data at time {}".format(uid_test, dt))
        
# ===========================================





