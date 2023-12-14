def argf(d:dict, filt:str) :
    """
    d : dictionary with keys "prefix1.prefix2.(...).prefixn.value"
    return a dict with {k:v} for all keys with prefix1 "filt" removing the prefix
    """
    return {'.'.join(k.split('.')[1:]):v for k,v in d.items() if k.split('.')[0] == filt}
