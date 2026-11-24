from . import aad, cowa, dac, gsfda, mbns, nrc, shot, smna

BASELINES = {"shot": shot, "nrc": nrc, "cowa": cowa, "mbns": mbns, "gsfda": gsfda,
             "aad": aad, "dac": dac, "smna": smna}

DISPLAY_NAMES = {"shot": "SHOT", "nrc": "NRC", "cowa": "CoWA", "mbns": "MBNS", "gsfda": "G-SFDA",
                 "aad": "AaD", "dac": "DaC", "smna": "SMNA", "cab_sfda": "CAB-SFDA",
                 "source_only": "Source only"}
