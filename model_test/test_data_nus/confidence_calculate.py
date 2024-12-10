import numpy as np

def confidence_mean(l,p):
    l = np.sort(l)
    drop = int(len(l)*p)
    l = l[drop:len(l)-drop]
    confidencemean = np.mean(l)

    return confidencemean,l
