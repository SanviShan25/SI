import numpy as np

def allan_curve(freq):

    taus = [1,10,50,100]

    devs = []

    for tau in taus:

        diff = freq[tau:] - freq[:-tau]

        dev = np.sqrt(
            0.5*np.mean(diff**2)
        )

        devs.append(dev)

    return taus,devs