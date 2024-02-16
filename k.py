import numpy as np
import pandas as pd

k = np.array([
    [0,1,2,3],
    [4,5,6,7],
    [8,9,10,11]
])
print(pd.DataFrame(k,columns = ['haudrsoff dis','sensi','speci','dice'],index=['et','tc','wt']))