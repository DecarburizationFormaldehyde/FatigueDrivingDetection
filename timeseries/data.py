import numpy as np
import pandas as pd
data=np.random.randint(1,100,(50,))
data=pd.DataFrame(data)
data.to_csv("D:/random.csv")