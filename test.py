import numpy as np
z = np.array([1, np.nan])
print(z)
print(z.argmax())
print(z[z.argmax()])