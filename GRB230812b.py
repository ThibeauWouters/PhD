import numpy as np

# Results from run 29/09/2023

Z_LANL = -83.284
Z_Ka = -83.686
Z_Bu = -83.584
Z = -83.701
Z_SN = -60.207

# Bayes factors: preference wrt base model being just GRB:

Z_list = [Z_LANL, Z_Bu, Z_Ka, Z_SN]
names = ["LANL", "Bulla", "Kasen", "SN"]

for i, name in enumerate(names):
    z = Z_list[i]
    print(f"Bayes factor {name} is {np.round(z - Z, 4)}")
    
# Bayes factor LANL is 0.417
# Bayes factor Bulla is 0.117
# Bayes factor Kasen is 0.015
# Bayes factor SN is 23.494