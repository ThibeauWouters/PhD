import numpy as np

# Results from run 29/09/2023



# Bayes factors: preference wrt base model being just GRB:

Z_list = [-93.278, -42.866, -92.718, -55.997, -94.767, -61.447]
names = ["Powerlaw", "Powerlaw-SN", "Gauss", "Gauss-SN", "Tophat", "Tophat-SN"]

base_idx = np.argmax(Z_list)

base_Z = Z_list.pop(base_idx)
base_name = names.pop(base_idx)

Z_list = np.array(Z_list)
names = np.array(names)

sort_idx = np.argsort(Z_list)[::-1]

Z_list = Z_list[sort_idx]
names = names[sort_idx]

# Print to screen
print(f"Best performing: {base_name}")
print(f"Bayes factors:")
for z, name in zip(Z_list, names):
    print(f"   - {name}: {np.round(z - base_Z, 4)}")
    