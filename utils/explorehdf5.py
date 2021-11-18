import h5py
from matplotlib import pyplot as plt 
import numpy as np

f = h5py.File(r"C:\drive\project\project_ecg\data_dump\automatic_ecg_diagnosis\data\ecg_tracings.hdf5", "r")

print(type(f))
print(list(f))

tracings = f['tracings']
print(type(tracings))

print(tracings.shape)
print(tracings.dtype)

ecg_0 = tracings[0]

print(type(ecg_0))
print(ecg_0.shape)

ecg_0 = np.transpose(ecg_0)
print(ecg_0.shape)
# print(ecg_0)

# plt.plot(ecg_0)
# plt.show()

ecg_0_lead_1 = ecg_0[0]
print(ecg_0_lead_1)

plt.plot(ecg_0_lead_1)
plt.show()


lead_signals = []
