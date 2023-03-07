import matplotlib.pyplot as plt
import numpy as np

#B1: camera: video -> YOLO, 1080*720*4*32
#B2: audio:  FFT -> SVM, 1024*768
#B3: neural: FFT -> PPO
#B4: PII:    AES -> regex
#B5: database:  Gzip -> hash-join

labels = ['1 kernel', '2 kernels', '4 kernels', '8 kernels', '16 kernels']
#fig, ax = plt.subplots()
# CPU baseline, ARM baseline, PCIe-SIMD, On-device-SIMD

# benchmark-2
data_motion= np.array([0.126, 0.235, 0.559, 0.762, 0.880])
kernel = np.array([0.874, 0.765, 0.441, 0.238, 0.120])

#data_motion= np.array([0.063, 0.263, 0.494, 0.457])
#kernel = np.array([0.937, 0.737, 0.506, 0.543])


fig, ax = plt.subplots(figsize=(5,5))
ax.bar(labels, kernel, width=0.3, label='kernel')
bottom = kernel
ax.bar(labels, data_motion, width=0.3, bottom=bottom, label='data motion')
ax.legend(loc='best')
ax.set_ylabel('Percentage (%)')
#plt.show()
plt.savefig('breakdown'+'.png', format='png', dpi=200)

#fig, ax = plt.subplots(figsize=(5,5))
