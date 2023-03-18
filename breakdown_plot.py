import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gmean
import sys

#B1: camera: video -> YOLO, 1080*720*4*32
#B2: audio:  FFT -> SVM, 1024*768
#B3: neural: FFT -> PPO
#B4: PII:    AES -> regex
#B5: database:  Gzip -> hash-join

#fig, ax = plt.subplots()
# CPU baseline, ARM baseline, PCIe-SIMD, On-device-SIMD

# benchmark-2, no RDT
# labels = ['1 kernel', '2 kernels', '4 kernels', '8 kernels', '16 kernels']
# data_motion= np.array([0.126, 0.235, 0.559, 0.762, 0.880])
# kernel = np.array([0.874, 0.765, 0.441, 0.238, 0.120])

# benchmark 1 to 4.
# labels = ['2 kernels\nbench-1', '16 kernels\nbench-1', '2 kernels\nbench-2', '16 kernels\nbench-2', '2 kernels\nbench-3', '16 kernels\nbench-3', '2 kernels\nbench-4', '16 kernels\nbench-4']
# data_motion= np.array([0.437, 0.888, 0.426, 0.875, 0.224, 0.749, 0.151, 0.693])
# kernel = np.array([0.563, 0.112, 0.574, 0.125, 0.776, 0.251, 0.849, 0.307])

num_cores = 4
num_cores = int(sys.argv[1])

w1 = 5.078 + 46.498 
w2 = 9.747 + 37.711
w3 = 12.685 + 3.267
w4 = 65.601 + 22.421
w5 = 33.381 + 39.713

second_cpu_kernel = np.array([46.498, 46.498, 37.711, 37.711, 3.267, 3.267, 22.421, 22.421, 39.713, 39.713])

cpu_kernel  = np.array([ w1, w1, w2, w2, w3, w3, w4, w4, w5 ,w5])


fig_title = f'{num_cores} cores with proportional LLC and memory bw \nw for workload order, c for # of cores, k for # of kernels'
labels = [f'w1-{num_cores}c-2k', f'w1-{num_cores}c-16k', f'w2-{num_cores}c-2k' , f'w2-{num_cores}c-16k', 
          f'w3-{num_cores}c-2k' , f'w3-{num_cores}c-16k', f'w4-{num_cores}c-2k', f'w4-{num_cores}c-16k', 
          f'w5-{num_cores}c-2k', f'w5-{num_cores}c-16k']
## 4 cores
if num_cores == 4:
    e2e = np.array([ 69.507, 390.280, 65.298, 355.582, 57.478, 365.522, 76.573, 226.377, 142.416, 670.138])
elif num_cores == 8:
    e2e = np.array([ 55.943, 171.499, 31.421, 166.630, 29.213, 157.239, 72.681, 108.156, 85.514, 294.272])
elif num_cores == 16:
    e2e = np.array([51.943, 132.137, 26.849, 162.747, 28.762, 149.923, 70.983, 98.271, 79.973, 247.592])
else:
    print("invalid num of cores")
    exit()

total = e2e + second_cpu_kernel
data_motion = total - cpu_kernel
data_motion_ratio = data_motion/total
cpu_kernel_ratio  = cpu_kernel/total


fig, ax = plt.subplots(figsize=(15,5))
#ax.bar(labels, e2e, width=0.5, label='emulated kernel + data motion')
ax.bar(labels, cpu_kernel_ratio, width=0.5, label='kernel')
bottom = cpu_kernel_ratio
ax.bar(labels, data_motion_ratio, width=0.5, bottom=bottom, label='data motion')
ax.legend(loc='best')
#ax.set_ylabel('Percentage (%)')
ax.grid(True)
ax.set_ylabel('Latency (ms)')
#plt.show()
ax.set_title(fig_title)
plt.savefig(f'breakdown_{num_cores}cores'+'.png', format='png', dpi=200)

#fig, ax = plt.subplots(figsize=(5,5))
