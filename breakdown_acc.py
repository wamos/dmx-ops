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



b1_k1 = 16.66/8
b2_k1 = 22.801/8
b3_k1 = 30.507/8
b4_k1 = 1.138/2
b5_k1 = 8.095/8

b1_k2 = 38.3/8  
b2_k2 = 21.12/8
b3_k2 = 31.85/8
b4_k2 = 7.761/8
b5_k2 = 6.602/8

b1_dma = 921.316 * 0.001 * 2 # round-trip DMA in ms
b2_dma = 1228.153 * 0.001 * 2
b3_dma = 921.316 * 0.001 * 2
b4_dma = 460.365 * 0.001 * 2
b5_dma =  614.113 * 0.001 *2

#an end-to-end overhead of 70ns for CXL reads, compared to NUMA-local DRAM reads
control_pool_overhead = 70*0.001*0.001 # ns to ms

b1 = b1_k1 + b1_k2 + b1_dma + control_pool_overhead
b2 = b2_k1 + b2_k2 + b2_dma + control_pool_overhead
b3 = b3_k1 + b3_k2 + b3_dma + control_pool_overhead
b4 = b4_k1 + b4_k2 + b4_dma + control_pool_overhead
b5 = b5_k1 + b5_k2 + b5_dma + control_pool_overhead
# 4, 8, 16 cores for 1, 5 ,10, 15 kernels
benchmark_name = sys.argv[1]

cpu_kernel = np.ones(12)
second_cpu_kernel = np.ones(12)


fig_title = f'{benchmark_name}: 4, 8, and 12 cores with proportional LLC and memory bw \nc for # of cores, k for # of kernels'

labels = [f'4c-1k', f'4c-5k', f'4c-10k' , f'4c-15k', 
          f'8c-1k', f'8c-5k', f'8c-10k' , f'8c-15k',  
          f'16c-1k', f'16c-5k', f'16c-10k' , f'16c-15k']
if benchmark_name == "benchmark2":
    e2e = np.array([64.448, 109.906, 224.654, 343.100,
                    28.070, 51.743, 102.899, 155.003,
                    22.706, 51.117, 101.510, 154.074])
    second_cpu_kernel.fill(b2_k2)
    cpu_kernel.fill(b2)
elif benchmark_name == "benchmark3":
    e2e = np.array([56.560, 106.481, 213.897, 324.556,
                    50.039, 47.955, 94.817, 145.658,
                    28.111, 47.446, 93.278, 141.426])
    second_cpu_kernel.fill(b3_k2)
    cpu_kernel.fill(b3)
elif benchmark_name == "benchmark1":
    e2e = np.array([65.778, 114.323, 231.393, 355.835,
                    34.533, 63.499, 105.247, 158.783,
                    48.078, 63.743, 91.783, 125.691])
    second_cpu_kernel.fill(b1_k2)
    cpu_kernel.fill(b1)
elif benchmark_name == "benchmark4":
    e2e = np.array([76.118, 76.981, 129.35, 197.469,
                    71.135, 72.709, 74.250, 104.201,
                    70.431, 71.660, 73.014, 93.102])
    second_cpu_kernel.fill(b4_k2)
    cpu_kernel.fill(b4)
elif benchmark_name == "benchmark5":
    e2e = np.array([136.946, 201.275, 404.315, 614.216,
                    75.604, 114.644, 183.729, 270.437,
                    62.378, 110.997, 169.177, 232.950])
    second_cpu_kernel.fill(b5_k2)
    cpu_kernel.fill(b5)
else:
    print(f"invalid benchmark name {benchmark_name}")

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
ax.set_ylabel('Percentage (%)')
ax.grid(True)
#ax.set_ylabel('Latency (ms)')
#plt.show()
ax.set_title(fig_title)
plt.savefig(f'breakdown_{benchmark_name}_acc.png', format='png', dpi=200)

#fig, ax = plt.subplots(figsize=(5,5))
