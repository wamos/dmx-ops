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


# w1 = 5.078 + 46.498 
# w2 = 9.747 + 37.711
# w3 = 12.685 + 3.267
# w4 = 27.776 + 22.421
# w5 = 33.381 + 39.713

b1 = 10.078 + 46.498 
b2 = 9.747 + 37.711
b3 = 12.685 + 12.762
b4 = 22.421 + 65.601
b5 = 33.381 + 69.625

b1_k2 = 46.498 
b2_k2 = 37.711
b3_k2 = 12.762
b4_k2 = 65.601
b5_k2 = 69.625

#second_cpu_kernel = np.array([46.498, 46.498, 37.711, 37.711, 3.267, 3.267, 22.421, 22.421, 39.713, 39.713])

# 4, 8, 16 cores for 2 and 16 kernels
# num_cores = 4
# num_cores = int(sys.argv[1])
# second_cpu_kernel = np.array([46.498, 46.498, 37.711, 37.711, 3.267, 3.267, 22.421, 22.421, 39.713, 39.713])
# cpu_kernel  = np.array([ w1, w1, w2, w2, w3, w3, w4, w4, w5 ,w5])

# fig_title = f'{num_cores} cores with proportional LLC and memory bw \nw for workload order, c for # of cores, k for # of kernels'
# labels = [f'w1-{num_cores}c-2k', f'w1-{num_cores}c-16k', f'w2-{num_cores}c-2k' , f'w2-{num_cores}c-16k', 
#           f'w3-{num_cores}c-2k' , f'w3-{num_cores}c-16k', f'w4-{num_cores}c-2k', f'w4-{num_cores}c-16k', 
#           f'w5-{num_cores}c-2k', f'w5-{num_cores}c-16k']
# ## 4 cores
# if num_cores == 4:
#     e2e = np.array([ 69.507, 390.280, 65.298, 355.582, 57.478, 365.522, 76.573, 226.377, 142.416, 670.138])
# elif num_cores == 8:
#     e2e = np.array([ 55.943, 171.499, 31.421, 166.630, 29.213, 157.239, 72.681, 108.156, 85.514, 294.272])
# elif num_cores == 16:
#     e2e = np.array([51.943, 132.137, 26.849, 162.747, 28.762, 149.923, 70.983, 98.271, 79.973, 247.592])
# else:
#     print("invalid num of cores")
#     exit()

# 4, 8, 16 cores for 1, 5 ,10, 15 kernels
benchmark_name = sys.argv[1]
mode = "latency"
mode = sys.argv[2]

cpu_kernel = np.ones(12)
second_cpu_kernel = np.ones(12)


fig_title = f'{benchmark_name}: 4, 8, and 16 cores with proportional LLC and memory bw \nc for # of cores, k for # of kernels'

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
#print(f"data_motion_ratio:{data_motion_ratio}")
cpu_kernel_ratio  = cpu_kernel/total
print(f"{benchmark_name}:")
print("e2e total running time:")
print(f"4 cores:{total[0:4]}")
print(f"8 cores:{total[4:8]}")
print(f"16 cores:{total[8:12]}")

print("breakdown cpu kernel ratio:")
print(f"4 cores:{cpu_kernel_ratio[0:4]}")
print(f"8 cores:{cpu_kernel_ratio[4:8]}")
print(f"16 cores:{cpu_kernel_ratio[8:12]}")

fig, ax = plt.subplots(figsize=(15,5))
#ax.bar(labels, e2e, width=0.5, label='emulated kernel + data motion')
if mode == "latency":
    ax.bar(labels, cpu_kernel,width=0.5, label='kernel')
    bottom = cpu_kernel
    p = ax.bar(labels, data_motion, width=0.5, bottom=bottom, label='data restructuring')
    # bottom = acc_kernel + data_motion
    # p = ax.bar(labels, data_movement, width=0.5, bottom=bottom, label='data movement')
    ax.bar_label(p, fmt='%.2f', label_type='edge')

    ax.legend(loc='best')
    ax.set_ylabel('Latency (ms)')
    title = f"latency_stacks_{benchmark_name}_allcpu.png"

elif mode == "breakdown":
    ax.bar(labels, cpu_kernel_ratio, width=0.5, label='kernel')
    bottom = cpu_kernel_ratio
    ax.bar(labels, data_motion_ratio, width=0.5, bottom=bottom, label='data motion')
    ax.legend(loc='best')
    ax.set_ylabel('Percentage (%)')
    title = f"breakdown_{benchmark_name}_allcpu.png"

ax.grid(True)
ax.set_title(fig_title)
plt.savefig(title, format='png', dpi=200)

#fig, ax = plt.subplots(figsize=(5,5))
