import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gmean
import sys
from pcie_dma_model import dma_time

#TODO:
# this doesn't have end-to-end meausurement, data motion on DMX is simulated by genesys.sim 
# we still have kernel numbers from acc 
# will have different dmx placement options: cpu, pcie, acc

b1_shape = (4,1024,768)
b1_data_size = b1_shape[0] * b1_shape[1] * b1_shape[2]

b2_shape = (4,1024,1024)
b2_data_size = b2_shape[0] * b2_shape[1] * b2_shape[2]

b3_shape = (4,1024,768)
b3_data_size = b3_shape[0] * b3_shape[1] * b3_shape[2]

b4_shape = (128,768,16)
b4_data_size = b4_shape[0] * b4_shape[1] * b4_shape[2]

b5_shape = (4, 1024, 512)
b5_data_size = b5_shape[0] * b5_shape[1] * b5_shape[2]

b1_k1 = 16.66/8
b2_k1 = 91.521/8
b3_k1 = 30.507/8 # checked
b4_k1 = 1.138*2
b5_k1 = 8.095

b1_k2 = 38.3/2  
b2_k2 = 21.12/8
b3_k2 = 31.85/8
b4_k2 = 7.761*1.5
b5_k2 = 6.602

# b1_dma = 921.316 * 0.001 * 2 # round-trip DMA in ms
# b2_dma = 1228.153 * 0.001 * 2
# b3_dma = 921.316 * 0.001 * 2
# b4_dma = 460.365 * 0.001 * 2
# b5_dma =  614.113 * 0.001 *2

#an end-to-end overhead of 70ns for CXL reads, compared to NUMA-local DRAM reads
control_pool_overhead = 70*0.001*0.001 # ns to ms

benchmark_name = sys.argv[1]
# pci_gen = sys.argv[2]
# cpu_vendor = sys.argv[3]
pci_gen = "gen3"
cpu_vendor = "intel"
dmx_placement = "cpu"
# def dma_time(data_size:int, num_kernel: int, pcie_gen: str, cpu_vendor: str) -> float: 
# b1_dma_1k = dma_time(dmx_placement, b1_data_size, 1, pci_gen, cpu_vendor)
# b1_dma_5k = dma_time(dmx_placement, b1_data_size, 5, pci_gen, cpu_vendor)
# b1_dma_10k = dma_time(dmx_placement, b1_data_size, 10, pci_gen, cpu_vendor)
# b1_dma_15k = dma_time(dmx_placement, b1_data_size, 15, pci_gen, cpu_vendor)
#print([b1_dma_1k, b1_dma_5k, b1_dma_10k, b1_dma_15k])

b1_dma_1k = dma_time(dmx_placement, b1_data_size, 1, pci_gen, cpu_vendor)
b1_dma_5k = dma_time(dmx_placement, b1_data_size, 5, pci_gen, cpu_vendor)
b1_dma_10k = dma_time(dmx_placement, b1_data_size, 10, pci_gen, cpu_vendor)
b1_dma_15k = dma_time(dmx_placement, b1_data_size, 15, pci_gen, cpu_vendor)
#print([b1_dma_1k, b1_dma_5k, b1_dma_10k, b1_dma_15k])

b2_dma_1k = dma_time(dmx_placement, b2_data_size, 1, pci_gen, cpu_vendor)
b2_dma_5k = dma_time(dmx_placement, b2_data_size, 5, pci_gen, cpu_vendor)
b2_dma_10k = dma_time(dmx_placement, b2_data_size, 10, pci_gen, cpu_vendor)
b2_dma_15k = dma_time(dmx_placement, b2_data_size, 15, pci_gen, cpu_vendor)

b3_dma_1k = dma_time(dmx_placement, b3_data_size, 1, pci_gen, cpu_vendor)
b3_dma_5k = dma_time(dmx_placement, b3_data_size, 5, pci_gen, cpu_vendor)
b3_dma_10k = dma_time(dmx_placement, b3_data_size, 10, pci_gen, cpu_vendor)
b3_dma_15k = dma_time(dmx_placement, b3_data_size, 15, pci_gen, cpu_vendor)

b4_dma_1k = dma_time(dmx_placement, b4_data_size, 1, pci_gen, cpu_vendor)
b4_dma_5k = dma_time(dmx_placement, b4_data_size, 5, pci_gen, cpu_vendor)
b4_dma_10k = dma_time(dmx_placement, b4_data_size, 10, pci_gen, cpu_vendor)
b4_dma_15k = dma_time(dmx_placement, b4_data_size, 15, pci_gen, cpu_vendor)

b5_dma_1k = dma_time(dmx_placement, b5_data_size, 1, pci_gen, cpu_vendor)
b5_dma_5k = dma_time(dmx_placement, b5_data_size, 5, pci_gen, cpu_vendor)
b5_dma_10k = dma_time(dmx_placement, b5_data_size, 10, pci_gen, cpu_vendor)
b5_dma_15k = dma_time(dmx_placement, b5_data_size, 15, pci_gen, cpu_vendor)

b1 = b1_k1 + b1_k2
#b1_movement = b1_dma + control_pool_overhead 
b2 = b2_k1 + b2_k2
#b2_movement = b2_dma + control_pool_overhead 
b3 = b3_k1 + b3_k2 
#b3_movement = b3_dma + control_pool_overhead
b4 = b4_k1 + b4_k2
#b4_movement = b4_dma + control_pool_overhead
b5 = b5_k1 + b5_k2
#b5_movement = b5_dma + control_pool_overhead
# 4, 8, 16 cores for 1, 5 ,10, 15 kernels


acc_kernel = np.ones(12)
second_kernel = np.ones(12)
#data_movement = np.ones(12)


fig_title = f'{benchmark_name}: 4, 8, and 16 cores with proportional LLC and memory bw\n' + "c for # of cores, k for # of kernels, " + f"PCIe {pci_gen} with CPU vendor {cpu_vendor}"


labels = [f'4c-1k', f'4c-5k', f'4c-10k' , f'4c-15k', 
          f'8c-1k', f'8c-5k', f'8c-10k' , f'8c-15k',  
          f'16c-1k', f'16c-5k', f'16c-10k' , f'16c-15k']
if benchmark_name == "benchmark2":
    e2e = np.array([64.869, 108.695, 217.210, 331.811,
                    28.110, 51.153, 102.879, 154.617,
                    22.334, 50.579, 100.806, 153.319])
    second_kernel.fill(b2_k2)
    acc_kernel.fill(b2)
    data_movement = [b2_dma_1k, b2_dma_5k, b2_dma_10k, b2_dma_15k] * 3
    data_movement = np.array(data_movement)
    data_movement = data_movement + control_pool_overhead    
    #print(data_movement)
    #exit()
elif benchmark_name == "benchmark3":
    e2e = np.array([57.164, 107.751, 213.897, 328.855,
                    30.178, 47.783, 95.118, 145.566,
                    20.77, 47.276, 93.770, 142.145])
    second_kernel.fill(b3_k2)
    acc_kernel.fill(b3)
    data_movement = [b3_dma_1k, b3_dma_5k, b3_dma_10k, b3_dma_15k] * 3
    data_movement = np.array(data_movement)
    data_movement = data_movement + control_pool_overhead
elif benchmark_name == "benchmark1":
    e2e = np.array([65.756, 116.895, 230.675, 354.092,
                    50.006, 66.734, 103.713, 157.667,
                    47.851, 64.933, 92.831, 126.483])
    second_kernel.fill(b1_k2)
    acc_kernel.fill(b1)
    data_movement = [b1_dma_1k, b1_dma_5k, b1_dma_10k, b1_dma_15k] * 3
    data_movement = np.array(data_movement)
    data_movement = data_movement + control_pool_overhead
elif benchmark_name == "benchmark4":
    e2e = np.array([76.118, 76.981, 129.35, 197.469,
                    71.135, 72.709, 74.250, 104.201,
                    70.431, 71.660, 73.014, 93.102])
    second_kernel.fill(b4_k2)
    acc_kernel.fill(b4)
    data_movement = [b4_dma_1k, b4_dma_5k, b4_dma_10k, b4_dma_15k] * 3
    data_movement = np.array(data_movement)
    data_movement = data_movement + control_pool_overhead
elif benchmark_name == "benchmark5":
    e2e = np.array([136.946, 201.275, 404.315, 614.216,
                    75.604, 114.644, 183.729, 270.437,
                    62.378, 110.997, 169.177, 232.950])
    second_kernel.fill(b5_k2)
    acc_kernel.fill(b5)
    data_movement = [b5_dma_1k, b5_dma_5k, b5_dma_10k, b5_dma_15k] * 3
    data_movement = np.array(data_movement)
    data_movement = data_movement + control_pool_overhead
else:
    print(f"invalid benchmark name {benchmark_name}")

data_movement = data_movement*2 # rx + tx
total = e2e + second_kernel + data_movement
data_motion = total - acc_kernel - data_movement
print(f"data motion: {data_motion}")
data_motion_ratio = data_motion/total
data_movement_ratio = data_movement/total
acc_kernel_ratio  = acc_kernel/total
#print(data_movement_ratio)
print(f"data_movement: {data_movement}")

fig, ax = plt.subplots(figsize=(15,5))
#ax.bar(labels, e2e, width=0.5, label='emulated kernel + data motion')
ax.bar(labels, acc_kernel,width=0.5, label='kernel')
bottom = acc_kernel
ax.bar(labels, data_motion, width=0.5, bottom=bottom, label='data motion')
bottom = acc_kernel + data_motion
p = ax.bar(labels, data_movement, width=0.5, bottom=bottom, label='dma')
data_movement = np.round(data_movement,2)
ax.bar_label(p, data_movement, fmt='%.2f', label_type='edge')

ax.legend(loc='best')
ax.set_ylabel('Latency (ms)')

# ax.bar(labels, acc_kernel_ratio, width=0.5, label='kernel')
# bottom = acc_kernel_ratio
# ax.bar(labels, data_motion_ratio, width=0.5, bottom=bottom, label='data motion')
# bottom = acc_kernel_ratio + data_motion_ratio
# ax.bar(labels, data_movement_ratio, width=0.5, bottom=bottom, label='dma')
# ax.legend(loc='best')
# ax.set_ylabel('Percentage (%)')

ax.grid(True)
#ax.set_ylabel('Latency (ms)')
#plt.show()
ax.set_title(fig_title)
plt.savefig(f'latency_stacks_{benchmark_name}_acc.png', format='png', dpi=200)
#plt.savefig(f'breakdown_{benchmark_name}_acclatency.png', format='png', dpi=200)

#fig, ax = plt.subplots(figsize=(5,5))
