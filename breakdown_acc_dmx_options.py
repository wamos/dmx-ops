import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gmean
import sys
from pcie_dma_model import dma_time

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

b1_shape = (4,1024,768)
b1_data_size = b1_shape[0] * b1_shape[1] * b1_shape[2] * 4 # 4-byte float

b2_shape = (4,1024,1024)
b2_data_size = b2_shape[0] * b2_shape[1] * b2_shape[2] * 4 # 4-byte float

b3_shape = (4,1024,768)
b3_data_size = b3_shape[0] * b3_shape[1] * b3_shape[2] * 4 # 4-byte float

b4_shape = (128,768,16)
b4_data_size = b4_shape[0] * b4_shape[1] * b4_shape[2] * 4 # 4-byte float

b5_shape = (4, 1024, 512)
b5_data_size = b5_shape[0] * b5_shape[1] * b5_shape[2] * 4 # 4-byte float

b1_k1 = 16.66/8
b2_k1 = 30.507/8*4/3 # scaled it right
b3_k1 = 30.507/8 # checked
b4_k1 = 1.138*2
b5_k1 = 8.095

b1_k2 = 38.3/2  
b2_k2 = 21.12/8
b3_k2 = 31.85/8
b4_k2 = 7.761
b5_k2 = 6.602

# b1_dma = 921.316 * 0.001 * 2 # round-trip DMA in ms
# b2_dma = 1228.153 * 0.001 * 2
# b3_dma = 921.316 * 0.001 * 2
# b4_dma = 460.365 * 0.001 * 2
# b5_dma =  614.113 * 0.001 *2

#an end-to-end overhead of 70ns for CXL reads, compared to NUMA-local DRAM reads
control_pool_overhead = 70*0.001*0.001 # ns to ms

benchmark_name = sys.argv[1]
mode = "latency"
mode = sys.argv[2]

b1_dmx = 4.607
b2_dmx = 1.491
b3_dmx = 1.383
b4_dmx = 2.097
b5_dmx = 2.789

# This is because CPU can only have half of the DMX units
# if dmx_placement == "cpu":
#     b1_dmx = b1_dmx*2
#     b2_dmx = b2_dmx*2
#     b3_dmx = b3_dmx*2
#     b4_dmx = b4_dmx*2
#     b5_dmx = b5_dmx*2

b1 = b1_k1 + b1_k2
#b1_movement = b1_dma + control_pool_overhead 
b2 = b2_k1 + b2_k2
#b2_movement = b2_dma + control_pool_overhead 
b3 = b3_k1 + b3_k2 
#b3_movement = b3_dma + control_pool_overhead
b4 = b4_k1 + b4_k2
#b4_movement = b4_dma + control_pool_overhead
b5 = b5_k1 + b5_k2

acc_kernel = np.ones(12)
#second_kernel = np.ones(12)
dmx_exec = np.ones(12)

pci_gen = "gen4"
cpu_vendor = "intel"

if benchmark_name == "benchmark2":
    data_size = b2_data_size
    dmx_time = b2_dmx
    acc_kernel.fill(b2)
elif benchmark_name == "benchmark3":
    data_size = b3_data_size
    dmx_time = b3_dmx
    acc_kernel.fill(b3)
elif benchmark_name == "benchmark1":
    data_size = b1_data_size
    dmx_time = b1_dmx
    acc_kernel.fill(b1)
elif benchmark_name == "benchmark4":
    data_size = b4_data_size
    dmx_time = b4_dmx
    acc_kernel.fill(b4)
elif benchmark_name == "benchmark5":
    data_size = b5_data_size
    dmx_time = b5_dmx
    acc_kernel.fill(b5)
else:
    print(f"invalid benchmark name {benchmark_name}")
    exit()

b_dma_cpu = [dma_time("cpu", data_size, 1, pci_gen, cpu_vendor), dma_time("cpu", data_size, 5, pci_gen, cpu_vendor), dma_time("cpu", data_size, 10, pci_gen, cpu_vendor), dma_time("cpu", data_size, 15, pci_gen, cpu_vendor)]
b_dma_pcie_overprovisioned = [dma_time("pcie", data_size, 1, pci_gen, cpu_vendor), dma_time("pcie", data_size, 5, pci_gen, cpu_vendor), dma_time("pcie", data_size, 10, pci_gen, cpu_vendor), dma_time("pcie", data_size, 15, pci_gen, cpu_vendor)]
#b_dma_pcie_underprovisioned = [dma_time("pcie-under", data_size, 1, pci_gen, cpu_vendor), dma_time("pcie-under", data_size, 5, pci_gen, cpu_vendor), dma_time("pcie-under", data_size, 10, pci_gen, cpu_vendor), dma_time("pcie-under", data_size, 15, pci_gen, cpu_vendor)]
b_dma_acc = [dma_time("acc", data_size, 1, pci_gen, cpu_vendor), dma_time("acc", data_size, 5, pci_gen, cpu_vendor), dma_time("acc", data_size, 10, pci_gen, cpu_vendor), dma_time("acc", data_size, 15, pci_gen, cpu_vendor)]

b_dmx_cpu  = [dmx_time, dmx_time, dmx_time*10/8, dmx_time*15/8]
b_dmx_pcie_overprovisioned = [dmx_time, dmx_time, dmx_time, dmx_time]
#b_dmx_pcie_underprovisioned = [1.25*dmx_time, 1.25*dmx_time, 1.25*dmx_time, 1.25*dmx_time]  
b_dmx_acc  = [dmx_time, dmx_time, dmx_time, dmx_time]

dmx_exec = [b_dmx_cpu, b_dmx_pcie_overprovisioned, b_dmx_acc]
dmx_exec = np.array(dmx_exec)
dmx_exec = dmx_exec.flatten()

data_movement = [b_dma_cpu, b_dma_pcie_overprovisioned, b_dma_acc]
data_movement = np.array(data_movement)
data_movement = data_movement.flatten()

data_movement = data_movement + control_pool_overhead   


fig_title = f'{benchmark_name}: DMX configs (cpu,pcie,acc),' + f"PCIe {pci_gen} with CPU vendor {cpu_vendor}\n" + f"pcie+ for overprovisioned DRX" #, pcie- for underprovisioned DRX"


labels = [f'cpu 1k', f'cpu 5k', f'cpu 10k' , f'cpu 15k', 
          f'pcie+1k', f'pcie+5k', f'pcie+10k' , f'pcie+15k',
          #f'pcie-1k', f'pcie-5k', f'pcie-10k' , f'pcie-15k',  
          f'acc 1k', f'acc 5k', f'acc 10k' , f'acc 15k']

#data_movement = data_movement*2 # rx + tx
total = dmx_exec + acc_kernel + data_movement
data_motion = dmx_exec #+ data_movement
#print(f"data motion: {dmx_exec}")
data_motion_ratio = data_motion/total
data_movement_ratio = data_movement/total
#print(f"data_movement:{data_movement}")
acc_kernel_ratio  = acc_kernel/total
#print(f"data_movement_ratio:{data_movement_ratio}")
print(f"{benchmark_name}:")
print("e2e total running time:")
print(f"cpu config:{total[0:4]}")
print(f"pcie config:{total[4:8]}")
print(f"acc config:{total[8:12]}")

print("\nacc_kernel_ratio:")
print(f"cpu config:{acc_kernel_ratio[0:4]}")
print(f"pcie config:{acc_kernel_ratio[4:8]}")
print(f"acc config:{acc_kernel_ratio[8:12]}")

print("\ndma_ratio:")
print(f"cpu config:{data_movement_ratio[0:4]}")
print(f"pcie config:{data_movement_ratio[4:8]}")
print(f"acc config:{data_movement_ratio[8:12]}")

fig, ax = plt.subplots(figsize=(15,5))
if mode == "latency":
    ax.bar(labels, acc_kernel,width=0.5, label='kernel')
    bottom = acc_kernel
    ax.bar(labels, data_motion, width=0.5, bottom=bottom, label='data restructuring')
    bottom = acc_kernel + data_motion
    p = ax.bar(labels, data_movement, width=0.5, bottom=bottom, label='data movement')
    #data_movement = np.round(data_movement,2)
    ax.bar_label(p, fmt='%.2f', label_type='edge')

    ax.legend(loc='best')
    ax.set_ylabel('Latency (ms)')
    title = f"latency_{benchmark_name}_acc_dmx_configs_{cpu_vendor}_{pci_gen}.png"

elif mode == "breakdown":
    ax.bar(labels, acc_kernel_ratio, width=0.5, label='kernel')
    bottom = acc_kernel_ratio
    ax.bar(labels, data_motion_ratio, width=0.5, bottom=bottom, label='data restructurin')
    bottom = acc_kernel_ratio + data_motion_ratio
    ax.bar(labels, data_movement_ratio, width=0.5, bottom=bottom, label='data movement')
    ax.legend(loc='best')
    ax.set_ylabel('Percentage (%)')
    title = f'breakdown_{benchmark_name}_acc_dmx_configs_{cpu_vendor}_{pci_gen}.png'


ax.grid(True)
#ax.set_ylabel('Latency (ms)')
#plt.show()
ax.set_title(fig_title)
plt.savefig(title, format='png', dpi=200)

#fig, ax = plt.subplots(figsize=(5,5))
