import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gmean
import sys
from pcie_dma_model import pcie_power

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
mode = sys.argv[2]

b1_dmx = 4.607
b2_dmx = 1.491
b3_dmx = 1.383
b4_dmx = 2.097
b5_dmx = 16.004

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

# acc_kernel = np.ones(12)
# #second_kernel = np.ones(12)
# dmx_exec = np.ones(12)

pci_gen = "gen4"
cpu_vendor = "intel"

if benchmark_name == "benchmark2":
    data_size = b2_data_size
    dmx_time = b2_dmx
elif benchmark_name == "benchmark3":
    data_size = b3_data_size
    dmx_time = b3_dmx
elif benchmark_name == "benchmark1":
    data_size = b1_data_size
    dmx_time = b1_dmx
elif benchmark_name == "benchmark4":
    data_size = b4_data_size
    dmx_time = b4_dmx
elif benchmark_name == "benchmark5":
    data_size = b5_data_size
    dmx_time = b5_dmx
else:
    print(f"invalid benchmark name {benchmark_name}")
    exit()

# pcie_energy = data movement PCIe per bit power*time + data movement PCIe switch power*time
pcie_power_cpu = [pcie_power("cpu", data_size, 1, pci_gen, cpu_vendor), pcie_power("cpu", data_size, 5, pci_gen, cpu_vendor), pcie_power("cpu", data_size, 10, pci_gen, cpu_vendor), pcie_power("cpu", data_size, 15, pci_gen, cpu_vendor)]
pcie_power_pcie_overprovisioned = [pcie_power("pcie", data_size, 1, pci_gen, cpu_vendor), pcie_power("pcie", data_size, 5, pci_gen, cpu_vendor), pcie_power("pcie", data_size, 10, pci_gen, cpu_vendor), pcie_power("pcie", data_size, 15, pci_gen, cpu_vendor)]
pcie_power_pcie_under= [pcie_power("pcie-under", data_size, 1, pci_gen, cpu_vendor), pcie_power("pcie-under", data_size, 5, pci_gen, cpu_vendor), pcie_power("pcie-under", data_size, 10, pci_gen, cpu_vendor), pcie_power("pcie-under", data_size, 15, pci_gen, cpu_vendor)]
pcie_power_acc = [pcie_power("acc", data_size, 1, pci_gen, cpu_vendor), pcie_power("acc", data_size, 5, pci_gen, cpu_vendor), pcie_power("acc", data_size, 10, pci_gen, cpu_vendor), pcie_power("acc", data_size, 15, pci_gen, cpu_vendor)]
pcie_power_acc_nosw = [pcie_power("acc-nosw", data_size, 1, pci_gen, cpu_vendor), pcie_power("acc-nosw", data_size, 5, pci_gen, cpu_vendor), pcie_power("acc-nosw", data_size, 10, pci_gen, cpu_vendor), pcie_power("acc-nosw", data_size, 15, pci_gen, cpu_vendor)]

#pcie_energy_total = [pcie_energy_cpu, pcie_energy_pcie_overprovisioned, pcie_energy_acc, pcie_energy_pcie_under]
pcie_power_total = [pcie_power_cpu, pcie_power_pcie_overprovisioned, pcie_power_acc, pcie_power_acc_nosw]
pcie_power_total = np.array(pcie_power_total)
pcie_power_total = pcie_power_total.flatten()

drx_asic_power = 4.1
# in mJ
drx_power_cpu  = np.array([8*drx_asic_power, 8*drx_asic_power, 8*drx_asic_power, 8*drx_asic_power])

drx_power_pcie_overprovisioned = [4*drx_asic_power, 8*drx_asic_power, 12*drx_asic_power, 16*drx_asic_power]
drx_power_pcie_under = [4*drx_asic_power, 4*drx_asic_power, 8*drx_asic_power, 12*drx_asic_power]

drx_power_acc  = [drx_asic_power, 5*drx_asic_power, 10*drx_asic_power, 15*drx_asic_power]
drx_power_acc_nosw = drx_power_acc

#drx_energy_total = [drx_energy_cpu, drx_energy_pcie_overprovisioned, drx_energy_acc, drx_energy_pcie_under]
drx_power_total = [drx_power_cpu, drx_power_pcie_overprovisioned, drx_power_acc, drx_power_acc_nosw]
drx_power_total = np.array(drx_power_total)
drx_power_total = drx_power_total.flatten()

dmx_energy_total = drx_power_total + pcie_power_total

fig_title = f'{benchmark_name}: DMX configs (cpu,pcie,acc),' + f"PCIe {pci_gen} with CPU vendor {cpu_vendor}\n" + f"pcie+ for overprovisioned DRX, pcie- for underprovisioned DRX, acc- means without internal pcie switch BITW"


# pcie- means underprovision
# acc- means the accelerators don't have an internal PCIe switch
labels = [f'cpu 1k', f'cpu 5k', f'cpu 10k' , f'cpu 15k', 
          f'pcie+1k', f'pcie+5k', f'pcie+10k' , f'pcie+15k',          
          f'acc 1k', f'acc 5k', f'acc 10k' , f'acc 15k',
          #f'pcie-1k', f'pcie-5k', f'pcie-10k' , f'pcie-15k',
          f'acc- 1k', f'acc- 5k', f'acc- 10k' , f'acc- 15k']


#print("pcie_energy_total")
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
print(f"pcie_power_total:{pcie_power_total}")
print(f"drx_power_total:{drx_power_total}")

drx_ratio = drx_power_total/dmx_energy_total
data_movement_ratio = pcie_power_total/dmx_energy_total

fig, ax = plt.subplots(figsize=(15,5))
if mode == "power":
    ax.bar(labels, drx_power_total, width=0.5, label='data restructuring')
    bottom = drx_power_total

    p = ax.bar(labels, pcie_power_total, width=0.5, bottom=bottom, label='data movement')
    ax.bar_label(p, fmt='%.3f', label_type='edge')

    ax.legend(loc='best')
    ax.set_ylabel('Power (Watts)')
    title = f"power_{benchmark_name}_acc_dmx_configs_{cpu_vendor}_{pci_gen}.png"

elif mode == "breakdown":
    ax.bar(labels, drx_ratio, width=0.5, label='data restructuring')
    bottom = drx_ratio
    ax.bar(labels, data_movement_ratio, width=0.5, bottom=bottom, label='data movement')

    ax.legend(loc='best')
    ax.set_ylabel('Percentage (%)')
    title = f'power_breakdown_{benchmark_name}_acc_dmx_configs_{cpu_vendor}_{pci_gen}.png'


ax.grid(True)
#ax.set_ylabel('Latency (ms)')
#plt.show()
ax.set_title(fig_title)
plt.savefig(title, format='png', dpi=200)

#fig, ax = plt.subplots(figsize=(5,5))
