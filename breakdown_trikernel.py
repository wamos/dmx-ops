import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gmean
import sys
from pcie_dma_model import dma_time

b4_shape = (128,768,16)
b4_data_size = b4_shape[0] * b4_shape[1] * b4_shape[2] * 4 # 4-byte float

acc_b4_k1 = 1.138*2
acc_b4_k2 = 7.761
# Each input  has 16 sentences with 128-token, and each token uses a 768 dimension embedding  
acc_b4_k3 = 8.342*16 
cpu_b4 = 22.421 + 65.601
cpu_b4_k3 = 41.7*16

#an end-to-end overhead of 70ns for CXL reads, compared to NUMA-local DRAM reads
control_pool_overhead = 70*0.001*0.001 # ns to ms

benchmark_name = sys.argv[1]
mode = "latency"
mode = sys.argv[2]

# pci_gen = sys.argv[2]
# cpu_vendor = sys.argv[3]
pci_gen = "gen4"
cpu_vendor = "intel"
dmx_placement = "cpu-only"

b4_dma_1k = dma_time(dmx_placement, b4_data_size, 1, pci_gen, cpu_vendor)
b4_dma_5k = dma_time(dmx_placement, b4_data_size, 5, pci_gen, cpu_vendor)
b4_dma_10k = dma_time(dmx_placement, b4_data_size, 10, pci_gen, cpu_vendor)
b4_dma_15k = dma_time(dmx_placement, b4_data_size, 15, pci_gen, cpu_vendor)


acc_b4_kernels = acc_b4_k1 + acc_b4_k2 + acc_b4_k3
cpu_b4_kernels = cpu_b4 + cpu_b4_k3


kernels = np.ones(12)
third_kernel = np.ones(12)

fig_title = f'{benchmark_name}: 4, 8, and 16 cores with proportional LLC and memory bw\n' + "c for # of cores, k for # of kernels, " + f"PCIe {pci_gen} with CPU vendor {cpu_vendor}"


labels = [f'4c-1k', f'4c-5k', f'4c-10k' , f'4c-15k', 
          f'8c-1k', f'8c-5k', f'8c-10k' , f'8c-15k',  
          f'16c-1k', f'16c-5k', f'16c-10k' , f'16c-15k']
if benchmark_name == "b4-tri-kernel-cpu":
    pair1_e2e = np.array([76.118, 76.981, 129.35, 197.469,
                    71.135, 72.709, 74.250, 104.201,
                    70.431, 71.660, 73.014, 93.102])
    pair2_e2e = np.array([50.099, 238.434, 472.579, 715.952,
                          33.045, 180.246, 268.458, 407.501,
                          28.451, 162.743, 237.862, 370.161])
    e2e = pair1_e2e + pair2_e2e
    kernels.fill(cpu_b4_kernels)
    third_kernel.fill(cpu_b4_k3)

    data_movement = np.zeros(12)  
    total = e2e + third_kernel
    data_motion = total - kernels
    data_motion_ratio = data_motion/total
    kernel_ratio  = kernels/total
    print(f"{benchmark_name}:")
    print("e2e total running time:")
    print(f"4 cores:{total[0:4]}")
    print(f"8 cores:{total[4:8]}")
    print(f"16 cores:{total[8:12]}")

    print("breakdown cpu kernel ratio:")
    print(f"4 cores:{kernel_ratio[0:4]}")
    print(f"8 cores:{kernel_ratio[4:8]}")
    print(f"16 cores:{kernel_ratio[8:12]}")

    #exit()
elif benchmark_name == "b4-tri-kernel-acc":
    pair1_e2e = np.array([72.231, 74.201, 128.21, 192.523,
                    70.237, 70.510, 73.219, 101.879,
                    69.311, 70.623, 72.786, 91.988])
    pair2_e2e = np.array([88.140, 246.539, 494.055, 752.130,
                          77.617, 181.791, 277.315, 424.391,
                          75.495, 172.228, 248.221, 398.323])
    e2e = pair1_e2e + pair2_e2e
    kernels.fill(acc_b4_kernels)
    third_kernel.fill(acc_b4_k3)
    
    data_movement = [b4_dma_1k, b4_dma_5k, b4_dma_10k, b4_dma_15k] * 3
    data_movement = np.array(data_movement)
    data_movement = data_movement + control_pool_overhead
    total = e2e + third_kernel + data_movement
    data_motion = total - kernels - data_movement
    print(f"data motion: {data_motion}")
    data_motion_ratio = data_motion/total
    data_movement_ratio = data_movement/total
    kernel_ratio  = kernels/total
    #print(data_movement_ratio)
    print(f"data_movement: {data_movement}")

    print(f"{benchmark_name}-e2e total running time:")
    print(f"4 cores:{total[0:4]}")
    print(f"8 cores:{total[4:8]}")
    print(f"16 cores:{total[8:12]}")

    print(f"{benchmark_name}-acc-kernel-ratio:")
    print(f"4 cores:{kernel_ratio[0:4]}")
    print(f"8 cores:{kernel_ratio[4:8]}")
    print(f"16 cores:{kernel_ratio[8:12]}")

    print(f"{benchmark_name}-dma-ratio:")
    print(f"4 cores:{data_movement_ratio[0:4]}")
    print(f"8 cores:{data_movement_ratio[4:8]}")
    print(f"16 cores:{data_movement_ratio[8:12]}")

elif benchmark_name == "b4-tri-kernel-acc-drx":
    labels = [f'cpu 1k', f'cpu 5k', f'cpu 10k' , f'cpu 15k', 
          f'pcie+1k', f'pcie+5k', f'pcie+10k' , f'pcie+15k',
          f'acc 1k', f'acc 5k', f'acc 10k' , f'acc 15k']
    
    kernels.fill(acc_b4_kernels)

    b4_dmx1 = 2.097 # dmx between kernel-1 and kernel-2
    b4_dmx2 = 1.083 # dmx between kernel-2 and kernel-3
    data_size = b4_data_size # kernel-1-DMX-kernel2 + kernel2-DMX-kernel3
    dmx_time = b4_dmx1 + b4_dmx2
    b_dma_cpu = [dma_time("cpu", data_size, 1, pci_gen, cpu_vendor)*3/2, dma_time("cpu", data_size, 5, pci_gen, cpu_vendor)*3/2, dma_time("cpu", data_size, 10, pci_gen, cpu_vendor)*3/2, dma_time("cpu", data_size, 15, pci_gen, cpu_vendor)*3/2]
    b_dma_pcie_overprovisioned = [dma_time("pcie-tri-kernel", data_size, 1, pci_gen, cpu_vendor), dma_time("pcie-tri-kernel", data_size, 5, pci_gen, cpu_vendor), dma_time("pcie-tri-kernel", data_size, 10, pci_gen, cpu_vendor), dma_time("pcie-tri-kernel", data_size, 15, pci_gen, cpu_vendor)]
    b_dma_acc = [dma_time("acc-tri-kernel", data_size, 1, pci_gen, cpu_vendor), dma_time("acc-tri-kernel", data_size, 5, pci_gen, cpu_vendor), dma_time("acc-tri-kernel", data_size, 10, pci_gen, cpu_vendor), dma_time("acc-tri-kernel", data_size, 15, pci_gen, cpu_vendor)]

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
    total = dmx_exec + kernels + data_movement
    data_motion = dmx_exec #+ data_movement
    data_motion_ratio = data_motion/total
    data_movement_ratio = data_movement/total
    acc_kernel_ratio  = kernels/total

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
    
    fig_title = f'{benchmark_name}: DMX configs (cpu,pcie,acc),' + f"PCIe {pci_gen} with CPU vendor {cpu_vendor}\n" + f"pcie+ for overprovisioned DRX" #, pcie- for underprovisioned DRX"
else:
    print(f"invalid benchmark name {benchmark_name}")


fig, ax = plt.subplots(figsize=(15,5))

if mode == "latency":
    ax.bar(labels, kernels, width=0.5, label='kernel')
    bottom = kernels
    if benchmark_name == "b4-tri-kernel-cpu":
        p = ax.bar(labels, data_motion, width=0.5, bottom=bottom, label='data restructuring')
        ax.bar_label(p, fmt='%.2f', label_type='edge')
        title = f"latency_stacks_{benchmark_name}_cpu.png"

    else:
        ax.bar(labels, data_motion, width=0.5, bottom=bottom, label='data restructuring')
        bottom = kernels + data_motion
        p = ax.bar(labels, data_movement, width=0.5, bottom=bottom, label='data movement')
        ax.bar_label(p, fmt='%.2f', label_type='edge')
        if benchmark_name == "b4-tri-kernel-acc":
            title = f"latency_stacks_{benchmark_name}_acc_cpu.png"
        else:
            title = f"latency_stacks_{benchmark_name}_acc_dmx_configs_{cpu_vendor}_{pci_gen}.png"          
    
    ax.legend(loc='best')
    ax.set_ylabel('Latency (ms)')

elif mode == "breakdown":
    ax.bar(labels, kernel_ratio, width=0.5, label='kernel')
    bottom = kernel_ratio
    ax.bar(labels, data_motion_ratio, width=0.5, bottom=bottom, label='data restructuring')

    if benchmark_name == "b4-tri-kernel-cpu":        
        title = f"breakdown_{benchmark_name}_cpu.png"
    else: 
        bottom = kernel_ratio + data_motion_ratio
        ax.bar(labels, data_movement_ratio, width=0.5, bottom=bottom, label='data movement')
        if benchmark_name == "b4-tri-kernel-acc":
            title = f"breakdown_{benchmark_name}_acc_cpu.png"
        else:
            title = f'breakdown_{benchmark_name}_acc_dmx_configs_{cpu_vendor}_{pci_gen}.png'
        
    ax.legend(loc='best')
    ax.set_ylabel('Percentage (%)')

ax.grid(True)
ax.set_title(fig_title)
plt.savefig(title, format='png', dpi=200)
