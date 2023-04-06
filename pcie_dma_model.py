import numpy as np

def dma_transfer_time(data_size:int, num_kernel: int, pcie_gen: str, cpu_vendor: str) -> float:
    pcie_mps = 256 # max_payload_size of PCIe we assume, which is very common
    mem_wrtie_hdr = 24 # 2B framing, 6B DLL header, 4B TLP header, and 12B MWr header
    raw_size = data_size
    data_size = np.ceil(data_size/pcie_mps) * mem_wrtie_hdr + data_size
    #print(f"total size {data_size}, raw size {raw_size}")

    time = 0
    # 240 lanes for 15 kernels using 16 lanes each 
    # AMD has 128 lanes PCIe 3.0 on its Zen Gen1 CPU
    # We can have 8 upstream ports with x16, 15 downstream ports. 
    # 240/16 = 15. We’ll do it as 16 downstream ports with x16 in total. 
    # Each PCIe switch accommodates 2 accelerators and use 48-lane PCIe switches.
    #
    # x16 has 15.754 GB/s
    if pcie_gen == "gen3" and cpu_vendor == "amd": 
        acc_pcie_lane = 16
        cpu_pcie_lane = 128
        full_rate = 15.754 * 1e9 # x16 rate
        if acc_pcie_lane * num_kernel > cpu_pcie_lane:
        # we need to use PCIe switch with x16 upstream port
        # the exec time is determined by the slowest kernel. 10 kernels and 15 kernels 
        # are the same as they both need a layer of PCIe switches
        # for both of them 2 acceelerator per switch
        # time * rate = size, time = size/rate
            rate = full_rate/2
        else:
            rate = full_rate

    elif pcie_gen == "gen3" and cpu_vendor == "intel": 
        acc_pcie_lane = 16
        cpu_pcie_lane = 48
        full_rate = 15.754 * 1e9 # x16 rate
        if acc_pcie_lane * num_kernel > cpu_pcie_lane:
            # for the case of 5, 10, 15 kernels
            if num_kernel == 5:
            # 5*16=80 lanes, 80/3 = (32, 32, 16)
                rate = full_rate/2
            elif num_kernel == 10:
            # 10*16=160 lanes, 160/3 = (64, 48, 48)
                rate = full_rate/4
            elif num_kernel == 15:
            # 15*16=240 lanes, 240/3 = (80, 80, 80)
                rate = full_rate/5
            else:
                print("un-supported number of kernels")
        else:
            rate = full_rate

    elif pcie_gen == "gen4" and cpu_vendor == "amd":
        acc_pcie_lane = 8
        cpu_pcie_lane = 128
        full_rate = 15.754 * 1e9 # x8 rate
        if acc_pcie_lane * num_kernel > cpu_pcie_lane:
            print("no calculation now, unlikely to be here")
        else:
            rate = full_rate

    elif pcie_gen == "gen4" and cpu_vendor == "intel":
        acc_pcie_lane = 8
        cpu_pcie_lane = 64
        full_rate = 15.754 * 1e9 # x8 rate
        if acc_pcie_lane * num_kernel > cpu_pcie_lane:
            if num_kernel == 10:
            # 10*8=80 lanes, 80/4 = (24, 24, 16, 16)
                rate = full_rate/3
            elif num_kernel == 15:
            # 15*8=120 lanes, 120/4 = (32,32,32,24)
                rate = full_rate/4
            else:
                print("un-supported number of kernels")
        else:
            rate = full_rate
    elif pcie_gen == "gen5" and cpu_vendor == "amd":
        acc_pcie_lane = 4
        cpu_pcie_lane = 128
        full_rate = 15.754 * 1e9 # x4 rate
        if acc_pcie_lane * num_kernel > cpu_pcie_lane:
            print("no calculation now, unlikely to be here")          
        else:
            rate = full_rate
    elif pcie_gen == "gen5" and cpu_vendor == "intel":
        acc_pcie_lane = 4
        cpu_pcie_lane = 80
        full_rate = 15.754 * 1e9 # x4 rate
        if acc_pcie_lane * num_kernel > cpu_pcie_lane:
            print("no calculation now, unlikely to be here")         
        else:
            rate = full_rate
    else:
        print("un-supported")

    time = data_size/rate
    return time * 1000 # in ms