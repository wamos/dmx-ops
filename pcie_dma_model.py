import numpy as np
#TODO energy modeling of each design 

def pcie_switching_delay(data_size:int, pcie_gen: str) -> float:
    pcie_mps = 256
    mem_wrtie_hdr = 24
    #data_size = np.ceil(data_size/pcie_mps) * mem_wrtie_hdr + data_size
    num_pkts = np.ceil(data_size/pcie_mps)
    delay = 0
    if pcie_gen == "gen3":
        delay = num_pkts * 150 #ns
    elif pcie_gen == "gen4":
        delay = num_pkts * 105 #ns
    elif pcie_gen == "gen5":
        delay = num_pkts * 115 #ns
    else:
        print("PCIe gen not supported")

    delay = delay/1e6 # ns to ms
    return delay

def dma_time(dmx_placment: str,data_size:int, num_kernel: int, pcie_gen: str, cpu_vendor: str) -> float:
    time = 0
    if dmx_placment == "cpu-only":
        time = dma_time_dmx_on_cpu(data_size, num_kernel, pcie_gen, cpu_vendor)
    elif dmx_placment == "cpu":
        time = dma_time_dmx_on_cpu(data_size, num_kernel, pcie_gen, cpu_vendor)
    elif dmx_placment == "pcie":
        time = dma_time_dmx_on_pcie(data_size, num_kernel, pcie_gen, cpu_vendor)
    elif dmx_placment == "acc":
        time = dma_time_dmx_on_acc(data_size, num_kernel, pcie_gen, cpu_vendor)
    else:
        print("unsuppoted DMX placement")
    return time

def dma_time_dmx_on_acc(data_size:int, num_kernel: int, pcie_gen: str, cpu_vendor: str) -> float:
    pcie_mps = 256 # max_payload_size of PCIe we assume, which is very common
    mem_wrtie_hdr = 24 # 2B framing, 6B DLL header, 4B TLP header, and 12B MWr header
    raw_size = data_size
    data_size = np.ceil(data_size/pcie_mps) * mem_wrtie_hdr + data_size
    #print(f"total size {data_size}, raw size {raw_size}")

    time = 0
    # x16 has 15.754 GB/s
    if pcie_gen == "gen3" and cpu_vendor == "amd": 
        acc_pcie_lane = 16
        acc_pcie_lane = acc_pcie_lane * 2 # a pair of accelerator
        cpu_pcie_lane = 128
        full_rate = 15.754 * 1e9 # x16 rate
        # for the case of 5, 10, 15 kernels
        if num_kernel == 1 or num_kernel == 5:
            rate = full_rate
            time = data_size/rate
            time = time * 1000 # in ms               
        elif num_kernel == 10 or num_kernel == 15:
        # 10*16=160 lanes, 160/3 = (64, 48, 48)
            rate = full_rate*0.93
            time = data_size/rate
            time = time * 1000 # in ms
            time = time + pcie_switching_delay(raw_size,pcie_gen)
        elif num_kernel == 15:
        # 15*16=240 lanes, 240/3 = (80, 80, 80)
            rate = full_rate*0.78
            time = data_size/rate
            time = time * 1000 # in ms
            time = time + pcie_switching_delay(raw_size,pcie_gen)
        else:
            print("un-supported number of kernels")

    elif pcie_gen == "gen3" and cpu_vendor == "intel": 
        acc_pcie_lane = 16
        acc_pcie_lane = acc_pcie_lane * 2 # a pair of accelerator
        cpu_pcie_lane = 48
        full_rate = 15.754 * 1e9 # x16 rate
        # for the case of 5, 10, 15 kernels
        if num_kernel == 1:
            rate = full_rate
            time = data_size/rate
            time = time * 1000 # in ms
        elif num_kernel == 5:
        # 5*16=80 lanes, 80/3 = (32, 32, 16)
            rate = full_rate*0.75
            time = data_size/rate
            time = time * 1000 # in ms
            time = time + 2*pcie_switching_delay(raw_size,pcie_gen)
        elif num_kernel == 10:
        # 10*16=160 lanes, 160/3 = (64, 48, 48)
            rate = full_rate*0.625
            time = data_size/rate
            time = time * 1000 # in ms
            time = time + 2*pcie_switching_delay(raw_size,pcie_gen)
        elif num_kernel == 15:
        # 15*16=240 lanes, 240/3 = (80, 80, 80)
            rate = full_rate*0.625
            time = data_size/rate
            time = time * 1000 # in ms
            time = time + 2*pcie_switching_delay(raw_size,pcie_gen)
        else:
            print("un-supported number of kernels")

    elif pcie_gen == "gen4" and cpu_vendor == "amd":
        acc_pcie_lane = 8
        acc_pcie_lane = acc_pcie_lane * 2 # a pair of accelerator
        cpu_pcie_lane = 64
        full_rate = 15.754 * 1e9 # x8 rate
        if num_kernel == 1 or num_kernel == 5:
        # 10*8=80 lanes, 80/4 = (24, 24, 16, 16)
            rate = full_rate
            time = data_size/rate
            time = time * 1000 # in ms
        elif num_kernel == 10 or num_kernel == 15:
        # 15*8=120 lanes, 120/4 = (32,32,32,24)
            rate = full_rate/2
            time = data_size/rate
            time = time * 1000 # in ms
            time = time + pcie_switching_delay(raw_size,pcie_gen)
        else:
            print("un-supported number of kernels")

    elif pcie_gen == "gen4" and cpu_vendor == "intel":
        acc_pcie_lane = 8
        acc_pcie_lane = acc_pcie_lane * 2 # a pair of accelerator
        cpu_pcie_lane = 128
        full_rate = 15.754 * 1e9 # x8 rate
        if num_kernel == 1:
            rate = full_rate
            time = data_size/rate
            time = time * 1000 # in ms
        elif num_kernel == 5 or num_kernel == 10:
            rate = full_rate
            time = data_size/rate
            time = time * 1000 # in ms
            time = time + pcie_switching_delay(raw_size,pcie_gen)
        elif num_kernel == 15:
            rate = full_rate*0.7825
            time = data_size/rate
            time = time * 1000 # in ms
            time = time + pcie_switching_delay(raw_size,pcie_gen)
        else:
            print("un-supported number of kernels")
    elif pcie_gen == "gen5" and cpu_vendor == "amd":
        acc_pcie_lane = 4
        cpu_pcie_lane = 80
        full_rate = 15.754 * 1e9 # x4 rate
        if num_kernel == 1 or num_kernel == 5 or num_kernel == 10 or num_kernel == 15:
        # 10*8=80 lanes, 80/4 = (24, 24, 16, 16)
            rate = full_rate
            time = data_size/rate
            time = time * 1000 # in ms
        else:
            print("un-supported number of kernels")    
    elif pcie_gen == "gen5" and cpu_vendor == "intel":
        acc_pcie_lane = 4
        acc_pcie_lane = acc_pcie_lane * 2 # a pair of accelerator
        cpu_pcie_lane = 128
        full_rate = 15.754 * 1e9 # x4 rate
        if num_kernel == 1 or num_kernel == 5 or num_kernel == 10:
        # 10*8=80 lanes, 80/4 = (24, 24, 16, 16)
            rate = full_rate
            time = data_size/rate
            time = time * 1000 # in ms
        elif num_kernel == 15:
            rate = full_rate
            time = data_size/rate
            time = time * 1000 # in ms
            time = time + pcie_switching_delay(raw_size,pcie_gen)
        else:
            print("un-supported number of kernels")
    else:
        print("un-supported")

    # if acc_pcie_lane * num_kernel > cpu_pcie_lane:
    #     time = time + pcie_switching_delay(raw_size,pcie_gen) # both in ms

    return time # in ms

def dma_time_dmx_on_pcie(data_size:int, num_kernel: int, pcie_gen: str, cpu_vendor: str) -> float:
    pcie_mps = 256 # max_payload_size of PCIe we assume, which is very common
    mem_wrtie_hdr = 24 # 2B framing, 6B DLL header, 4B TLP header, and 12B MWr header
    raw_size = data_size
    data_size = np.ceil(data_size/pcie_mps) * mem_wrtie_hdr + data_size
    #print(f"total size {data_size}, raw size {raw_size}")

    time = 0
    # x16 has 15.754 GB/s
    if pcie_gen == "gen3" and cpu_vendor == "amd": 
        acc_pcie_lane = 16
        acc_pcie_lane = acc_pcie_lane * 2 # a pair of accelerator
        cpu_pcie_lane = 128
        full_rate = 15.754 * 1e9 # x16 rate
        # for the case of 5, 10, 15 kernels
        if num_kernel == 1:
            rate = full_rate
            time = data_size/rate
            time = time * 1000 # in ms                
        elif num_kernel == 5:
        # 5*16=80 lanes, 80/3 = (32, 32, 16)
            rate = full_rate
            time = data_size/rate
            time = time * 1000 # in ms
            time = time + pcie_switching_delay(raw_size,pcie_gen)
        elif num_kernel == 10 or num_kernel == 15:
            rate = full_rate*0.875
            time = data_size/rate
            time = time * 1000 # in ms
            time = time + pcie_switching_delay(raw_size,pcie_gen)
        elif num_kernel == 15:
            rate = full_rate*0.7
            time = data_size/rate
            time = time * 1000 # in ms
            time = time + 2*pcie_switching_delay(raw_size,pcie_gen)
        else:
            print("un-supported number of kernels")

    elif pcie_gen == "gen3" and cpu_vendor == "intel": 
        acc_pcie_lane = 16
        acc_pcie_lane = acc_pcie_lane * 2 # a pair of accelerator
        cpu_pcie_lane = 48
        full_rate = 15.754 * 1e9 # x16 rate
        # for the case of 5, 10, 15 kernels
        if num_kernel == 1:
            rate = full_rate
            time = data_size/rate
            time = time * 1000 # in ms
        elif num_kernel == 5:
        # 5*16=80 lanes, 80/3 = (32, 32, 16)
            rate = full_rate*0.75
            time = data_size/rate
            time = time * 1000 # in ms
            time = time + 2*pcie_switching_delay(raw_size,pcie_gen)
        elif num_kernel == 10:
        # 10*16=160 lanes, 160/3 = (64, 48, 48)
            rate = full_rate*0.625
            time = data_size/rate
            time = time * 1000 # in ms
            time = time + 2*pcie_switching_delay(raw_size,pcie_gen)
        elif num_kernel == 15:
        # 15*16=240 lanes, 240/3 = (80, 80, 80)
            rate = full_rate*0.6
            time = data_size/rate
            time = time * 1000 # in ms
            time = time + 2*pcie_switching_delay(raw_size,pcie_gen)
        else:
            print("un-supported number of kernels")

    elif pcie_gen == "gen4" and cpu_vendor == "amd":
        acc_pcie_lane = 8
        acc_pcie_lane = acc_pcie_lane * 2 # a pair of accelerator
        cpu_pcie_lane = 128
        full_rate = 15.754 * 1e9 # x8 rate
        if num_kernel == 1:
            rate = full_rate
            time = data_size/rate
            time = time * 1000 # in ms
        elif num_kernel == 5 or num_kernel == 10 or num_kernel == 15:
            rate = full_rate
            time = data_size/rate
            time = time * 1000 # in ms
            time = time + pcie_switching_delay(raw_size,pcie_gen)
        else:
            print("un-supported number of kernels")

    elif pcie_gen == "gen4" and cpu_vendor == "intel":
        acc_pcie_lane = 8
        acc_pcie_lane = acc_pcie_lane * 2 # a pair of accelerator
        cpu_pcie_lane = 64
        full_rate = 15.754 * 1e9 # x8 rate
        if num_kernel == 1:
        # 10*8=80 lanes, 80/4 = (24, 24, 16, 16)
            rate = full_rate
            time = data_size/rate
            time = time * 1000 # in ms
        elif num_kernel == 5:
            rate = full_rate
            time = data_size/rate
            time = time * 1000 # in ms
            time = time + pcie_switching_delay(raw_size,pcie_gen)
        elif num_kernel == 10:
        # 15*8=120 lanes, 120/4 = (32,32,32,24)
            rate = full_rate*0.875
            time = data_size/rate
            time = time * 1000 # in ms
            time = time + pcie_switching_delay(raw_size,pcie_gen)
        elif num_kernel == 15:
        # 15*8=120 lanes, 120/4 = (32,32,32,24)
            rate = full_rate*0.75
            time = data_size/rate
            time = time * 1000 # in ms
            time = time + pcie_switching_delay(raw_size,pcie_gen)
        else:
            print("un-supported number of kernels")
    elif pcie_gen == "gen5" and cpu_vendor == "amd":
        acc_pcie_lane = 4
        cpu_pcie_lane = 80
        full_rate = 15.754 * 1e9 # x4 rate
        if num_kernel == 1 or num_kernel == 5 or num_kernel == 10: 
        # 10*8=80 lanes, 80/4 = (24, 24, 16, 16)
            rate = full_rate
            time = data_size/rate
            time = time * 1000 # in ms
        elif num_kernel == 15:
            rate = full_rate
            time = data_size/rate
            time = time * 1000 # in ms
            time = time + pcie_switching_delay(raw_size,pcie_gen)
        else:
            print("un-supported number of kernels")    
    elif pcie_gen == "gen5" and cpu_vendor == "intel":
        acc_pcie_lane = 4
        acc_pcie_lane = acc_pcie_lane * 2 # a pair of accelerator
        cpu_pcie_lane = 128
        full_rate = 15.754 * 1e9 # x4 rate
        if num_kernel == 1 or num_kernel == 5:
        # 10*8=80 lanes, 80/4 = (24, 24, 16, 16)
            rate = full_rate
            time = data_size/rate
            time = time * 1000 # in ms
        elif num_kernel == 10 or num_kernel == 15:
            rate = full_rate
            time = data_size/rate
            time = time * 1000 # in ms
            time = time + pcie_switching_delay(raw_size,pcie_gen)
        else:
            print("un-supported number of kernels")
    else:
        print("un-supported")

    # if acc_pcie_lane * num_kernel > cpu_pcie_lane:
    #     time = time + pcie_switching_delay(raw_size,pcie_gen) # both in ms

    return time # in ms

def dma_time_dmx_on_cpu(data_size:int, num_kernel: int, pcie_gen: str, cpu_vendor: str) -> float:
    pcie_mps = 256 # max_payload_size of PCIe we assume, which is very common
    mem_wrtie_hdr = 24 # 2B framing, 6B DLL header, 4B TLP header, and 12B MWr header
    raw_size = data_size
    data_size = np.ceil(data_size/pcie_mps) * mem_wrtie_hdr + data_size
    #print(f"total size {data_size}, raw size {raw_size}")

    time = 0
    # x16 has 15.754 GB/s
    if pcie_gen == "gen3" and cpu_vendor == "amd": 
        acc_pcie_lane = 16
        acc_pcie_lane = acc_pcie_lane * 2 # a pair of accelerator
        cpu_pcie_lane = 128
        full_rate = 15.754 * 1e9 # x16 rate
        # for the case of 5, 10, 15 kernels
        if num_kernel == 1:
            rate = full_rate
            time = data_size/rate
            time = time * 1000 # in ms                
        elif num_kernel == 5:
        # 5*16=80 lanes, 80/3 = (32, 32, 16)
            rate = full_rate
            time = data_size/rate
            time = time * 1000 # in ms
            time = time + pcie_switching_delay(raw_size,pcie_gen)
        elif num_kernel == 10 or num_kernel == 15:
        # 10*16=160 lanes, 160/3 = (64, 48, 48)
            rate = full_rate/2
            time = data_size/rate
            time = time * 1000 # in ms
            time = time + 2*pcie_switching_delay(raw_size,pcie_gen)
        # elif num_kernel == 15:
        # # 15*16=240 lanes, 240/3 = (80, 80, 80)
        #     rate = full_rate/2
        #     time = data_size/rate
        #     time = time * 1000 # in ms
        #     time = time + 2*pcie_switching_delay(raw_size,pcie_gen)
        else:
            print("un-supported number of kernels")

    elif pcie_gen == "gen3" and cpu_vendor == "intel": 
        acc_pcie_lane = 16
        acc_pcie_lane = acc_pcie_lane * 2 # a pair of accelerator
        cpu_pcie_lane = 48
        full_rate = 15.754 * 1e9 # x16 rate
        # for the case of 5, 10, 15 kernels
        if num_kernel == 1:
            rate = full_rate
            time = data_size/rate
            time = time * 1000 # in ms
        elif num_kernel == 5:
        # 5*16=80 lanes, 80/3 = (32, 32, 16)
            rate = full_rate/2
            time = data_size/rate
            time = time * 1000 # in ms
            time = time + pcie_switching_delay(raw_size,pcie_gen)
        elif num_kernel == 10:
        # 10*16=160 lanes, 160/3 = (64, 48, 48)
            rate = full_rate/4
            time = data_size/rate
            time = time * 1000 # in ms
            time = time + 2*pcie_switching_delay(raw_size,pcie_gen)
        elif num_kernel == 15:
        # 15*16=240 lanes, 240/3 = (80, 80, 80)
            rate = full_rate/5
            time = data_size/rate
            time = time * 1000 # in ms
            time = time + 2*pcie_switching_delay(raw_size,pcie_gen)
        else:
            print("un-supported number of kernels")

    elif pcie_gen == "gen4" and cpu_vendor == "amd":
        acc_pcie_lane = 8
        acc_pcie_lane = acc_pcie_lane * 2 # a pair of accelerator
        cpu_pcie_lane = 128
        full_rate = 15.754 * 1e9 # x8 rate
        if num_kernel == 1 or num_kernel == 5:
            rate = full_rate
            time = data_size/rate
            time = time * 1000 # in ms
        elif num_kernel == 10 or num_kernel == 15:
            rate = full_rate
            time = data_size/rate
            time = time * 1000 # in ms
            time = time + pcie_switching_delay(raw_size,pcie_gen)
        else:
            print("un-supported number of kernels")

    elif pcie_gen == "gen4" and cpu_vendor == "intel":
        acc_pcie_lane = 8
        acc_pcie_lane = acc_pcie_lane * 2 # a pair of accelerator
        cpu_pcie_lane = 64
        full_rate = 15.754 * 1e9 # x8 rate
        if num_kernel == 1:
        # 10*8=80 lanes, 80/4 = (24, 24, 16, 16)
            rate = full_rate
            time = data_size/rate
            time = time * 1000 # in ms
        elif num_kernel == 5:
            rate = full_rate
            time = data_size/rate
            time = time * 1000 # in ms
            time = time + pcie_switching_delay(raw_size,pcie_gen)
        elif num_kernel == 10 or num_kernel == 15:
        # 15*8=120 lanes, 120/4 = (32,32,32,24)
            rate = full_rate/2
            time = data_size/rate
            time = time * 1000 # in ms
            time = time + pcie_switching_delay(raw_size,pcie_gen)
        else:
            print("un-supported number of kernels")
    elif pcie_gen == "gen5" and cpu_vendor == "amd":
        acc_pcie_lane = 4
        cpu_pcie_lane = 80
        full_rate = 15.754 * 1e9 # x4 rate
        if num_kernel == 1 or num_kernel == 5 or num_kernel == 10 or num_kernel == 15:
        # 10*8=80 lanes, 80/4 = (24, 24, 16, 16)
            rate = full_rate
            time = data_size/rate
            time = time * 1000 # in ms
        else:
            print("un-supported number of kernels")    
    elif pcie_gen == "gen5" and cpu_vendor == "intel":
        acc_pcie_lane = 4
        acc_pcie_lane = acc_pcie_lane * 2 # a pair of accelerator
        cpu_pcie_lane = 128
        full_rate = 15.754 * 1e9 # x4 rate
        if num_kernel == 1 or num_kernel == 5 or num_kernel == 10:
        # 10*8=80 lanes, 80/4 = (24, 24, 16, 16)
            rate = full_rate
            time = data_size/rate
            time = time * 1000 # in ms
        elif num_kernel == 15:
            rate = full_rate
            time = data_size/rate
            time = time * 1000 # in ms
            time = time + pcie_switching_delay(raw_size,pcie_gen)
        else:
            print("un-supported number of kernels")
    else:
        print("un-supported")

    # if acc_pcie_lane * num_kernel > cpu_pcie_lane:
    #     time = time + pcie_switching_delay(raw_size,pcie_gen) # both in ms

    return time # in ms