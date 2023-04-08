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

#TODO
#def dmx_compute_power()

#TODO
# def pcie_transfer_power()
# PCIe:11 pJ/b

def pcie_switching_power(dmx_placment: str,data_size:int, num_kernel: int, pcie_gen: str, cpu_vendor: str) -> float:
    acc_axi_delay = 57*0.001*0.001 # 57 ns
    time = 0
    if dmx_placment == "cpu-only":
        time = pcie_switching_power_dmx_on_cpu(data_size, num_kernel, pcie_gen, cpu_vendor)*4
    elif dmx_placment == "cpu":
        time = pcie_switching_power_dmx_on_cpu(data_size, num_kernel, pcie_gen, cpu_vendor)*4
    elif dmx_placment == "pcie":
        time = pcie_switching_power_dmx_on_pcie(data_size, num_kernel, pcie_gen, cpu_vendor)*4
    elif dmx_placment == "acc":
        time = pcie_switching_power_dmx_on_acc(data_size, num_kernel, pcie_gen, cpu_vendor)*3 + acc_axi_delay*2
    else:
        print("unsuppoted DMX placement")
    return time

# TODO: pseudo code to determine the topology with least bandwidth oversubscription on PCIe switch upstream links
# acc_pcie_lane = 16
# total_acc_pcie_lane = acc_pcie_lane*num_kernel
# total_acc_ports = 2*num_kernel
# cpu_pcie_lane = 128
# num_cpu_upstream_ports = cpu_pcie_lane/acc_pcie_lane
# max_pcie_switch_downstram_ports = (max_pcie_switch_lanes - acc_pcie_lane)/acc_pcie_lane
# num_used_ports = -1
# for first_layer_num_downstream_ports in range(max_pcie_switch_downstram_ports):
#   if num_cpu_upstream_ports*num_downstream_ports > total_acc_ports:
#        num_used_ports = num_downstream_ports
#        lookup the energy based on num_used_ports*acc_pcie_lane in a predefined table   
#
# if num_used_ports == -1: # we need another layer
# for second_level_num_downstream_ports in range(2,max_pcie_switch_downstram_ports):
#   for first_layer_num_downstream_ports in range(2, max_pcie_switch_downstram_ports) 
#       if num_cpu_upstream_ports*first_layer_num_downstream_ports*second_level_num_downstream_ports > total_acc_ports
#           #break both for loop as we found the answer already.

def pcie_switching_power_dmx_on_acc(data_size:int, num_kernel: int, pcie_gen: str, cpu_vendor: str) -> float:
    pcie_mps = 256 # max_payload_size of PCIe we assume, which is very common
    mem_wrtie_hdr = 24 # 2B framing, 6B DLL header, 4B TLP header, and 12B MWr header
    raw_size = data_size
    data_size = np.ceil(data_size/pcie_mps) * mem_wrtie_hdr + data_size
    #print(f"total size {data_size}, raw size {raw_size}")

    power = 0
    # x16 has 15.754 GB/s
    if pcie_gen == "gen3" and cpu_vendor == "amd": 
        acc_pcie_lane = 16
        #acc_pcie_lane = acc_pcie_lane * 2 # a pair of accelerator
        cpu_pcie_lane = 128
        full_rate = 15.754 * 1e9 # x16 rate
        upstream_ports = int(cpu_pcie_lane/acc_pcie_lane)

        if num_kernel == 1 and 5:
            power = 0
        elif num_kernel == 5:
            power = num_kernel * 14.5 # use 49-lane, 14.5 watts, lanes = 16*2 (2 downstream) + 16 (upstream) = 48
        elif num_kernel == 10:
            power = upstream_ports * 16.2 # use 65-lane, 16.2  watts, lanes = 16*3 (3 downstream) + 16 (upstream) = 64
        elif num_kernel == 15:
            power = upstream_ports * 22.5 # use 81-lane, 22.5 watts, lanes = 16*4 (4 downstream) + 16 (upstream) = 80
        else:
            print("un-supported number of kernels")

    elif pcie_gen == "gen3" and cpu_vendor == "intel": 
        acc_pcie_lane = 16
        #acc_pcie_lane = acc_pcie_lane * 2 # a pair of accelerator
        cpu_pcie_lane = 48
        upstream_ports = int(cpu_pcie_lane/acc_pcie_lane)

        full_rate = 15.754 * 1e9 # x16 rate
        # for the case of 5, 10, 15 kernels
        if num_kernel == 1:
            power = 0
        elif num_kernel == 5:
            first_layer_power = upstream_ports * 22.5 # use 81-lane, 22.5 watts, lanes = 16*4 (4 downstream) + 16 (upstream) = 80
        elif num_kernel == 10: 
            first_layer_power = upstream_ports * 22.5 # use 81-lane, lanes = 16*4 (4 downstream) + 16 (upstream) = 80
            second_layer_power = upstream_ports * 4 * 14.5 # 4 first-layer PCIe switch, use 49-lane, lanes = 16*2 (2 downstream) + 16 (upstream) = 48
            power = first_layer_power + second_layer_power
        elif num_kernel == 15:
            first_layer_power = upstream_ports * 24.3  # use 97-lane, lanes = 16*5 (4 downstream) + 16 (upstream) = 80
            second_layer_power = upstream_ports * 5 * 14.5 # 5 first-layer PCIe switch, use 49-lane, lanes = 16*2 (2 downstream) + 16 (upstream) = 48
            power = first_layer_power + second_layer_power
        else:
            print("un-supported number of kernels")

    elif pcie_gen == "gen4" and cpu_vendor == "amd":
        acc_pcie_lane = 8
        #acc_pcie_lane = acc_pcie_lane * 2 # a pair of accelerator
        cpu_pcie_lane = 128
        upstream_ports = int(cpu_pcie_lane/acc_pcie_lane)

        if num_kernel == 1 or num_kernel == 5:
            power = 0
        elif num_kernel == 10: 
            power = num_kernel*11.44
            #use 26-lane, lanes = 8*2 (2 downstream) + 8 (upstream) = 24
        elif num_kernel == 15:
            power = num_kernel*11.44
            #use 26-lane, lanes = 8*2 (2 downstream) + 8 (upstream) = 24
        else:
            print("un-supported number of kernels")

    elif pcie_gen == "gen4" and cpu_vendor == "intel":
        acc_pcie_lane = 8
        cpu_pcie_lane = 64
        upstream_ports = int(cpu_pcie_lane/acc_pcie_lane)

        if num_kernel == 1:
            power = 0
        elif num_kernel == 5:
            power = upstream_ports * 11.44 # use 26-lane, lanes = 8*2 (2 downstream) + 8 (upstream) = 24
        elif num_kernel == 10:
            power = upstream_ports * 13.18 # use 34-lane, lanes = 8*3 (2 downstream) + 8 (upstream) = 32
        elif num_kernel == 15:
            power = upstream_ports * 18.81 # use 50-lane, lanes = 8*4 (2 downstream) + 8 (upstream) = 40
        else:
            print("un-supported number of kernels")

    elif pcie_gen == "gen5" and cpu_vendor == "amd":
        acc_pcie_lane = 4
        cpu_pcie_lane = 128
        upstream_ports = int(cpu_pcie_lane/acc_pcie_lane)
        if num_kernel == 1 or num_kernel == 5 or num_kernel == 10 or num_kernel == 15:
            power = 0
        else:
            print("un-supported number of kernels")    
    elif pcie_gen == "gen5" and cpu_vendor == "intel":
        acc_pcie_lane = 4
        cpu_pcie_lane = 80
        upstream_ports = int(cpu_pcie_lane/acc_pcie_lane)
        if num_kernel == 1 or num_kernel == 5 or num_kernel == 10:
            power = 0
        elif num_kernel == 15:
            power = (num_kernel*2 - upstream_ports)*14.0 # use 4*3 + 4 = 16 but we the min is 24 lanes
        else:
            print("un-supported number of kernels")
    else:
        print("un-supported")


    return power # in watts


def pcie_switching_power_dmx_on_pcie(data_size:int, num_kernel: int, pcie_gen: str, cpu_vendor: str) -> float:
    pcie_mps = 256 # max_payload_size of PCIe we assume, which is very common
    mem_wrtie_hdr = 24 # 2B framing, 6B DLL header, 4B TLP header, and 12B MWr header
    raw_size = data_size
    data_size = np.ceil(data_size/pcie_mps) * mem_wrtie_hdr + data_size
    #print(f"total size {data_size}, raw size {raw_size}")

    power = 0
    # x16 has 15.754 GB/s
    if pcie_gen == "gen3" and cpu_vendor == "amd": 
        acc_pcie_lane = 16
        #acc_pcie_lane = acc_pcie_lane * 2 # a pair of accelerator
        cpu_pcie_lane = 128
        full_rate = 15.754 * 1e9 # x16 rate
        upstream_ports = int(cpu_pcie_lane/acc_pcie_lane)

        if num_kernel == 1:
            power = 0
        elif num_kernel == 5:
            power = upstream_ports * 16.2 # use 65-lane, 16.2 watts, lanes = 16*3 (2 downstream) + 16 (upstream) = 64
        elif num_kernel == 10:
            power = upstream_ports * 22.5 # use 81-lane, 22.5 watts, lanes = 16*4 (3 downstream) + 16 (upstream) = 80
        elif num_kernel == 15:
            power = upstream_ports * 24.3 # use 97-lane, 24.3 watts, lanes = 16*5 (4 downstream) + 16 (upstream) = 96
        else:
            print("un-supported number of kernels")

    elif pcie_gen == "gen3" and cpu_vendor == "intel": 
        acc_pcie_lane = 16
        #acc_pcie_lane = acc_pcie_lane * 2 # a pair of accelerator
        cpu_pcie_lane = 48
        upstream_ports = int(cpu_pcie_lane/acc_pcie_lane)

        full_rate = 15.754 * 1e9 # x16 rate
        # for the case of 5, 10, 15 kernels
        if num_kernel == 1:
            power = 0
        elif num_kernel == 5:
            first_layer_power = upstream_ports * 22.5 # use 81-lane, lanes = 16*4 (4 downstream) + 16 (upstream) = 80
            second_layer_power = upstream_ports * 2 * 14.5 # 2 first-layer PCIe switch, use 49-lane, lanes = 16*2 (2 downstream) + 16 (upstream) = 48
        elif num_kernel == 10: 
            first_layer_power = upstream_ports * 22.5 # use 81-lane, lanes = 16*4 (4 downstream) + 16 (upstream) = 80
            second_layer_power = upstream_ports * 4 * 16.2 # 4 first-layer PCIe switch, use 65-lane, lanes = 16*3 (2 downstream) + 16 (upstream) = 64
            power = first_layer_power + second_layer_power
        elif num_kernel == 15:
            first_layer_power = upstream_ports * 22.5 # use 81-lane, lanes = 16*4 (4 downstream) + 16 (upstream) = 80
            second_layer_power = upstream_ports * 5 * 16.2 # 5 first-layer PCIe switch, use 65-lane, lanes = 16*3 (2 downstream) + 16 (upstream) = 64
            power = first_layer_power + second_layer_power
        else:
            print("un-supported number of kernels")

    elif pcie_gen == "gen4" and cpu_vendor == "amd":
        acc_pcie_lane = 8
        #acc_pcie_lane = acc_pcie_lane * 2 # a pair of accelerator
        cpu_pcie_lane = 128
        upstream_ports = int(cpu_pcie_lane/acc_pcie_lane)

        if num_kernel == 1 or num_kernel == 5:
            power = 0
        elif num_kernel == 10: 
            power = num_kernel*11.44
            #use 26-lane, lanes = 8*2 (2 downstream) + 8 (upstream) = 24
        elif num_kernel == 15:
            power = num_kernel*11.44
            #use 26-lane, lanes = 8*2 (2 downstream) + 8 (upstream) = 24
        else:
            print("un-supported number of kernels")

    elif pcie_gen == "gen4" and cpu_vendor == "intel":
        acc_pcie_lane = 8
        cpu_pcie_lane = 64
        upstream_ports = int(cpu_pcie_lane/acc_pcie_lane)

        if num_kernel == 1:
            power = 0
        elif num_kernel == 5:
            power = upstream_ports * 11.44 # use 26-lane, lanes = 8*2 (2 downstream) + 8 (upstream) = 24
        elif num_kernel == 10:
            power = upstream_ports * 18.81 # use 50-lane, lanes = 8*4 (2 downstream) + 8 (upstream) = 40
        elif num_kernel == 15:
            power = upstream_ports * 26.12  # use 66-lane, lanes = 8*6 (2 downstream) + 8 (upstream) = 56
        else:
            print("un-supported number of kernels")

    elif pcie_gen == "gen5" and cpu_vendor == "amd":
        acc_pcie_lane = 4
        cpu_pcie_lane = 128
        upstream_ports = int(cpu_pcie_lane/acc_pcie_lane)
        if num_kernel == 1 or num_kernel == 5 or num_kernel == 10: 
            power = 0
        elif num_kernel == 15:
            power = (num_kernel*3 - upstream_ports)*14.0 # use 4*2 + 4 = 12 but we the min is 24 lanes
        else:
            print("un-supported number of kernels")    
    elif pcie_gen == "gen5" and cpu_vendor == "intel":
        acc_pcie_lane = 4
        cpu_pcie_lane = 80
        upstream_ports = int(cpu_pcie_lane/acc_pcie_lane)
        if num_kernel == 1 or num_kernel == 5: 
            power = 0
        elif num_kernel == 10:
            power = (num_kernel*3 - upstream_ports)*14.0 # use 4*2 + 4 = 12 but we the min is 24 lanes
        elif num_kernel == 15:
            power = upstream_ports*14.0 # use 4*3 + 4 = 16 but we the min is 24 lanes
        else:
            print("un-supported number of kernels")
    else:
        print("un-supported")


    return power # in watts


def pcie_switching_power_dmx_on_cpu(data_size:int, num_kernel: int, pcie_gen: str, cpu_vendor: str) -> float:
    pcie_mps = 256 # max_payload_size of PCIe we assume, which is very common
    mem_wrtie_hdr = 24 # 2B framing, 6B DLL header, 4B TLP header, and 12B MWr header
    raw_size = data_size
    data_size = np.ceil(data_size/pcie_mps) * mem_wrtie_hdr + data_size
    #print(f"total size {data_size}, raw size {raw_size}")

    power = 0
    # x16 has 15.754 GB/s
    if pcie_gen == "gen3" and cpu_vendor == "amd": 
        acc_pcie_lane = 16
        #acc_pcie_lane = acc_pcie_lane * 2 # a pair of accelerator
        cpu_pcie_lane = 128
        full_rate = 15.754 * 1e9 # x16 rate
        upstream_ports = int(cpu_pcie_lane/acc_pcie_lane)

        if num_kernel == 1:
            power = 0
        elif num_kernel == 5:
            power = upstream_ports * 14.5 # use 49-lane, lanes = 16*2 (2 downstream) + 16 (upstream) = 48
        elif num_kernel == 10:
            power = upstream_ports * 16.2 # use 65-lane, lanes = 16*3 (3 downstream) + 16 (upstream) = 64
        elif num_kernel == 15:
            power = upstream_ports * 22.5 # use 81-lane, lanes = 16*4 (4 downstream) + 16 (upstream) = 80
        else:
            print("un-supported number of kernels")

    elif pcie_gen == "gen3" and cpu_vendor == "intel": 
        acc_pcie_lane = 16
        #acc_pcie_lane = acc_pcie_lane * 2 # a pair of accelerator
        cpu_pcie_lane = 48
        upstream_ports = int(cpu_pcie_lane/acc_pcie_lane)

        full_rate = 15.754 * 1e9 # x16 rate
        # for the case of 5, 10, 15 kernels
        if num_kernel == 1:
            power = 0
        elif num_kernel == 5:
            power = upstream_ports * 22.5 # use 81-lane, lanes = 16*4 (4 downstream) + 16 (upstream) = 80
        elif num_kernel == 10: 
            first_layer_power = upstream_ports * 22.5 # use 81-lane, lanes = 16*4 (4 downstream) + 16 (upstream) = 80
            second_layer_power = upstream_ports * 4 * 14.5 # 4 first-layer PCIe switch, use 49-lane, lanes = 16*2 (2 downstream) + 16 (upstream) = 48
            power = first_layer_power + second_layer_power
        elif num_kernel == 15:
            first_layer_power = upstream_ports * 22.5 # use 81-lane, lanes = 16*4 (4 downstream) + 16 (upstream) = 80
            second_layer_power = upstream_ports * 5 * 14.5 # 4 first-layer PCIe switch, use 49-lane, lanes = 16*2 (2 downstream) + 16 (upstream) = 48
            power = first_layer_power + second_layer_power
        else:
            print("un-supported number of kernels")

    elif pcie_gen == "gen4" and cpu_vendor == "amd":
        acc_pcie_lane = 8
        #acc_pcie_lane = acc_pcie_lane * 2 # a pair of accelerator
        cpu_pcie_lane = 128
        upstream_ports = int(cpu_pcie_lane/acc_pcie_lane)

        if num_kernel == 1 or num_kernel == 5:
            power = 0
        elif num_kernel == 10: 
            power = (num_kernel*2 - upstream_ports)*11.44 # needs 20 ports, we have 16 ports already
            #use 26-lane, lanes = 8*2 (2 downstream) + 8 (upstream) = 24
        elif num_kernel == 15:
            power = (num_kernel*2 - upstream_ports)*11.44
            #use 26-lane, lanes = 8*2 (2 downstream) + 8 (upstream) = 24
        else:
            print("un-supported number of kernels")

    elif pcie_gen == "gen4" and cpu_vendor == "intel":
        acc_pcie_lane = 8
        cpu_pcie_lane = 64
        upstream_ports = int(cpu_pcie_lane/acc_pcie_lane)

        if num_kernel == 1:
            power = 0
        elif num_kernel == 5:
            power = upstream_ports * 11.44 # use 26-lane, lanes = 8*2 (2 downstream) + 8 (upstream) = 24
        elif num_kernel == 10:
            power = upstream_ports * 13.18 # use 34-lane, lanes = 8*3 (2 downstream) + 8 (upstream) = 32
        elif num_kernel == 15:
            power = upstream_ports * 18.81  # use 50-lane, lanes = 8*4 (2 downstream) + 8 (upstream) = 40
        else:
            print("un-supported number of kernels")

    elif pcie_gen == "gen5" and cpu_vendor == "amd":
        acc_pcie_lane = 4
        cpu_pcie_lane = 128
        if num_kernel == 1 or num_kernel == 5 or num_kernel == 10 or num_kernel == 15:
            power = 0
        else:
            print("un-supported number of kernels")    
    elif pcie_gen == "gen5" and cpu_vendor == "intel":
        acc_pcie_lane = 4
        cpu_pcie_lane = 80
        upstream_ports = int(cpu_pcie_lane/acc_pcie_lane)
        if num_kernel == 1 or num_kernel == 5 or num_kernel == 10:
            power = 0
        elif num_kernel == 15:
            power = (num_kernel*2 - upstream_ports)*14.0 # use 4*2 + 4 = 12 but we the min is 24 lanes
        else:
            print("un-supported number of kernels")
    else:
        print("un-supported")


    return power # in watts


def dma_time(dmx_placment: str,data_size:int, num_kernel: int, pcie_gen: str, cpu_vendor: str) -> float:
    acc_axi_delay = 57*0.001*0.001 # 57 ns
    time = 0
    if dmx_placment == "cpu-only":
        time = dma_time_dmx_on_cpu(data_size, num_kernel, pcie_gen, cpu_vendor)*4
    elif dmx_placment == "cpu":
        time = dma_time_dmx_on_cpu(data_size, num_kernel, pcie_gen, cpu_vendor)*4
    elif dmx_placment == "pcie":
        time = dma_time_dmx_on_pcie(data_size, num_kernel, pcie_gen, cpu_vendor)*4
    elif dmx_placment == "acc":
        time = dma_time_dmx_on_acc(data_size, num_kernel, pcie_gen, cpu_vendor)*3 + acc_axi_delay*2
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
            rate = full_rate*0.9375
            time = data_size/rate
            time = time * 1000 # in ms
            time = time + pcie_switching_delay(raw_size,pcie_gen)
        elif num_kernel == 15:
        # 15*16=240 lanes, 240/3 = (80, 80, 80)
            rate = full_rate*0.78125
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
        elif num_kernel == 10:
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
        elif num_kernel == 10:
        # 10*16=160 lanes, 160/3 = (64, 48, 48)
            rate = full_rate*2/3
            time = data_size/rate
            time = time * 1000 # in ms
            time = time + pcie_switching_delay(raw_size,pcie_gen)
        elif num_kernel == 15:
        # 15*16=240 lanes, 240/3 = (80, 80, 80)
            rate = full_rate/2
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