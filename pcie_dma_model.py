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
    if dmx_placment == "cpu-only": # DMX uses CPU not DRX
        time = pcie_switching_power_dmx_on_cpu(data_size, num_kernel, pcie_gen, cpu_vendor)*4
    elif dmx_placment == "cpu": # DRX on CPU
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

pci_gen3_power = {33:8.1, 49:14.5, 65:16.2, 81:22.5, 97:24.3}
pci_gen4_power = {33:8.1, 49:14.5, 65:16.2, 81:22.5, 97:24.3}

def pcie_switching_power_dmx_on_pcie(data_size:int, num_kernel: int, pcie_gen: str, cpu_vendor: str) -> float:
    power = 0
    # x16 has 15.754 GB/s
    if pcie_gen == "gen3" and cpu_vendor == "amd": 
        acc_pcie_lane = 16
        #acc_pcie_lane = acc_pcie_lane * 2 # a pair of accelerator
        cpu_pcie_lane = 128
        upstream_ports = int(cpu_pcie_lane/acc_pcie_lane)
        max_pcie_switch_lane = max(pci_gen3_power.keys())
        max_downstream_ports = np.floor((max_pcie_switch_lane - acc_pcie_lane)/acc_pcie_lane)

        if num_kernel == 1:            
            power = 0
        elif num_kernel == 5:
            #for downstream_ports in range(1, max_downstream_ports+1):
            single_device_pcie = upstream_ports*2 - (num_kernel*2 + 2)
            num_switches = upstream_ports - single_device_pcie
            power = num_switches * 14.5 # use 49-lane, 14.5 watts, lanes = 16*2 (2 downstream) + 16 (upstream) = 48
        elif num_kernel == 10:
            power = upstream_ports * 16.2 # use 65-lane, 16.2 watts, lanes = 16*3 (3 downstream) + 16 (upstream) =640
        elif num_kernel == 15:
            five_device_switch = (num_kernel*2 + 4) - upstream_ports * 4
            power = (upstream_ports - five_device_switch)*22.5 + five_device_switch * 24.3
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
            single_device_pcie = upstream_ports*2 - (num_kernel*2 + 2)
            num_switches = upstream_ports - single_device_pcie
            power = num_switches * 11.44
            #use 26-lane, lanes = 8*2 (2 downstream) + 8 (upstream) = 24
        elif num_kernel == 15:
            power = upstream_ports*11.44
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
            #for downstream_ports in range(1, max_downstream_ports+1):
            single_device_pcie = upstream_ports*2 - (num_kernel*2 + 2)
            num_switches = upstream_ports - single_device_pcie
            power = num_switches * 11.44  # use 24-lane, 11.44 watts, lanes = 8*2 (2 downstream) + 8 (upstream) = 24
        elif num_kernel == 10:
            power = upstream_ports * 13.18 # use 34-lane, 13.18 watts, lanes = 8*3 (3 downstream) + 8 (upstream) = 32
        elif num_kernel == 15:
            power = upstream_ports * 18.81 # use 34-lane, 13.18 watts, lanes = 8*5 (3 downstream) + 8 (upstream) = 32
        else:
            print("un-supported number of kernels")
    else:
        print("un-supported")

    return power

def pcie_switching_power_single_dmx_on_pcie(data_size:int, num_kernel: int, pcie_gen: str, cpu_vendor: str) -> float:
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
    if dmx_placment == "cpu-only": # DMX uses CPU not DRX
        time = dma_time_dmx_on_cpu(data_size, num_kernel, pcie_gen, cpu_vendor)*4
    elif dmx_placment == "cpu": # DRX on CPU
        time = dma_time_dmx_on_cpu(data_size, num_kernel, pcie_gen, cpu_vendor)*4
    elif dmx_placment == "pcie": # DRX on PCIe card, over-provisioned DRX, aka SIMD units
        time = dma_time_dmx_on_pcie(data_size, num_kernel, pcie_gen, cpu_vendor)
    elif dmx_placment == "acc": # DRX on Acc
        # 2x CPU -> Acc data movement with possible bandwidth over-subscription
        # (a) 2x CPU -> Acc data movement with possible bandwidth over-subscription
        # (b) 1x of full bandwidth between two accelerator 
        time = dma_time_dmx_on_acc(data_size, num_kernel, pcie_gen, cpu_vendor) + acc_axi_delay*2
    else:
        print(f"unsuppoted DMX placement {dmx_placment}")
    return time

def dma_time_dmx_on_acc(data_size:int, num_kernel: int, pcie_gen: str, cpu_vendor: str) -> float:
    pcie_mps = 256 # max_payload_size of PCIe we assume, which is very common
    mem_wrtie_hdr = 24 # 2B framing, 6B DLL header, 4B TLP header, and 12B MWr header
    raw_size = data_size
    data_size = np.ceil(data_size/pcie_mps) * mem_wrtie_hdr + data_size
    time = 0
    # x16 has 15.754 GB/s *2

    if pcie_gen == "gen4" and cpu_vendor == "amd":
        acc_pcie_lane = 8
        acc_pcie_lane = acc_pcie_lane * 2 # a pair of accelerator
        cpu_pcie_lane = 64
        full_rate = 15.754 * 1e9 * 2# x16 rate
        if num_kernel == 1 or num_kernel == 5:
        # 10*8=80 lanes, 80/4 = (24, 24, 16, 16)
            rate = full_rate
            time = data_size/rate
            time = time * 1000 # in ms
        elif num_kernel == 10 or num_kernel == 15:
        # 15*8=120 lanes, 120/4 = (32,32,32,24)
            rate = full_rate
            time = data_size/rate
            time = time * 1000 # in ms
            time = time + pcie_switching_delay(raw_size,pcie_gen)
        else:
            print("un-supported number of kernels")

    elif pcie_gen == "gen4" and cpu_vendor == "intel":
        acc_pcie_lane = 8
        acc_pcie_lane = acc_pcie_lane * 2 # a pair of accelerator
        cpu_pcie_lane = 128
        full_rate = 15.754 * 1e9 * 2# x16 rate
        if num_kernel == 1: #or num_kernel == 5:
            rate = full_rate
            one_trip_time = data_size/rate
            one_trip_time = one_trip_time * 1000 # in ms
            cpu_acc_time = one_trip_time * 2
            inter_acc_time = one_trip_time
            time = cpu_acc_time + inter_acc_time
        elif num_kernel == 5:
            rate = full_rate
            one_trip_time = data_size/rate
            one_trip_time = one_trip_time * 1000 # in ms

            cpu_acc_time = one_trip_time * 2

            inter_acc_time = one_trip_time + (2/5)*pcie_switching_delay(raw_size,pcie_gen)

            time = cpu_acc_time + inter_acc_time

        elif num_kernel == 10:
            downlink_rate = full_rate*(6/8*1+2/8*1/2)
            inter_acc_rate = full_rate

            cpu_acc_trip_time = data_size/downlink_rate
            cpu_acc_trip_time = cpu_acc_trip_time * 1000 # in ms
            cpu_acc_time = cpu_acc_trip_time * 2

            inter_acc_trip_time = data_size/inter_acc_rate
            inter_acc_trip_time = inter_acc_trip_time * 1000
            inter_acc_time = inter_acc_trip_time + pcie_switching_delay(raw_size,pcie_gen)

            time = cpu_acc_time + inter_acc_time
        elif num_kernel == 15:
            downlink_rate = full_rate*(7/8*1/2+1/8*1)
            inter_acc_rate = full_rate

            cpu_acc_trip_time = data_size/downlink_rate
            cpu_acc_trip_time = cpu_acc_trip_time * 1000 # in ms
            cpu_acc_time = cpu_acc_trip_time * 2

            inter_acc_trip_time = data_size/inter_acc_rate
            inter_acc_trip_time = inter_acc_trip_time * 1000
            inter_acc_time = inter_acc_trip_time + pcie_switching_delay(raw_size,pcie_gen)

            time = cpu_acc_time + inter_acc_time
        else:
            print("un-supported number of kernels")
    else:
        print("un-supported")

    # if acc_pcie_lane * num_kernel > cpu_pcie_lane:
    #     time = time + pcie_switching_delay(raw_size,pcie_gen) # both in ms

    return time # in ms

# 4-unit SIMD
def dma_time_dmx_on_pcie(data_size:int, num_kernel: int, pcie_gen: str, cpu_vendor: str) -> float:
    pcie_mps = 256 # max_payload_size of PCIe we assume, which is very common
    mem_wrtie_hdr = 24 # 2B framing, 6B DLL header, 4B TLP header, and 12B MWr header
    raw_size = data_size
    data_size = np.ceil(data_size/pcie_mps) * mem_wrtie_hdr + data_size
    full_rate = 15.754 * 1e9 * 2

    time = 0
    # TODO for AND if
    # [ ](a) 2x CPU -> Acc data movement with possible bandwidth over-subscription
    # [X](b) 1x of slightly reduced bandwidth between two accelerator 
    if pcie_gen == "gen4" and cpu_vendor == "amd": 
        acc_pcie_lane = 8
        acc_pcie_lane = acc_pcie_lane * 2 # a pair of accelerator
        cpu_pcie_lane = 128        
        if num_kernel == 1:
            rate = full_rate
            time = data_size/rate
            time = time * 1000 # in ms
        elif num_kernel == 5:
            rate = full_rate*2/3
            time = data_size/rate
            time = time * 1000 # in ms
        elif num_kernel == 10:
            rate = full_rate*2/3
            time = data_size/rate
            time = time * 1000 # in ms
            # 3/10*1 + 7/10*2, 3 pairs of accelerators are co-located with DRX, 7 others needs to go out of its own PCIe switch
            # + the PCIe switch of the DRX
            time = time + (3/10*2 + 7/10*1)*pcie_switching_delay(raw_size,pcie_gen)
        elif num_kernel == 15:
            rate = full_rate*1/2
            time = data_size/rate
            time = time * 1000 # in ms
            time = time + (4/15*2 + 11/15*1)*pcie_switching_delay(raw_size,pcie_gen)
        else:
            print("un-supported number of kernels")

    elif pcie_gen == "gen4" and cpu_vendor == "intel":
        acc_pcie_lane = 8
        acc_pcie_lane = acc_pcie_lane * 2 # a pair of accelerator
        cpu_pcie_lane = 64
        if num_kernel == 1:
        # 10*8=80 lanes, 80/4 = (24, 24, 16, 16)
            rate = full_rate
            one_trip_time = data_size/rate
            one_trip_time = one_trip_time * 1000 # in ms
            cpu_acc_time = one_trip_time * 2
            inter_acc_time = one_trip_time
            time = cpu_acc_time + inter_acc_time
        elif num_kernel == 5:
            rate = full_rate*2/3
            one_trip_time = data_size/rate            
            one_trip_time = one_trip_time * 1000 # in ms
            cpu_acc_time = one_trip_time * 2
            # 3 PCIe switches have 2 devices, 5 other upstream directly connect a single device
            inter_acc_time = one_trip_time + (2/5*2 + 3/5*1)*pcie_switching_delay(raw_size,pcie_gen)
            time = cpu_acc_time + inter_acc_time
        elif num_kernel == 10:
            #2-SIMD -> 1.25 x compute, 0.16 full rate, 3-SIMD -> 1x compute, 0.24 full rate
            rate = full_rate*2/3
            one_trip_time = data_size/rate            
            one_trip_time = one_trip_time * 1000 # in ms
            cpu_acc_time = one_trip_time * 2
            inter_acc_time = one_trip_time + (4/10*1 + 6/10*2)*pcie_switching_delay(raw_size,pcie_gen)
            time = cpu_acc_time + inter_acc_time
        elif num_kernel == 15:
            # 3-SIMD -> 1.25 x compute, 0.1125 full rate, 4-SIMD -> 1x compute, 0.15 full rate
            rate = full_rate*1/2
            one_trip_time = data_size/rate            
            one_trip_time = one_trip_time * 1000 # in ms
            cpu_acc_time = one_trip_time * 2
            inter_acc_time = one_trip_time + 2*pcie_switching_delay(raw_size,pcie_gen)
            time = cpu_acc_time + inter_acc_time
        else:
            print("un-supported number of kernels")

    else: # gen 3 intel and gen 5 intel + AMD
        print("un-supported")
        time = 0

    return time

def dma_time_dmx_on_cpu(data_size:int, num_kernel: int, pcie_gen: str, cpu_vendor: str) -> float:
    pcie_mps = 256 # max_payload_size of PCIe we assume, which is very common
    mem_wrtie_hdr = 24 # 2B framing, 6B DLL header, 4B TLP header, and 12B MWr header
    raw_size = data_size
    data_size = np.ceil(data_size/pcie_mps) * mem_wrtie_hdr + data_size
    #print(f"total size {data_size}, raw size {raw_size}")

    time = 0
    if pcie_gen == "gen4" and cpu_vendor == "amd":
        acc_pcie_lane = 8
        acc_pcie_lane = acc_pcie_lane * 2 # a pair of accelerator
        cpu_pcie_lane = 128
        full_rate = 15.754 * 1e9 * 2# x16 rate
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
        full_rate = 15.754 * 1e9 * 2# x16 rate
        if num_kernel == 1:
        # 10*8=80 lanes, 80/4 = (24, 24, 16, 16)
            rate = full_rate
            time = data_size/rate
            time = time * 1000 # in ms
        elif num_kernel == 5:
            rate = full_rate
            time = data_size/rate
            time = time * 1000 # in ms
            time = time + pcie_switching_delay(raw_size,pcie_gen) # 2 out of 5 kernel pairs use PCIe switch
        elif num_kernel == 10:
            rate = full_rate/2
            time = data_size/rate
            time = time * 1000 # in ms
            time = time + pcie_switching_delay(raw_size,pcie_gen)
        elif num_kernel == 15:
        # 15*8=120 lanes, 120/4 = (32,32,32,24)
            rate = full_rate/2
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