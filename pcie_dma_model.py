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

def pcie_energy(dmx_placment: str,data_size:int, num_kernel: int, pcie_gen: str, cpu_vendor: str) -> float:
    acc_axi_delay = 57*0.001*0.001 # 57 ns
    pcie_mps = 256 # max_payload_size of PCIe we assume, which is very common
    mem_wrtie_hdr = 24 # 2B framing, 6B DLL header, 4B TLP header, and 12B MWr header
    data_size = np.ceil(data_size/pcie_mps) * mem_wrtie_hdr + data_size

    energy = 0
    if dmx_placment == "cpu-only": # DMX uses CPU not DRX
        power = 0
    elif dmx_placment == "cpu": # DRX on CPU
        power = pcie_power_dmx_on_cpu(data_size, num_kernel, pcie_gen, cpu_vendor)
        time = dma_time_dmx_on_cpu(data_size, num_kernel, pcie_gen, cpu_vendor)*4
        print(f"cpu: num-kernels:{num_kernel}, pcie-power:{power}, pcie-dma-time:{time}")
    elif dmx_placment == "pcie-under":
        power = pcie_power_dmx_on_pcie_under(data_size, num_kernel, pcie_gen, cpu_vendor)
        time, _, _= dma_time_dmx_on_pcie_under(data_size, num_kernel, pcie_gen, cpu_vendor)
    elif dmx_placment == "pcie":
        power = pcie_power_dmx_on_pcie(data_size, num_kernel, pcie_gen, cpu_vendor)
        time, _, _ = dma_time_dmx_on_pcie(data_size, num_kernel, pcie_gen, cpu_vendor)
        print(f"pcie: num-kernels:{num_kernel}, pcie-power:{power}, pcie-dma-time:{time}")
    elif dmx_placment == "acc":
        power = pcie_power_dmx_on_acc(data_size, num_kernel, pcie_gen, cpu_vendor)
        time, _, _ = dma_time_dmx_on_acc(data_size, num_kernel, pcie_gen, cpu_vendor)        
        print(f"acc: num-kernels:{num_kernel}, pcie-power:{power}, pcie-dma-time:{time}")
    else:
        print("unsuppoted DMX placement")

    energy = time*power + data_size*11/1e12*1e3 # in mJ

    return energy

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

# 98 lanes: 35.78 watts
# 82 lanes: 30.98 watts
# 66 lanes: 26.12 watts
# 50 lanes: 18.81 watts
# 34 lanes: 13.18 watts
# 26 lanes: 11.44 watts

def pcie_power_dmx_on_acc(data_size:int, num_kernel: int, pcie_gen: str, cpu_vendor: str) -> float:
    power = 0
    if pcie_gen == "gen4" and cpu_vendor == "intel":
        acc_pcie_lane = 8
        cpu_pcie_lane = 64
        upstream_ports = int(cpu_pcie_lane/acc_pcie_lane)

        if num_kernel == 1:
            power = 0
        elif num_kernel == 5:
            power = 2 * 11.44 # use 26-lane, lanes = 8*2 (2 downstream) + 8 (upstream) = 24
        elif num_kernel == 10:
            power = 6*11.44 + 2*18.81 # lanes = 8*2 (2 downstream) + 8 (upstream) = 24 or 8*4 (2 downstream) + 8 (upstream) = 40
        elif num_kernel == 15:
            power = upstream_ports * 18.81 # use 50-lane, lanes = 8*4 (2 downstream) + 8 (upstream) = 40
        else:
            print("un-supported number of kernels")
    else:
        print("un-supported")

    # per DRX internal PCIe switch
    power = power + num_kernel*8.00

    return power # in watts

def pcie_power_dmx_on_pcie_under(data_size:int, num_kernel: int, pcie_gen: str, cpu_vendor: str) -> float:
    power = 0
    # x16 has 15.754 GB/s
    if pcie_gen == "gen4" and cpu_vendor == "intel":
        acc_pcie_lane = 8
        cpu_pcie_lane = 64
        upstream_ports = int(cpu_pcie_lane/acc_pcie_lane)

        if num_kernel == 1:            
            power = 0
        elif num_kernel == 5:
            power = 2*13.18 + 2*18.81
        elif num_kernel == 10:
            power = 1*30.98 + 3*26.12
        elif num_kernel == 15:
            power = 2*35.78 + 2*30.98
        else:
            print("un-supported number of kernels")
    else:
        print("un-supported")

    return power


def pcie_power_dmx_on_pcie(data_size:int, num_kernel: int, pcie_gen: str, cpu_vendor: str) -> float:
    power = 0
    # x16 has 15.754 GB/s
    if pcie_gen == "gen4" and cpu_vendor == "intel":
        acc_pcie_lane = 8
        cpu_pcie_lane = 64
        upstream_ports = int(cpu_pcie_lane/acc_pcie_lane)

        if num_kernel == 1:            
            power = 0
        elif num_kernel == 5:
            power = 1*13.18 + 3*18.81
        elif num_kernel == 10:
            power = 2*30.98 + 2*26.12
        elif num_kernel == 15:
            power = 3*35.78 + 1*30.98
        else:
            print("un-supported number of kernels")
    else:
        print("un-supported")

    return power


def pcie_power_dmx_on_cpu(data_size:int, num_kernel: int, pcie_gen: str, cpu_vendor: str) -> float:
    if pcie_gen == "gen4" and cpu_vendor == "intel":
        acc_pcie_lane = 8
        cpu_pcie_lane = 64
        upstream_ports = int(cpu_pcie_lane/acc_pcie_lane)

        if num_kernel == 1:
            power = 0
        elif num_kernel == 5:
            power = 2*11.44 # use 26-lane, lanes = 8*2 (2 downstream) + 8 (upstream) = 24
        elif num_kernel == 10:
            power = 6*11.44 + 2*18.81
        elif num_kernel == 15:
            power = upstream_ports * 18.81  # use 50-lane, lanes = 8*4 (2 downstream) + 8 (upstream) = 40
        else:
            print("un-supported number of kernels")
    else:
        print("un-supported")

    return power # in watts


# cxl.cache running on 68B Filt mode. 
def cxl_time(dmx_placment: str,data_size:int, num_kernel: int, pcie_gen: str, cpu_vendor: str) -> float:
    time = 0
    if dmx_placment == "cpu-only": # DMX uses CPU not DRX
        time = dma_time_cxl_on_cpu(data_size, num_kernel, pcie_gen, cpu_vendor)*4
    elif dmx_placment == "cpu": # DRX on CPU
        time = dma_time_cxl_on_cpu(data_size, num_kernel, pcie_gen, cpu_vendor)*4
    else:
        print("placement not supported")

    return time 

def dma_time_cxl_on_cpu(data_size:int, num_kernel: int, pcie_gen: str, cpu_vendor: str) -> float:
    if pcie_gen == "gen4" and cpu_vendor == "intel":
        acc_pcie_lane = 8
        acc_pcie_lane = acc_pcie_lane * 2 # a pair of accelerator
        cpu_pcie_lane = 64
        full_rate = 15.754 * 1e9 * 2# x16 rate

        cxl_payload_size = 48 # G0 in slot 1,2,3
        cxl_cache_hdr = 16 + 2 + 2 # 16 for H0 in slot 0 per CXL spec, 2B for CRC, 2B for MUX
        raw_size = data_size
        data_size = np.ceil(data_size/cxl_payload_size) * cxl_cache_hdr + data_size

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

    return time

def dma_time(dmx_placment: str,data_size:int, num_kernel: int, pcie_gen: str, cpu_vendor: str) -> float:
    acc_axi_delay = 57*0.001*0.001 # 57 ns
    time = 0
    if dmx_placment == "cpu-only": # DMX uses CPU not DRX
        time = dma_time_dmx_on_cpu(data_size, num_kernel, pcie_gen, cpu_vendor)*4
    elif dmx_placment == "cpu": # DRX on CPU
        time = dma_time_dmx_on_cpu(data_size, num_kernel, pcie_gen, cpu_vendor)*4
    elif dmx_placment == "pcie": # DRX on PCIe card, over-provisioned DRX, aka SIMD units
        time, _, _ = dma_time_dmx_on_pcie(data_size, num_kernel, pcie_gen, cpu_vendor)
    elif dmx_placment == "pcie-under":
        time, _, _ = dma_time_dmx_on_pcie_under(data_size, num_kernel, pcie_gen, cpu_vendor)
    elif dmx_placment == "pcie-tri-kernel":
        time, _, inter_acc_time = dma_time_dmx_on_pcie(data_size, num_kernel, pcie_gen, cpu_vendor)
        time = time + inter_acc_time
    elif dmx_placment == "acc": # DRX on Acc
        # 2x CPU -> Acc data movement with possible bandwidth over-subscription
        # (a) 2x CPU -> Acc data movement with possible bandwidth over-subscription
        # (b) 1x of full bandwidth between two accelerator 
        time, _, _ = dma_time_dmx_on_acc(data_size, num_kernel, pcie_gen, cpu_vendor)#+ acc_axi_delay*2
    elif dmx_placment == "acc-tri-kernel":
        time, _, inter_acc_time = dma_time_dmx_on_acc(data_size, num_kernel, pcie_gen, cpu_vendor)
        time = time + inter_acc_time + acc_axi_delay*4
    else:
        print(f"unsuppoted DMX placement {dmx_placment}")
    return time

def dma_time_dmx_on_acc(data_size:int, num_kernel: int, pcie_gen: str, cpu_vendor: str) -> tuple:
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

    return time, cpu_acc_time, inter_acc_time # in ms


def dma_time_dmx_on_pcie_under(data_size:int, num_kernel: int, pcie_gen: str, cpu_vendor: str) -> tuple:
    pcie_mps = 256 # max_payload_size of PCIe we assume, which is very common
    mem_wrtie_hdr = 24 # 2B framing, 6B DLL header, 4B TLP header, and 12B MWr header
    raw_size = data_size
    data_size = np.ceil(data_size/pcie_mps) * mem_wrtie_hdr + data_size
    full_rate = 15.754 * 1e9 * 2

    time = 0
    if pcie_gen == "gen4" and cpu_vendor == "intel":
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
            rate = full_rate*1/4
            one_trip_time = data_size/rate            
            one_trip_time = one_trip_time * 1000 # in ms
            cpu_acc_time = one_trip_time * 2
            # 3 PCIe switches have 2 devices, 5 other upstream directly connect a single device
            inter_acc_time = one_trip_time + (2/5*2 + 3/5*1)*pcie_switching_delay(raw_size,pcie_gen)
            time = cpu_acc_time + inter_acc_time
        elif num_kernel == 10:
            #2-SIMD -> 1.25 x compute, 0.16 full rate, 3-SIMD -> 1x compute, 0.24 full rate
            rate = full_rate*1/5
            one_trip_time = data_size/rate            
            one_trip_time = one_trip_time * 1000 # in ms
            cpu_acc_time = one_trip_time * 2
            inter_acc_time = one_trip_time + (4/10*1 + 6/10*2)*pcie_switching_delay(raw_size,pcie_gen)
            time = cpu_acc_time + inter_acc_time
        elif num_kernel == 15:
            # 3-SIMD -> 1.25 x compute, 0.1125 full rate, 4-SIMD -> 1x compute, 0.15 full rate
            rate = full_rate*2/15
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

    return time, cpu_acc_time, inter_acc_time

# 4-unit SIMD
def dma_time_dmx_on_pcie(data_size:int, num_kernel: int, pcie_gen: str, cpu_vendor: str) -> tuple:
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

    return time, cpu_acc_time, inter_acc_time

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