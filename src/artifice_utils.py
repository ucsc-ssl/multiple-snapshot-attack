import math
from scipy.stats import binom

block_size = 4096
small_checksum = 0
num_days = 365

def calc_metadata_size_aont(blocks, parity, data, replicas, verbose):
    """
    Calculates the size of Artifice instances.

    blocks: number of blocks
    parity: number of shares per - reconstruction threshold
    data:
    replicas:
    verbose:
    """
    pointer_size = 4
    art_block_hash = 16

    amplification_factor = (parity + data)/data
    carrier_block_tuple = pointer_size + small_checksum
    record_size = (parity * carrier_block_tuple) + art_block_hash
    pointers_per_pointerblock = (block_size / pointer_size - 1)

    entries_per_block = math.floor(block_size / record_size)
    num_map_blocks = math.ceil((blocks / data) / entries_per_block)
    num_map_map_blocks = math.ceil((num_map_blocks / data) / entries_per_block)
    num_pointer_blocks = math.ceil(num_map_map_blocks / pointers_per_pointerblock)
    metadata_size = ((num_pointer_blocks + 1) * replicas) + num_map_map_blocks + num_map_blocks
    metadata_overhead = (metadata_size / blocks) * 100
    effective_artifice_data_size = math.ceil((blocks * amplification_factor))
    effective_artifice_size = effective_artifice_data_size + metadata_size
    #in the case of AONT-RS we can pack multiple shares in each free block. So a freepspace block can hold up to block_size/data shares
    shares_per_block = math.floor(block_size/data)
    metadata_stats = (num_map_blocks, num_map_map_blocks, num_pointer_blocks, replicas, metadata_size, effective_artifice_data_size, shares_per_block)

    if verbose == True:
        print("|---Metadata size for AONT-----|")
        print("Codeword configuration: {} data blocks, {} parity blocks".format(data, parity))
        print("Write amplification factor: {}".format(amplification_factor))
        print("Record Size: {} bytes".format(record_size))
        print("Entries per map block: {}".format(entries_per_block))
        print("Number of map blocks: {}".format(num_map_blocks))
        print("Number of map map blocks: {}".format(num_map_map_blocks))
        print("Number of pointer bocks: {}".format(num_pointer_blocks))
        print("Metadata size in blocks: {}".format(metadata_size))
        print("Metadata overhead: {} percent".format(metadata_overhead))
        print("effective artifice size: {}".format(effective_artifice_size))

    return metadata_stats

def prob_survival_aont(d, m, p):
    """
    Probability that an Artifice instance will survive.

    d: number of data blocks
    m: number of parity blocks
    p:
    """
    return binom.cdf(m, m+d, p)

def calc_total_size_aont(size, parity, data):
    """
    Calculates the total size of an Artifice instance, including metadata and data blocks.

    size: size of the disk in blocks
    parity: total shares - threshold to reconstruct data
    data: number of data blocks
    """
    volume_stats = calc_metadata_size_aont(size, parity, data, 8, False)
    return volume_stats[4] + volume_stats[5]

def prob_metadata_alive_aont(d, m, art_size, blocks_over, free_blocks):
    """
    Calculates the probability that Artifice's metadata is still functional.

    d: number of data blocks
    m: number of parity blocks
    art_size: size of Artifice instance
    blocks_over: blocks overwritten
    free_blocks: number of free blocks
    """
    prob = blocks_over / free_blocks
    metadata = calc_metadata_size_aont(art_size, m, d, 8, False)
    return math.pow(prob_survival_aont(d, m, prob), (metadata[4] * num_days))

def prob_artifice_alive_aont(d, m, art_size, blocks_over, free_blocks):
    """
    Probability the Artifice instance will survive.

    d: number of data blocks
    m: number of parity blocks
    art_size: size of Artifice instance
    blocks_over: blocks overwritten
    free_blocks: number of free blocks
    """
    prob = blocks_over / free_blocks
    total_size = calc_total_size_aont(art_size, m, d)
    return math.pow(prob_survival_aont(d, m, prob), (total_size * num_days))