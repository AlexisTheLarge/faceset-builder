
def get_num_bits_different(hash1, hash2):
	    return bin(hash1 ^ hash2).count('1')