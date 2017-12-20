import numpy as np

def generateX(bin_len):
	return np.round(np.random.rand(bin_len)).astype(int)

def generateY(bin_list):
	bin_str = ""
	for k in bin_list:
		bin_str += str(int(k))
	return int(bin_str, 2)
	
def datasets(num, bin_len):
	x = np.zeros((num, bin_len))
	y = np.zeros((num))

	for i in range(num):
		x[i] = generateX(bin_len)
		y[i] = generateY(x[i])
	return x, y	

