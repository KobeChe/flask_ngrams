import time
import re
import pickle
"""==================================================================== """
def save(dict):
	f = open("./pickle.txt", "wb+")
	pickle.dump(dict, f)
	f.close()

def load(path):
	f = open(path, "rb+")
	data = pickle.load(f)
	f.close()
	return data

"""==================================================================== """
def genDict(path):
	"""generate bigram dictionary"""
	d = {}
	with open(path, 'r') as f:
		for line in f.readlines():
			line = line.strip().split("\t")
			if line[0].lower() in d.keys():
				d[line[0].lower()] += [line[1].lower()] 
			else:
				d[line[0].lower()] = [line[1].lower()] 
	return d



def test(testData, dictionary):
	"""test bigram"""
	total_word = 0
	top_3_correct = 0
	used_count = 0
	with open(testData, 'r') as f:
		for line in f.readlines():
			line = line.strip().lower()
			line = re.sub(r'[^\w\s]','',line)
			line = line.split(' ')

			total_word += len(line)
			for i in range(1,len(line)):
				if line[i-1] in dictionary.keys():
					used_count += 1
					if line[i] in dictionary[line[i-1]][:3]:
						top_3_correct += 1
	print(top_3_correct/total_word)
	print(top_3_correct/used_count)

# path = './en_US_bigram.txt'
# # path = './en_US_20200716/en_US_bigram.txt'
# dictionary = genDict(path)

# data = './en_USData'
# test(data,dictionary)

"""==================================================================== """

def genDictN(path, n):
	"""generate ngram dictionary by setting n"""
	d = {}
	with open(path, 'r') as f:
		for line in f.readlines():
			line = line.strip().split("\t")
			key = ''
			for i in range(n-1):
				key+= line[i].lower()+' '
			key = key.rstrip() 
			if key in d.keys():
				d[key][line[n-1].lower()] = int(line[-1])
			else:
				d[key] = {}
				d[key][line[n-1].lower()] = int(line[-1])
	return d

def testN(testData, dictionary, n):
	"""test with ngram dictionary by setting n"""
	total_word = 0
	top_3_correct = 0
	used_count = 0
	start = time.time()
	with open(testData, 'r') as f:
		for line in f.readlines():
			line = line.strip().lower()
			line = re.sub(r'[^\w\s]','',line)
			line = line.split(' ')
			total_word += len(line)
			for i in range(n-1,len(line)):
				key = ''
				for j in range(n-1):
					key += line[i-(n-1-j)].lower() + ' '
				key = key.rstrip()
				if key in dictionary.keys():
					used_count += 1
					if line[i] in dictionary[key][:3]:
						top_3_correct += 1
	end = time.time()
	print("test with",n,"gram dict:")
	print(top_3_correct/total_word)
	print(top_3_correct/used_count)
	print("total time:", end-start)
	print("time on each word: %.3f ms" % ((end-start)*1000/total_word))
	print()
