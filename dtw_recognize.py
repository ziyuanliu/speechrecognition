#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: ziyuanliu
# @Date:   2014-11-06 15:12:42
# @Last Modified by:   ziyuanliu
# @Last Modified time: 2014-11-10 15:42:57


#TEAM: ZIYUAN LIU && PETER LOOMIS

from sys import argv
import sys
from collections import defaultdict
import glob
import os
import numpy as np 
from numpy import array, zeros, argmin, inf
from numpy.linalg import norm
import operator
import traceback 
from scikits.talkbox.features import mfcc
from scikits.audiolab import wavread
import math

k = 3

class MFCC(object):
	"""docstring for MFCC"""
	def __init__(self, model):
		super(MFCC, self).__init__()
		self.model = model
		self.speakers = model.keys()
		self.alphabets = model[self.speakers[0]] #expect consistency in speakership
		self.words = self._generatewords(model)

	def _generatewords(self,md):
		model = {}
		for k1 in md.keys():
			for k2 in md[k1].keys():
				key = '#'.join([k1,k2])
				model[key]=md[k1][k2]
		return model

def dtw(model,testword):
	previous_row = [float('inf') for i in model]
	previous_row.insert(0,0)
	norm = np.linalg.norm
	for i, c1 in enumerate(testword):
		current_row = [float('inf')]
		for j, c2 in enumerate(model):
			sub_cost = norm(c1-c2)
			# sub_cost = normalize(c1-c2)
			prev_min = min(previous_row[j],current_row[j],previous_row[j+1])
			current_row.append(sub_cost+prev_min)
			
		previous_row = current_row
	return previous_row[-1]

def multiprocess_knn(word):
	testauthor,testword = word.split('#')
	testmodel = testingMFCC.words[word]
	wrd = []
	for w in trainingMFCC.words.keys():
		trainingauthor,trainingword = w.split('#')
		trainingmodel = trainingMFCC.words[w]
		# print "hi"
		_dtw = dtw(trainingmodel,testmodel)
		# print _dtw
		wrd.append(('#'.join([trainingauthor,trainingword]),_dtw))
	ordered = sorted(wrd,key=lambda x: x[1])
	print "processed",testword
	sys.stdout.flush()
	return {word:ordered[:k]}

def finished_knn(args):
	correct = 0
	results = {}
	[results.update(arg) for arg in args]
	for key in results.keys():
		result = results[key]
		print key,result
		ctr = defaultdict(int)
		for k in result:
			speaker, word = k[0].split('#')
			ctr[word]+=1
		max_key = max(ctr.iteritems(), key=operator.itemgetter(1))[0]
		print "hypothesis for ",word,'is',max_key,'correct?:',word==max_key,ctr
		if word==max_key:
			correct +=1
	print "accuracy is",(float(correct)/len(args))*100

def readmfcc(training,testing):
	trainingdict = defaultdict(dict)
	
	os.chdir(training)
	for f in glob.glob("*.mfc"):
		speaker,letter,_ = f.split('.')[0].split('-')
		with open(f) as fd:
			temp = []
			for row in fd.read().split("\n"):
				if len(row)>0:
					temp.append([float(num) for num in row.split()])

			trainingdict[speaker][letter] = np.array(temp)
	trainingMFCC = MFCC(trainingdict)

	testingdict = defaultdict(dict)
	os.chdir(testing)
	for f in glob.glob("*.mfc"):
		speaker,letter,_ = f.split('.')[0].split('-')
		with open(f) as fd:
			temp = []
			for row in fd.read().split("\n"):
				if len(row)>0:
					temp.append([float(num) for num in row.split()])

			testingdict[speaker][letter] = np.array(temp)
	testingMFCC = MFCC(testingdict)

	return trainingMFCC,testingMFCC

def readwav(trainfolder,testfolder):

	os.chdir(trainfolder)
	for f in glob.glob("*.wav"):
		speaker,letter,_ = f.split('.')[0].split('-')
		mfccname = f.split('.')[0]+".mfc"
		data, fs = wavread(f)[:2]

		cep= mfcc(data, fs=fs, nwin=int(fs*0.025))[0]
		np.savetxt(mfccname,cep,fmt='%.10f')
		


	os.chdir(testfolder)
	for f in glob.glob("*.wav"):
		speaker,letter,_ = f.split('.')[0].split('-')
		mfccname = f.split('.')[0]+".mfc"
		data, fs = wavread(f)[:2]

		cep= mfcc(data, fs=fs, nwin=int(fs*0.025))[0]
		np.savetxt(mfccname,cep,fmt='%.10f')
	

if __name__ == '__main__':
	cwd = os.getcwd()
	foldername = cwd+'/'+argv[1]
	testfolder = foldername+"/test/"
	trainfolder = foldername+"/train/"

	# readwav(trainfolder,testfolder)

	global trainingMFCC,testingMFCC
	trainingMFCC,testingMFCC = readmfcc(trainfolder,testfolder)
	
	import multiprocessing

	p = multiprocessing.Pool(8)
	try:
		p.map_async(multiprocess_knn, testingMFCC.words.keys(), callback=finished_knn)
		p.close()
		p.join()
	except Exception as e:
		traceback.print_exc(e)
		print e




