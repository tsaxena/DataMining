import pprint

class MyClassifier(object):

	def __init__(self, trainset):
		self.data_points = 0
		self.unique_features = {}
		self.unique_klasses = {}
		self.train_set = trainset
		self.__train()

	def train(self):
		self.data_points = 0
	
		for (features, c) in self.train_set:
			self.data_points += 1 
			if not self.unique_klasses.has_key(c): 
				self.unique_klasses[c] = {"total" : 1, "prob" : 0.0, "feature_count" : 0 , "features": {}}
			else:
				self.unique_klasses[c]["total"] += 1
			for f in features:
				if not self.unique_features.has_key(f):
					self.unique_features[f] = 0
				if not self.unique_klasses[c]["features"].has_key(f):
					self.unique_klasses[c]["features"][f] =  1
				else:
					self.unique_klasses[c]["features"][f] += 1
				self.unique_klasses[c]["feature_count"] += 1
 
	
		# for all features in classes		
		for c in self.unique_klasses:
			self.unique_klasses[c]["prob"] = (1.0 *self.unique_klasses[c]["total"]) / self.data_points 
		

		#get the conditional probabilities for the features
		for c in self.unique_klasses:
			for f in self.unique_klasses[c]["features"]:
				self.unique_klasses[c]["features"][f] = (1.0 * self.unique_klasses[c]["features"][f] +  1) / (self.unique_klasses[c]["feature_count"] + len (self.unique_features))
		
		#+ 1)/ (self.unique_klasses[c]["feature_count"] + len(self.unique_features))
		
		pp1 = pprint.PrettyPrinter(indent=4)		
		pp1.pprint(self.unique_klasses)
				
		pp = pprint.PrettyPrinter(indent=2)		
		pp.pprint(self.unique_features)		


	__train = train

	def classify(self, classify_set):
		#print classify_set
		
		prob_of_klasses = {}
		for c in self.unique_klasses:
			if not prob_of_klasses.has_key(c):
				prob_of_klasses[c] = self.unique_klasses[c]["prob"]
	

		for c in self.unique_klasses:
			for f in classify_set:
				if not self.unique_klasses[c]["features"].has_key(f):
				 	prob_of_klasses[c] *= 0.001		
				else:	
					prob_of_klasses[c] *= self.unique_klasses[c]["features"][f]

		pp = pprint.PrettyPrinter(indent=2)		
		pp.pprint(prob_of_klasses)	
		return max(prob_of_klasses, key=prob_of_klasses.get)
