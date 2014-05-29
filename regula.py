from numpy import *
import scipy.sparse

###############################
# HERE WE INITIALIZE 'T' AND 'A'
###############################
#
# A: Activation matrix
# anne : female, parent of tom
# tom 
# cristi : female
# rule : mother :- parent female


def unifies(hi,hj,i,j,ii,jj,vh1,vh2,vb11,vb12,vb21,vb22):
	rle = [hi,hj,i,j,ii,jj]
	gle = [vh1,vh2,vb11,vb12,vb21,vb22]
	return [x==y for x in rle for y in rle] == [x==y for x in gle for y in gle] 

sub2ind = lambda x,y: ravel_multi_index(y, dims=x )

class aedis:
	debug = False
	def __init__(self,entities,relations,rules):
		self.entities  = entities
		self.relations = relations
		self.rules     = rules
		self.nument    = entities.shape[0]
		self.numrel    = relations.shape[0]
		self.numpairs  = self.nument**2
		self.Ashape = (self.nument,self.nument,self.numrel)
		self.A = zeros((self.nument*self.nument*self.numrel,1))
		
		print("making templates")
		self.templates = [(rh,vh1,vh2,r1,vb11,vb12,r2,vb21,vb22) for rh in relations for r1 in relations for r2 in relations for (vh1,vh2,vb11,vb12,vb21,vb22) in rules]
		self.numrules = len(self.templates)
		print(self.numrules)
		print("alocating matrices ")
		self.Tshape = [self.nument,self.nument,self.numrel,self.nument,self.nument,self.numrel,self.nument,self.nument,self.numrel]
		self.T = scipy.sparse.lil_matrix((self.numrules,self.numrel*self.numpairs*self.numrel*self.numpairs*self.numrel*self.numpairs))
		self.wshape = (1,self.numrules)
		self.w  = scipy.sparse.lil_matrix(self.wshape)
		
		nm = ""
		for x in rules:
			for y in x:
				nm += y
		try:
			self.T = load(nm+".npy")[()]
			print("loaded templates from cache!")
		except:
			print("templates not in cache, generating all " +str(self.numrules) + " of them....")
			for tempth in range(self.numrules):
			    (relh,vh1,vh2,relb1,vb11,vb12,relb2,vb21,vb22) = self.templates[tempth]
			    rh = where(self.relations== relh)[0]
			    rb1 = where(self.relations== relb1)[0]
			    rb2 = where(self.relations== relb2)[0]
			    if not tempth % 50 : 
				    print(tempth);    
			    for hi in range(self.nument):
				for hj in range(self.nument):
				    for i in range(self.nument):
					for j in range(self.nument):
					    for ii in range(self.nument):
					       for jj in range(self.nument):
						       if unifies(hi,hj,i,j,ii,jj,vh1,vh2,vb11,vb12,vb21,vb22):

								self.T[tempth,sub2ind(self.Tshape,(hi,hj,rh,i,j,rb1,ii,jj,rb2))] = 1
			save(nm,self.T)

	def register(self,rel, entity1, entity2=0):
		try:
			if entity2:
				i1 = where(self.entities == entity1)[0]
				i2 = where(self.entities == entity2)[0]
				i3 = where(self.relations == rel)[0]
				self.A[sub2ind(self.Ashape,(i1,i2,i3))] = 0.999
			else:
				i1 = where(self.entities == entity1)[0]
				i3 = where(self.relations == rel)[0]
				self.A[sub2ind(self.Ashape,(i1,i1,i3))] = 0.999
		except:
			print("REGISTER FAIL:",rel,entity1,entity2)

	def regrule(self,relh, vh1, vh2, relb1, vb11,vb12, relb2, vb21, vb22,retry=True):
		for tempth in range(self.numrules) :
			(trelh,tvh1,tvh2,trelb1,tvb11,tvb12,trelb2,tvb21,tvb22) = self.templates[tempth]
			if (trelh == relh) and (trelb1 == relb1) and (trelb2 == relb2) and unifies(tvh1,tvh2,tvb11,tvb12,tvb21,tvb22,vh1,vh2,vb11,vb12,vb21,vb22):
				self.w[0,tempth] = 1
				return
		if retry : 
			return self.regrule(relh, vh1, vh2,relb2, vb21, vb22, relb1, vb11,vb12, retry=False) # try flipping conjunct

		print("REGRULEFAIL:",relh, vh1, vh2, relb1, vb11,vb12, relb2, vb21, vb22)
		exit()

#########################################
# PURRTY PRINT
##############################
#
	def printKnowledge(self,thresh=0.5) :
		for r in range(0,self.numrel):
		    for i in range(0,self.nument):
			for j in range(0,self.nument):
				if self.A[sub2ind((self.nument,self.nument,self.numrel),(i,j,r)),0] > thresh :
					rrr =  self.relations[r]
					eee1 = self.entities[i]
					eee2 = self.entities[j]
					print(  rrr+ "("+eee1+")." if eee1 == eee2 else rrr+"(" +eee1+","+eee2+")." )
		print("\n\n")


	def printRules(self,thresh=0.5):
	        print([self.templates[x] for x in range(self.w.shape[1]) if self.w[0,x]>thresh])


	def getForward(self):
		Fp = self.w.dot(self.T)
		F = Fp.todense()
		F = F.reshape((self.numrel*self.numpairs,self.numrel*self.numpairs*self.numrel*self.numpairs))
		self.F = F
		return F

	def step(self):
		F = self.getForward();
		A = self.A
	     	Aout = zeros(self.A.shape)
	     	for i in range(self.A.shape[0]):
		     	Aout[i,0] = ((self.A.transpose()).dot(F[i,:].reshape((self.numpairs*self.numrel,self.numpairs*self.numrel)))).dot(self.A)

		if self.debug:
			A = self.A
			AunOptimized = F.dot(kron(A,A))
			assert all(equal(AunOptimized , Aout))
		self.A =  (exp(Aout)-exp(-1*Aout))/(exp(Aout)+exp(-1*Aout))
		return Aout


	def run(self,numiters =1 ):
		out = [] 
		for t in range(0,numiters):
			A = self.step()
			out.append(A)
		return out


if __name__ == '__main__':
	entities  = array(["anne" , "tom" ,"cristi","rob"])
	relations = array(["mother" , "parent", "female","male", "father","eloped"])
	variables = [("X","Y","X","Z","Y","Z"),("X","Y","X","X","X","Y"),("X","Y","X","Y","X","Y"),("X","X","X","X","X","X")]
	aed = aedis(entities,relations,variables)
	##############################
	#   Register ground facts
	##############################
	#
	aed.register("female","anne")
	aed.register("male","rob")
	aed.register("parent","anne","tom")
	aed.register("parent","rob","tom")
	aed.register("female","cristi")
	#####################
	#   Register a rule
	#####################
	#
	aed.regrule("mother" , "X", "Y" , "female" , "X","X", "parent", "X", "Y")
	aed.regrule("father" , "X", "Y" , "male" , "X","X", "parent", "X", "Y")
	aed.regrule("eloped" , "X", "Y" , "father" , "X","Z", "mother", "Y", "Z")
	[aed.regrule(r,"X","Y" , r , "X", "Y" , r, "X" , "Y"   ) for r in relations]
	[aed.regrule(r,"X","X" , r , "X", "X" , r, "X" , "X"   ) for r in relations]
	#
	####
	# RUN!!!
	####
	aed.debug=True
	aed.printKnowledge()
	aed.run(2)
	aed.printKnowledge()
	aed.printRules()
