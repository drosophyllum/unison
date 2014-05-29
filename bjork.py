from numpy import *
import scipy.sparse
from regula import *

entities  = array(["anne" , "chris" ,"tom"])
relations = array(["father", "mother", "parent","male","female"])
rules = [("X","Y","X","Z","Y","Z"),("X","Y","X","X","X","Y"),("X","Y","X","Y","X","Y"),("X","X","X","X","X","X")]

correct = aedis(entities,relations,rules)
correct.register("female","anne","anne")
correct.register("male","chris","chris")
correct.register("parent","anne","tom")
correct.register("parent","chris","tom")
correct.regrule("mother" , "X", "Y" , "parent", "X", "Y","female","X","X")
correct.regrule("father" , "X", "Y" , "parent", "X", "Y","male" , "X","X")
[correct.regrule(r,"X","Y" ,r ,"X","Y",r,"X","Y") for r in relations]
[correct.regrule(r,"X","X" ,r ,"X","X",r,"X","X") for r in relations]

correct.printKnowledge()
correct.step()
correct.printKnowledge()
correct.printRules();

def mag(wx):
	return sum(sum(square(wx)))

x = aedis(entities,relations,rules)
x.A = 0.1*random.rand(correct.A.shape[0],correct.A.shape[1])
x.w = scipy.sparse.lil_matrix(0.1*random.rand(correct.w.shape[0],correct.w.shape[1]))
x.A0 = x.A
error = None;
epoch = 0;
momentum = 0.1
noise = 0
learningrate = 0.1
death = 0.1
beta  = 0.8
x.wv  = 0*x.w; 
x.A0v = 0*x.A0;
while(( error == None or error > 0.0)) :
	epoch += 1 	
	x.A = x.A0 
	A1 = x.step()
	d1 = multiply((correct.A - A1),(1-square(A1)))
	error = mag(correct.A-A1)
	print("Epoch " + str(epoch) + " : " + str(error) + " entries" )
	dF =  d1.dot(kron(x.A0,x.A0).transpose())
	print(dF.shape)
	print(x.T.shape)
	dw = (x.T * (dF.flatten().transpose())).transpose()
	dA0= zeros((x.A.shape[0],x.A.shape[0]))  
	for i in range(dA0.shape[0]):
		F = x.getForward()
		B = F[i,:].reshape((x.A.shape[0],x.A.shape[0]))
		B = B + B.transpose()
		dA0[i,:] =  B.dot(x.A0).transpose() 
	d0 = multiply(dA0.transpose().dot(d1) + beta*(2*(-1*divide(A1*0.5+0.5,correct.A*0.5 +0.5) + divide((1-0.5*A1+0.5),(1-0.5*correct.A+0.5))) -1), (1-square(x.A0)))

	print("weight speed: " + str(mag(dw)))
	print("A0 speed: " + str(mag(d0)))
	print("AO:")
	x.A = x.A0
	x.printKnowledge(0.1)
	print("Pred:")	
	x.step()
	x.printKnowledge(0.1)
	#purrtyPrint(atx)
	print("RULES:")	
	x.printRules( 0.1)
	print("###########################")

	x.wv = momentum*x.wv + dw
	x.w = x.w + learningrate*x.wv -death*x.w #+ noise*scipy.sparse.lil_matrix(random.randn(1,numrules))
	x.w[where(x.w<0)]=0
	x.w[where(x.w>1)]=1
	x.w = scipy.sparse.lil_matrix(x.w)
	x.A0v = momentum*x.A0v + d0
	x.A0 = x.A0 + x.A0v
	x.A0[where(x.A0<0)]=0
	x.A0[where(x.A0>1)]=1
	
