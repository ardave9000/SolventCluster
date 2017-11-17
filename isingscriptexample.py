from Ising import Species, Mixture
import pickle
import scipy.constants
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
k = scipy.constants.physical_constants['Boltzmann constant in eV/K'][0]
N = scipy.constants.N_A


pkl_f=open('datalib.pkl','rb')
dicts=pickle.load(pkl_f)
an_tuple=dicts['AN params']
dn_tuple=dicts['DN params']

def gutmann_curve(x,a,b,c,d):
    """Take in x-points and 4-elements a,b,c,d. Return a*x/(b+c*x) + d"""
    return a*x/(b+c*x)+d
def loaddata(fname):
	def processline(headers,s):
		new = s[:-1].split('\t')
		species = new[0]
		data = {}
		#floats = [float(new[i]) for i in range(1,len(new))]
		for i in range(1,len(new)):
			data[headers[i]] = float(new[i])
		#floats.append(new[0])
		return species,data

	length = sum(1 for line in open(fname))
	data = {}
	with open(fname) as f:
		headers = f.readline()[:-1].split('\t')
		for line in f:
			species,params = processline(headers,line)
			data[species] = Species(name = species, **params)
		return data
def plotEntropy(mix):
	# 15 moles = 1 liter of solvent. Assume 1M salt. All concentrations in mole fraction (i.e. c_salt = 1./16)
	entropy = []
	x = np.linspace(0,15,num=1000)
	for c1 in x:
		mix.c1 = c1/16
		mix.c2 = abs((15-c1)/16)
		mix.csalt = 1/17
		mix.IdealMixing()
		entropy.append((mix.S1+mix.S2+mix.S3))

	plt.plot(x/16,entropy)
	plt.xlabel("Mole fraction of solvent 1")
	plt.ylabel("Mixing Entropy (eV)")
	plt.show()
def nonlinear(mean_occs,mix,solutecharge):
	h = mix.Hamiltonians(mean_occs,solutecharge)
	def bexp(H,T):
		return math.exp(-H/(k*T))
	n,m,l = mean_occs
	e1,e2,e3 = bexp(h[0],mix.T), bexp(h[1],mix.T), bexp(h[2],mix.T)
	Z = e1+e2+e3
	return [n-e1/Z,m-e2/Z,l-e3/Z]


fname = 'species.txt'
species = loaddata(fname)
for specie in species:
	if specie != "Li":
		species[specie].calculateHcation(gutmann_curve,*dn_tuple)
	if specie != "PF6":
		species[specie].calculateHanion(gutmann_curve,*an_tuple)
	if species[specie].charge == 0:
		species[specie].Jii = -0.01
	if species[specie].charge != 0:
		species[specie].Jii = 0.1

#Plug in DFT values
species['EC'].hcation =-1.4
species['EMC'].hcation = -1.16
species['DMC'].hcation = -1.08
#species['PF6'].hcation = -1.3

concentrations = np.linspace(0.01,1,1000)
GLi = []
p1 = []
p2 = []
pion = []
#GF = []
for c1 in concentrations:
	mix = Mixture(solvent1 = species['EC'], solvent2 = species['DMC'], cation = species['Li'], anion = species['PF6'], c1 = c1,c2=(1.-c1), T=310)
	mix.IdealMixing()
	mix.totalmol=mix.c1+mix.c2+mix.csalt
	mix.Initialize()
	mix.J12 = -0.01
	#probLi=root(nonlinear,[mix.z*mix.c1/mix.totalmol,mix.z*mix.c2/mix.totalmol,0],(mix,1))
	probLi=root(nonlinear,[mix.c1/mix.totalmol*mix.z,mix.c2/mix.totalmol*mix.z,0],(mix,1))

	n,m,l=probLi.x
	p1.append(n)
	p2.append(m)
	pion.append(l)
	GLi.append(mix.z*(n*mix.solvent1.hcation+m*mix.solvent2.hcation+l*mix.anion.hcation)
	'''
	probF=root(nonlinear,[mix.c1,mix.c2,0],(mix,-1))
	p,q,r=probF.x
	GF.append(p*mix.solvent1.hanion+q*mix.solvent2.hanion+r*mix.cation.hanion)
	'''

# plt.plot(concentrations/1,GLi,'rx',label='Li+')
# #plt.plot(concentrations/16,GF,'b-',label='PF6-')
# plt.xlabel("Mole ratio of EC")
# plt.ylabel("Solvation G (eV)")
# plt.title("Li (1M of EC-to-DMC, ideal mix, h from DFT)")
# plt.legend()
# plt.show()
plt.plot(concentrations/1,p1,'ro',label='EC')
plt.plot(concentrations/1,p2,'bx',label='DMC')
plt.plot(concentrations/1,pion,'g^',label='PF6')
plt.xlabel("Mole ratio of EC")
plt.ylabel("Probability of occupation in shell (eV)")
plt.title("Probability in Li Shell (1M of EC-to-DMC, ideal mix, h from DFT)")
plt.legend()
plt.show()
