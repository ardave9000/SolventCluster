'''
This module defines classes for Mixture and Species

Mixture:
* Four constituent Species objects... 2x solvent (neutral); 2x ion (charged)
* CalculateEntropy() - calculates configurational entropy and adds to solvents/ions
* J_ij() - calculate interaction energy between each mixture component

Species:
* Solvent tagged by name, reads from solvent DB for parameters except h, S (entropy), and J_ii
'''
import numpy as np
import scipy.constants
import math

k = scipy.constants.physical_constants['Boltzmann constant in eV/K'][0]
N = scipy.constants.N_A

class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class InputError(Error):
    """Exception raised for errors in the input.
    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """
    def __init__(self, expression, message):
        self.expression = expression
        self.message = message

class Species:
	"""Solvent/ion object with many parameters, methods to set parameters'
	#name, dn, an, charge, di, ion_size, h, Jii, S
	#Species(name='EC',dn=16.4,an=18,charge=0, di=95.3, ion_size=4.23)
	"""
	def __init__(self,**kwargs):
		self.name = kwargs.pop('name', None)
		self.dn=kwargs.pop('dn',None)
		self.an=kwargs.pop('an',None)
		self.charge=kwargs.pop('charge',None)
		self.di=kwargs.pop('di',None)
		self.ion_size=kwargs.pop('ion_size',None)
		self.hcation=kwargs.pop('hcation',None)
		self.hanion=kwargs.pop('hanion',None)
		self.Jii=kwargs.pop('Jii',None)
		self.S=kwargs.pop('S',None)
		if len(kwargs) != 0:
			raise InputError('Species: __init__(self,**kwargs)','Spurious inputs into Species constructor present!')
	
	'''
	def __init__(self,*initialdata,**kwargs):
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])
		if len(kwargs) != 0:
			raise InputError('Mixture: __init__(self,**kwargs)','Spurious inputs into Mixture constructor present!')
	'''

	def calculateHcation(self,f,*args):
		if self.charge == None:
			raise InputError('Species: calculateH_cation(self,f,*args)','Define charge on species before calculating H terms!')
		elif self.charge > 0:
			raise InputError('Species: calculateH_cation(self,f,*args)','Calculation of cation-cation occupation requested. Currently not supported')
		elif self.dn == None:
			raise InputError('Species: calculateH_anion(self,f,*args)','Calculation of cation-occupation requires defined DN.')
		else:
			self.hcation=f(self.dn,*args)

	def calculateHanion(self,f,*args):
		if self.charge == None:
			raise InputError('Species: calculateH_anion(self,f,*args)','Define charge on species before calculating H terms!')
		elif self.charge < 0:
			raise InputError('Species: calculateH_anion(self,f,*args)','Calculation of anion-anion occupation requested. Currently not supported')
		elif self.an == None:
			raise InputError('Species: calculateH_anion(self,f,*args)','Calculation of anion-occupation requires defined AN.')
		else:
			self.hanion=f(self.an,*args)

class Mixture:
	"""Object of four constituent Species objects with related methods
	* CalculateEntropy() - calculates configurational entropy and adds to solvents/ions
	* J_ij() - calculate interaction energy between each mixture component
	"""
	def __init__(self,*initialdata,**kwargs):
		self.solvent1 = kwargs.pop('solvent1', None)
		self.solvent2 = kwargs.pop('solvent2', None)
		self.cation = kwargs.pop('cation', None)
		self.anion=kwargs.pop('anion', None)
		self.c1 = kwargs.pop('c1', 0)
		self.c2 = kwargs.pop('c2', 0)
		self.csalt = kwargs.pop('csalt', 1.)
		#self.c_cation = kwargs.pop('c_salt', 1)/2
		#self.c_anion = kwargs.pop('c_salt', 1)/2
		self.J12 = kwargs.pop('J12',None)
		self.J1c = kwargs.pop('J1c',None)
		self.J1a =  kwargs.pop('J1a',None)
		self.J2c =  kwargs.pop('J2c',None)
		self.J2a = kwargs.pop('J2a', None)
		self.T = kwargs.pop('T',298)
		self.z = kwargs.pop('z',2)
		self.S1 = kwargs.pop('S1',0)
		self.S2 = kwargs.pop('S2',0)
		self.S3 = kwargs.pop('S3',0)
		self.SL = kwargs.pop('SL',0)
		if len(kwargs) != 0:
			raise InputError('Mixture: __init__(self,**kwargs)','Spurious inputs into Mixture constructor present!')
	def Initialize(self):
		if self.c1==0:
			self.solvent1.hcation=0
			self.solvent1.hanion=0
		if self.c2==0:
			self.solvent2.hcation=0
			self.solvent2.hanion=0
		if self.csalt==0:
			self.cation.hanion=0
			self.anion.hcation=0
		self.J1c = self.solvent1.hcation
		self.J2c = self.solvent2.hcation
		self.J1a = self.solvent1.hanion
		self.J2a = self.solvent2.hanion
		if (self.J1c == None) or (self.J2c==None) or (self.J1a==None) or (self.J2a==None):
			raise InputError('Mixture: Initialize(self)','Initialize cannot complete due to missing h terms in solvent1 or solvent2')
	def IdealMixing(self):
		ln = math.log
		totalmol=self.c1+self.c2+self.csalt
		self.totalmol=totalmol
		if self.c1 != 0:
			self.S1= -k*self.T*self.c1/totalmol*ln(self.c1/totalmol)
		else:
			self.S1=0
		if self.c2 != 0:
			self.S2= -k*self.T*(self.c2)/totalmol*ln(self.c2/totalmol)
		else:
			self.S2=0
		if self.csalt != 0:
			self.S3= -k*self.T*self.csalt/totalmol*ln(self.csalt/totalmol)
		else:
			self.S3=0

	def Hamiltonians(self, mean_occs, solutecharge):
		n,m,l = mean_occs
		if solutecharge>0:
			h1,h2,h3 = self.solvent1.hcation,self.solvent2.hcation,self.anion.hcation
			j13,j23,j33 = self.solvent1.hanion,self.solvent2.hanion,self.anion.Jii
		if solutecharge<0:
			h1,h2,h3 = self.solvent1.hanion,self.solvent2.hanion,self.cation.hanion
			j13,j23,j33 = self.solvent1.hcation,self.solvent2.hcation,self.cation.Jii
		j11,j22 = self.solvent1.Jii, self.solvent2.Jii
		j12 = self.J12
		z = self.z
		def LatticeMixing(mix,mean_occs):
			ft = math.factorial
			ln = math.log
			fl = math.floor
			n,m,l=mean_occs
			A,B,C = fl(abs(n)),fl(abs(m)),fl(abs(l))
			Om = float(ft(mix.z))/(ft(A)*ft(B)*ft(C))
			mix.SL = ln(Om)
		#LatticeMixing(self,mean_occs)
		self.H1 = (h1+j11*z*n+j12*z/2*m+j13*z/2*l)-self.T*(self.S1)
		self.H2 = (h2+j22*z*m+j12*z/2*n+j23*z/2*l)-self.T*(self.S2)
		self.H3 = (h3+j33*z*l+j13*z/2*n+j23*z/2*m)-self.T*(self.S3)
		return self.H1,self.H2,self.H3

