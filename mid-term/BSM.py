import math
import numpy as np
from scipy.stats import norm

class BSM():
    def __init__(self,s0,k,r,sigma,T):
        self.s0 = s0
        self.k = k
        self.r = r
        self.sigma = sigma
        self.T = T
        self.d1 = (np.log(s0/k)+(r+sigma**2/2)*T) / (sigma * np.sqrt(T))
        self.d2 = ((np.log(s0/k)+(r+sigma**2/2)*T) / (sigma * np.sqrt(T))) - sigma*np.sqrt(T)


    def BSPrice(self): # calculate price 
        c = self.s0*norm.cdf(self.d1) - self.k*np.exp(-self.r*self.T)*norm.cdf(self.d2)
        p = self.k*np.exp(-self.r*self.T)*norm.cdf(-self.d2) - self.s0*norm.cdf(-self.d1)
        return c,p
        
# The term "Greeks" refers to the current exposure of an option to different risks. 
# The numerical values of Greeks indicate how the price of the option strategy would change when facing various risks. 
# There are five types of risks, which are Delta, Gamma, Vega, Theta, and Rho. 
# Each represents exposure to price changes, changes in price change, volatility, time, and risk-free interest rate, respectively.

    def BSDelta(self): # When the price of the underlying asset increases by 1 unit, the profit or loss of the option strategy increases by Delta units. 
        cDelta = norm.cdf(self.d1)
        pDelta = norm.cdf(self.d1)-1
        return cDelta,pDelta
    
    def BSGamma(self): #Definition: When the price of the underlying asset increases by 1 unit, the Delta of the option strategy increases by Gamma units.
        cGamma = norm.pdf(self.d1)/(self.s0*self.sigma*np.sqrt(self.T))
        pGamma = norm.pdf(self.d1)/(self.s0*self.sigma*np.sqrt(self.T))
        return cGamma,pGamma
    
    def BSVega(self): # Definition: When the annualized implied volatility of the underlying asset increases by 1%, the profit or loss of the option strategy increases by Vega units.
        cVega = self.s0*np.sqrt(self.T)*norm.pdf(self.d1)
        pVega = self.s0*np.sqrt(self.T)*norm.pdf(self.d1)
        return  cVega,pVega
    
    def BSTheta(self): # Definition: With each passing day, the profit or loss of the option strategy decreases by Theta units.
        cTheta = -self.s0*norm.pdf(self.d1)*self.sigma / (2*np.sqrt(self.T)) - self.r*self.k*np.exp(-self.r*self.T)*norm.cdf(self.d2)
        pTheta = -self.s0*norm.pdf(self.d1)*self.sigma / (2*np.sqrt(self.T)) + self.r*self.k*np.exp(-self.r*self.T)*norm.cdf(-self.d2)
        return cTheta, pTheta
    
    def BSRho(self): 
        cRho = self.k*self.T*np.exp(-self.r*self.T) *norm.cdf(self.d2)
        pRho = -self.k*self.T*np.exp(-self.r*self.T)*norm.cdf(-self.d2)
        return cRho,pRho