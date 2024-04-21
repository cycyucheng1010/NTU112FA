import numpy as np
from scipy.stats import norm

class BlackScholesModel():
    def __init__(self, s0, k, term_structure, sigma, T):
        self.s0 = s0 # init price
        self.k = k  # start price
        self.sigma = sigma # volatility of assets price
        self.T = T #years
        self.r = term_structure.get_rate(T)   #term structure, a rate which follow the timeline
        self.d1 = (np.log(s0/k) + (self.r + sigma**2/2)*T) / (sigma * np.sqrt(T)) 
        self.d2 = self.d1 - sigma * np.sqrt(T)

    def BSPrice(self):  # calculate price
        c = self.s0 * norm.cdf(self.d1) - self.k * np.exp(-self.r * self.T) * norm.cdf(self.d2)
        p = self.k * np.exp(-self.r * self.T) * norm.cdf(-self.d2) - self.s0 * norm.cdf(-self.d1)
        return c, p

# The term "Greeks" refers to the current exposure of an option to different risks. 
# The numerical values of Greeks indicate how the price of the option strategy would change when facing various risks. 
# There are five types of risks, which are Delta, Gamma, Vega, Theta, and Rho. 
# Each represents exposure to price changes, changes in price change, volatility, time, and risk-free interest rate, respectively.


    def BSDelta(self):
        cDelta = norm.cdf(self.d1)
        pDelta = norm.cdf(self.d1) - 1
        return cDelta, pDelta

    def BSGamma(self):
        gamma = norm.pdf(self.d1) / (self.s0 * self.sigma * np.sqrt(self.T))
        return gamma  # Same for call and put

    def BSVega(self):
        vega = self.s0 * np.sqrt(self.T) * norm.pdf(self.d1)
        return vega  # Same for call and put

    def BSTheta(self):
        cTheta = -self.s0 * norm.pdf(self.d1) * self.sigma / (2 * np.sqrt(self.T)) - self.r * self.k * np.exp(-self.r * self.T) * norm.cdf(self.d2)
        pTheta = -self.s0 * norm.pdf(self.d1) * self.sigma / (2 * np.sqrt(self.T)) + self.r * self.k * np.exp(-self.r * self.T) * norm.cdf(-self.d2)
        return cTheta, pTheta

    def BSRho(self):
        cRho = self.k * self.T * np.exp(-self.r * self.T) * norm.cdf(self.d2)
        pRho = -self.k * self.T * np.exp(-self.r * self.T) * norm.cdf(-self.d2)
        return cRho, pRho