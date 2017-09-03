"""Analyze a portfolio."""

import pandas as pd
import numpy as np
import datetime as dt
import math
import matplotlib.pyplot as plt
"""
import os
retval = os.getcwd()
print "Directory changed successfully: %s" % retval

dirIn = "D:\\GeorgiaTech\\Mach Learn For Trading\\GitRepo\\ML4T_2017Fall\\assess_portfolio"

# Read in the directory
os.chdir(dirIn)

retval = os.getcwd()
print "Directory changed successfully: %s" % retval

import util
"""

from util import get_data, plot_data

# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality
def assess_portfolio(sd = dt.datetime(2008,1,1), ed = dt.datetime(2009,1,1), \
    syms = ['GOOG','AAPL','GLD','XOM'], \
    allocs=[0.1,0.2,0.3,0.4], \
    sv=1000000, rfr=0.0, sf=252.0, \
    gen_plot=False):

    

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later 

    # normalize allocation so it always sums to 1
    allocsRawSum = sum(allocs)
    allocs = [x / allocsRawSum for x in allocs]


    # Get daily portfolio value
    port_val = prices_SPY # add code here to compute daily portfolio values

    # compute cumulative return for series
    firstRowP = prices.iloc[:1,:]
    # cumulRet + 1
    cumulRet = prices/firstRowP.values[0,:] 
   
    # creating various cumulative values for the portfolio
    # transpose alloc wgts for use
    allocsPD = (pd.DataFrame(allocs)).transpose()
    
    # allocation wgted cum return    
    cumulRetAllWgtd = cumulRet*allocsPD.values[0,:]

    daily_port_cr = cumulRetAllWgtd.sum(axis=1)
    daily_port_val = daily_port_cr*sv


    #video method for daily return 
    dailyRet_2 = prices.copy()     
    dailyRet_2[1:] = (prices[1:]/prices[:-1].values) - 1    
    dailyRet_2.ix[0,:] = 0 
    # allocation wgted cum return    
    dailyRet_2Wtg = dailyRet_2*allocsPD.values[0,:]
    dailyRet_2 = dailyRet_2Wtg.sum(axis=1)
    dailyRet_2 = dailyRet_2[1:]
 

    # daily return for risk free rate
    dailyRet_2LessRF = [x - rfr for x in dailyRet_2]


    # compute various means and volatility
    # means
    mean_daily_ret2 =  np.mean(dailyRet_2)
    mean_daily_ret2_risk_free =  np.mean(dailyRet_2LessRF) 

    # Sample volatility
    stdev_daily_ret2 = np.std(dailyRet_2) #,ddof=1)
    stdev_daily_ret2_risk_free = np.std(dailyRet_2LessRF) #,ddof=1)


    # compute annualized sharpe ratio
    sharpe =  math.sqrt(sf)*mean_daily_ret2_risk_free/stdev_daily_ret2_risk_free    
    


    # Get portfolio statistics (note: std_daily_ret = volatility)
    # add code here to compute stats

    # assign values to expected return
    crV = daily_port_cr [-1] - 1.0
    #print "Cumulative Return: " + str(cr)

    srV = sharpe
    #print "Annulized Sharpe Ratio: " + str(sr)

    adrV = mean_daily_ret2
    #print "Average Daily Return: " + str(adr)

    sddrV = stdev_daily_ret2 
    #print "Standard Deviatn Daily Return: " + str(sddr)


    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        pricesWtg = prices*allocsPD.values[0,:]
        portPrice = pricesWtg.sum(axis=1)
        initalShrs = sv/portPrice[0]
        port_val = portPrice*initalShrs

        df_temp = pd.concat([port_val, prices_SPY], keys=['Portfolio', 'SPY'], axis=1)
        df_temp = df_temp/df_temp.values[0,:]
        ax = df_temp.plot(title = "Normalized Portfolio and SPY Prices")
        ax.set_xlabel("Date")
        ax.set_ylabel("Normalized Price")
       
        plt.show()

    # Add code here to properly compute end value
    ev = daily_port_val[-1]

    cr, adr, sddr, sr = [crV, adrV, sddrV, srV]
    return cr, adr, sddr, sr, ev

def test_code():
    # This code WILL NOT be tested by the auto grader
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!
    start_date = dt.datetime(2010,1,1)
    #start_date = dt.datetime(2010,6,1)

    end_date = dt.datetime(2010,12,31)
    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM']
    allocations = [0.2, 0.3, 0.4, 0.1]
    #symbols = ['AXP', 'HPQ', 'IBM', 'HNZ']
    #allocations = [0.0,0.0,0.0,1.0]
    start_val = 1000000  
    risk_free_rate = 0.0
    sample_freq = 252
    gen_plotV=False
   

    # Assess the portfolio
    cr, adr, sddr, sr, ev = assess_portfolio(sd = start_date, ed = end_date,\
        syms = symbols, \
        allocs = allocations,\
        sv = start_val, \
        gen_plot = gen_plotV)

    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Allocations:", allocations
    print "Sharpe Ratio:", sr
    print "Volatility (stdev of daily returns):", sddr
    print "Average Daily Return:", adr
    print "Cumulative Return:", cr

if __name__ == "__main__":
    test_code()
