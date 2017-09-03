"""Analyze a portfolio."""

import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
"""
# used on local Machine
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

    """
    # echo print for test
    print "SD: " + str(sd)
    print "ED: " + str(ed)
    print "Syms: " 
    for p in syms: print p
    print "Allocs: " + str(allocs)
    print "SV: " + str(sv)
    print "Risk Free Rate: " + str(rfr)
    print "Trading days/Sampling Freq: " + str(sf)
    print "Generate Plot: " + str(gen_plot)
    """

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later 

    # normalize allocation so it always sums to 1
    allocsRawSum = sum(allocs)
    allocs = [x / allocsRawSum for x in allocs]

    # print "Allocs after normalization: " + str(allocs)

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
    #dailyRet_2LessRF = [x - rfr for x in dailyRet_2]


    # compute various means 
    # means use pandas function pandas and numpy function give same results

    #mean_daily_ret2_np =  np.mean(dailyRet_2)
    mean_daily_ret2_pd =  dailyRet_2.mean()
    # seems to be an adjustment needed
    lenU = len(dailyRet_2)
    adj_mean_daily_ret2= lenU*mean_daily_ret2_pd/(lenU-1)
 

    # ignore risk free adjustment for present
    #mean_daily_ret2_risk_free_np =  np.mean(dailyRet_2LessRF) 
    #mean_daily_ret2_risk_free_pd =  dailyRet_2LessRF.mean() 

    """
  
    #alternative video method for daily return 
    # not used here
    dailyRet_2_alt = (prices/prices.shift(1))-1
    dailyRet_2_alt.ix[0,:] = 0
    # for computing mean daily return
    dailyRet_2_alt_for_mean = dailyRet_2_alt.ix[1:,:]
    dailyRet_2_alt_wgtd_sum = (dailyRet_2_alt_for_mean*allocsPD.values[0,:]).sum()
  
    dailyRet_2_alt_col_mean = dailyRet_2_alt_for_mean.mean(axis=0) 
    
    dailyRet_2_alt_mean = (dailyRet_2_alt_col_mean*allocsPD.values[0,:]).sum() 

    lenU = len(dailyRet_2_alt_for_mean)
    adj_dailyRet_2_alt_mean = lenU*dailyRet_2_alt_mean/(lenU-1)

    dailyRet_2 = dailyRet_2_alt_wgtd_sum[1:]
    mean_daily_ret2_pd = adj_dailyRet_2_alt_mean
    """


    # Sample volatility
    # Ignore sample adjustment as well as RF adjustment for present 
    # Use np function not pd function
    stdev_daily_ret2_np = np.std(dailyRet_2) #,ddof=1)
    #stdev_daily_ret2_risk_free_np = np.std(dailyRet_2LessRF) #,ddof=1)


    stdev_daily_ret2_pd = dailyRet_2.std()
    #stdev_daily_ret2_risk_free = dailyRet_2LessRF.std()


    # compute annualized sharpe ratio
    sharpe =  np.sqrt(sf)*adj_mean_daily_ret2/stdev_daily_ret2_np
    


    # Get portfolio statistics (note: std_daily_ret = volatility)
    # add code here to compute stats

    # assign values to expected return
    crV = daily_port_cr [-1] - 1.0
    # print "Cumulative Return: " + str(crV)

    srV = sharpe
    # print "Annulized Sharpe Ratio: " + str(srV)

    adrV = adj_mean_daily_ret2
    # print "Average Daily Return: " + str(adrV)

    sddrV = stdev_daily_ret2_np 
    # print "Standard Deviatn Daily Return: " + str(sddrV)


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
    #start_date = dt.datetime(2010,1,1)
    start_date = dt.datetime(2010,6,1)
    sd = start_date

    end_date = dt.datetime(2010,12,31)
    ed = end_date

    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM']  
    #symbols = ['AXP', 'HPQ', 'IBM', 'HNZ']
    syms = symbols

    allocations = [0.2, 0.3, 0.4, 0.1]
    #allocations = [0.0,0.0,0.0,1.0]
    allocs = allocations

    start_val = 1000000  
    sv = start_val

    risk_free_rate = 0.0
    rfr = risk_free_rate

    sample_freq = 252
    sf = sample_freq

    gen_plot=False
   

    # Assess the portfolio
    cr, adr, sddr, sr, ev = assess_portfolio(sd = start_date, ed = end_date,\
        syms = symbols, \
        allocs = allocations,\
        sv = start_val, rfr = risk_free_rate, \
        gen_plot = gen_plot)

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
