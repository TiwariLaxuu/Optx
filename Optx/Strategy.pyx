from __future__ import print_function
from numpy import array,asarray,zeros,full,stack,isnan,savetxt
from numpy.lib.scimath import sqrt
cimport numpy as np
import json
from cpython.datetime cimport date,datetime,timedelta
from Optx.BlackScholes import (getBSinfo,
                               getimpliedvol)
from Optx.Support import (getPLprofile,
                          getPLprofilestock,
                          getPLprofileBS,
                          getprofitrange,
                          getsequentialprices,
                          getrandomprices,
                          getPoP)

cdef class Strategy:  
    cdef:
        readonly np.ndarray __s,__s_mc
        readonly np.ndarray profit,profit_mc,totprofit,totprofit_mc
        readonly list __strike,__premium,__n,__action,__type,__expiration,\
            __usebs,__profitranges,__profittargrange,__losslimitranges,\
            __prevpos,__days2maturity
        readonly list impvol,delta,gamma,vega,theta,itmprob,balance
        readonly int __days2target,__nmcprices
        readonly double __stockprice,__volatility,__r,__profittarg,__losslimit,\
            __optcommission,__stockcommission,__minstock,__maxstock,__daysinyear            
        readonly double totbalance,profitprob,profittargprob,losslimitprob
        readonly date __targetdate,__startdate
        readonly str __distribution
        readonly bint __compute_the_greeks,__compute_expectation,__use_dates,\
            __discard_nonbusinessdays
        
    def __init__(self):
        '''
        __init__ -> initializes class variables.

        Returns
        -------
        None.
        '''
        self.__s=array([])
        self.__s_mc=array([])
        self.__strike=[]
        self.__premium=[]
        self.__n=[]
        self.__action=[]
        self.__type=[]
        self.__expiration=[]
        self.__prevpos=[]
        self.__usebs=[]
        self.__profitranges=[]
        self.__profittargrange=[]
        self.__losslimitranges=[]
        self.__days2maturity=[]
        self.impvol=[]
        self.itmprob=[]
        self.delta=[]
        self.gamma=[]
        self.vega=[]
        self.theta=[]
        self.balance=[]
        self.__stockprice=-1.0
        self.__volatility=-1.0
        self.__startdate=date.today()
        self.__targetdate=self.__startdate
        self.__r=-1.0
        self.__profittarg=-1.0
        self.__losslimit=1.0
        self.__optcommission=0.0
        self.__stockcommission=0.0
        self.__minstock=-1.0
        self.__maxstock=-1.0
        self.__distribution="normal"
        self.totbalance=0.0
        self.profitprob=0.0
        self.profittargprob=0.0
        self.losslimitprob=0.0
        self.__days2target=30
        self.__nmcprices=100000
        self.__compute_expectation=True
        self.__use_dates=True
        self.__discard_nonbusinessdays=True
        self.__daysinyear=252.0
        
    cpdef double getbalance(self,int leg=-1):
        '''
        getbalance -> returns the money spent (debit) or collected (credit) for 
        either a leg or the whole strategy.
            
        Parameters
        ----------
        leg : integer, optional
            Index of the leg. The default is -1 (whole strategy).
            
        Returns
        -------
        balance : double
            Money spent or collected for either a leg or the whole strategy.
        '''
        if len(self.balance)>0 and leg>=0 and leg<len(self.balance):
            return self.balance[leg]
        else:
            return self.totbalance
        
    cpdef list getPL(self,int leg=-1):
        '''
        getPL -> returns a numpy array of stock prices and a numpy array of 
        profits/losses of either a leg or the whole strategy.
            
        Parameters
        ----------
        leg : integer, optional
            Index of the leg. The default is -1 (whole strategy).
            
        Returns
        -------
        profits/losses : tuple
            Stock prices and profits/losses of either a leg or the whole 
            strategy.        
        '''
        if self.profit.size>0 and leg>=0 and leg<self.profit.shape[0]:
            return [self.__s,self.profit[leg]]
        else:
            return [self.__s,self.totprofit]
        
    cpdef (double,double,double,double,double) getavgPLfromMC(self):
        '''
        getavgPLfromMC -> returns the average (expected) profit and loss 
        assuming terminal stock prices from Monte Carlo simulations.

        Returns
        -------
        average results : tuple
            Average profit, standard error of the profit, average loss,
            standard error of the loss, and probability of profit.
        '''        
        cdef double avgprofit=0.0,avgloss=0.0,stderrprofit=0.0,stderrloss=0.0,\
            pop=0.0
        
        if self.totprofit_mc.shape[0]>0:
            avgprofit=self.totprofit_mc[self.totprofit_mc>=0.01].mean()
            avgloss=self.totprofit_mc[self.totprofit_mc<0.01].mean()
            stderrprofit=self.totprofit_mc[self.totprofit_mc>=0.01].std()
            stderrprofit/=sqrt((self.totprofit_mc>=0.01).sum())
            stderrloss=self.totprofit_mc[self.totprofit_mc<0.01].std()
            stderrloss/=sqrt((self.totprofit_mc<0.01).sum())
            pop=(self.totprofit_mc>=0.01).sum()/self.totprofit_mc.shape[0]
                
            if isnan(avgprofit):
                avgprofit=0.0
                    
            if isnan(avgloss):
                avgloss=0.0
                
            if isnan(stderrprofit):
                stderrprofit=0.0
                
            if isnan(stderrloss):
                stderrloss=0.0
            
            if isnan(pop):
                pop=0.0
        
        return avgprofit,stderrprofit,avgloss,stderrloss,pop

    cpdef (double,double,double,double,double,double) getBSresults(self,
                                                                   int leg):
        '''
        getBSresults -> returns the implied volatility, the ITM probability, 
        and the Greeks Delta, Gamma, Theta and Vega for the specified leg in 
        the strategy.
            
        Parameters
        ----------
        leg : integer
            Index of the leg.
            
        Returns
        -------
        BS results : tuple
            Implied volatility, ITM probability, and the Greeks Delta, Gamma, 
            Theta and Vega for the specified leg in the strategy.
        '''
        if leg>=0 and leg<len(self.impvol):
            return (self.impvol[leg],self.itmprob[leg],self.delta[leg],
                    self.gamma[leg],self.theta[leg],self.vega[leg])
        else:
            raise ValueError("Invalid leg!")
        
    cpdef (double,double,double) getprobabilities(self):
        '''
        getprobabilities -> returns the the probabilities calculated from the 
        chosen distribution.
        
        Parameters
        ----------
        None.
        
        Returns
        -------
        probabilities : tuple
            Calculated probabilities.
        '''        
        return self.profitprob,self.profittargprob,self.losslimitprob
    
    cpdef list getprofitbounds(self):
        '''
        getprofitbounds -> returns the lower and upper bounds for every stock 
        price range for which the strategy is profitable.
        
        Parameters
        ----------
        None.
        
        Returns
        -------
        profit bounds : list
            List of lower and upper bounds in stock price profitable ranges.        
        '''
        return self.__profitranges
    
    cpdef (double,double) getmaxPL(self):
        '''
        getmaxPL -> returns the maximum loss and maximum profit of the 
            strategy in the range of stock prices considered in the 
            calculations.
        
        Parameters
        ----------
        None.
        
        Returns
        -------
        maximum P/L : tuple
            Maximum loss and maximum profit of the strategy.
        '''
        if self.totprofit.shape[0]>0:
            return self.totprofit.min(),self.totprofit.max()
        else:
            return 1.0,-1.0
    
    cpdef str dumpjsoninput(self):
        '''
        dumpjsoninput -> converts input to JSON format.
        
        Parameters
        ----------
        None.
        
        Returns
        -------
        jstr : string
            JSON string.
        '''
        cdef Py_ssize_t i
        cdef str jstr
        cdef list s=[]
        cdef list s_mc=[]
        cdef list expirationstr=[]
        
        if self.__s.shape[0]>0:
            s=self.__s.tolist()
            
        if self.__s_mc.shape[0]>0:
            s_mc=self.__s_mc.tolist()
        
        for i in range(len(self.__expiration)):
            expirationstr.append(self.__expiration[i].strftime("%Y-%m-%d"))
        
        jstr=json.dumps({
            "StockPrice":self.__stockprice,
            "Volatility":self.__volatility,
            "StartDate":self.__startdate.strftime("%Y-%m-%d"),
            "TargetDate":self.__targetdate.strftime("%Y-%m-%d"),
            "DaysToTargetDate":self.__days2target,
            "InterestRate":self.__r,
            "StockPriceRangeBounds":[self.__minstock,self.__maxstock],
            "ListOfStockPrices":s,
            "OptionCommission":self.__optcommission,
            "StockCommission":self.__stockcommission,
            "ProfitTarget":self.__profittarg,
            "LossLimit":self.__losslimit,
            "Type":self.__type,
            "Strike":self.__strike,
            "Premium":self.__premium,
            "Expiration":expirationstr,
            "DaysToMaturity":self.__days2maturity,
            "N":self.__n,
            "Action":self.__action,
            "PreviousPosition":self.__prevpos,
            "Distribution":self.__distribution,
            "NumberOfMCPrices":self.__nmcprices,
            "ListOfStockPricesFromMC":s_mc,
            "ComputeTheGreeks":self.__compute_the_greeks,
            "ComputeExpectationProfitAndLoss":self.__compute_expectation,
            "UseDatesInCalculations":self.__use_dates,
            "DiscardNonBusinessDays":self.__discard_nonbusinessdays,
            },indent=4,sort_keys=True)
        
        return jstr
    
    cpdef str dumpjsonoutput(self):
        '''
        dumpjsonoutput -> converts output to JSON format.
        
        Parameters
        ----------
        None.
        
        Returns
        -------
        jstr : string
            JSON string.
        '''
        cdef str jstr
        
        jstr=json.dumps({
            "ImpliedVolatility":self.impvol,
            "InTheMoneyProbability":self.itmprob,
            "Delta":self.delta,
            "Gamma":self.gamma,
            "Theta":self.theta,
            "Vega":self.vega,
            "ProfitRanges":self.__profitranges,
            "ProfitTargetRanges":self.__profittargrange,
            "StrategyBalance":self.totbalance,
            "PerLegBalance":self.balance,
            "ProbabilityOfProfit":self.profitprob,
            "ProbabilityOfProfitTarget":self.profittargprob,
            "ProbabilityOfLossLimit":self.losslimitprob,
            "AverageProfitAndLossFromMC":list(self.getavgPLfromMC()),
            "MaximumLossAndProfit":list(self.getmaxPL()),
            },indent=4,sort_keys=True)
        
        return jstr
    
    cpdef void getdata(self,double stockprice,
                       double volatility,
                       double interestrate,
                       double minstock,
                       double maxstock,
                       list strategy,
                       double profittarg=-1.0,
                       double losslimit=1.0,
                       double optcommission=0.0,
                       double stockcommission=0.0,
                       bint compute_the_greeks=True,
                       bint compute_expectation=True,
                       bint use_dates=True,
                       bint discard_nonbusinessdays=True,
                       str startdate="",
                       str targetdate="",
                       int days2targetdate=30,
                       str distribution="normal",
                       int nmcprices=100000,
                       double[:] s=array([]),
                       double[:] s_mc=array([])):
        '''
        getdata -> provides input data to a simulation directly from the 
        Python code.
        
        Parameters
        ----------
        stockprice : double
            Spot price of the underlying.
        volatility : double
            Annualized volatility.
        interestrate : double
            Annualized risk-free interest rate.
        minstock : double
            Minimum value of the stock in the simulated range.
        maxstock : double
            Maximum value of the stock in the simulated range.
        strategy : list
            A Python list that contains the strategy legs as Python dictionaries.
            For options, the dictionary should contain up to 7 keys:
                "type" : string
                    Either 'call' or 'put'. It is mandatory.
                "strike" : float
                    Option strike price. It is mandatory.
                "premium" : float
                    Option premium. It is mandatory.
                "n" : int
                    Number of options. It is mandatory
                "action" : string
                    Either 'buy' or 'sell'. It is mandatory.
                "prevpos" : float
                    Premium effectively paid or received in a previously open 
                    position. If positive, it means that the position remains 
                    open and the payoff calculation takes this price into 
                    account, not the current price of the option. If negative, 
                    it means that the position is closed and the difference 
                    between this price and the current price is considered in 
                    the payoff calculation.
                "expiration" : string | int
                    Expiration date in 'YYYY-MM-DD' format or number of days 
                    left before maturity, depending on the value in 'use_dates' 
                    (see below).
            For stocks, the dictionary should contain up to 4 keys:
                "type" : string
                    It must be 'stock'. It is mandatory.
                "n" : int
                    Number of shares. It is mandatory.
                "action" : string
                    Either 'buy' or 'sell'. It is mandatory.
                "prevpos" : double
                    Stock price effectively paid or received in a previously 
                    open position. If positive, it means that the position 
                    remains open and the payoff calculation takes this price 
                    into account, not the current price of the stock. If 
                    negative, it means that the position is closed and the 
                    difference between this price and the current price is 
                    considered in the payoff calculation.
        profittarg : double, optional
            Target profit level. Default is -1.0, which means it is not 
            calculated.
        losslimit : double, optional
            Limit loss level. Default is 1.0, which means it is not calculated.
        optcommission : double
            Broker commission for options transactions. Default is 0.0.
        stockcommission : double
            Broker commission for stocks transactions. Default is 0.0.
        compute_the_greeks : logical, optional
            Whether or not Black-Scholes formulas should be used to compute the 
            Greeks. The default is True.
        compute_expectation : logical, optional
            Whether or not the expected profit and loss must be 
            computed. The default is True.
        use_dates : logical, optional
            Whether the target and maturity dates are provided or then are the 
            days left to the target date and to maturity. The default is True.
        discard_nonbusinessdays : logical, optional
            Whether to discard Saturdays and Sundays from the days counting in 
            a range. The default is True.
        startdate : string
            Start date of the simulation, in 'YYYY-MM-DD' format. The Default 
            is "". Mandatory if 'use_dates' is True.
        targetdate : string
            Target date of the simulation, in 'YYYY-MM-DD' format. The Default 
            is "". Mandatory if 'use_dates' is True.
        days2targetdate : int, optional
            Number of days remaining until the target date. Not considered if 
            'use_dates' is True. The default is 30 days.
        distribution : string, optional
            Statistical distribution used to compute probabilities. It can be 
            'normal', 'normal-risk-neutral' or 'laplace'. Default is 'normal'.
        nmcprices : integer, optional
            Number of random prices to be generated when calculationg the 
            average profit and loss of a strategy. Default is 100,000.
        s : array, optional
            A numpy array containing sequential stock prices from a minimum 
            price up to a maximum price, with a 0.01 increment. Default is an 
            empyt array.
        s_mc : array, optional
            A numpy array of terminal stock prices generated by Monte Carlo 
            simulations used to compute the expected profit and loss. Default 
            is an empty array.
        
        Returns
        -------
        None.
        '''   
        cdef Py_ssize_t i
        cdef int days2maturitytmp,ndiscardeddays
        cdef date startdatetmp,targetdatetmp,expirationtmp
        
        if len(strategy)==0:
            raise ValueError("No strategy provided!")
            
        self.__type=[]
        self.__strike=[]
        self.__premium=[]
        self.__n=[]
        self.__action=[]
        self.__prevpos=[]
        self.__expiration=[]
        self.__days2maturity=[]
        self.__usebs=[]

        self.__discard_nonbusinessdays=discard_nonbusinessdays
            
        if self.__discard_nonbusinessdays:
            self.__daysinyear=252.0
        else:
            self.__daysinyear=365.0
        
        if use_dates:
            try:
                startdatetmp=datetime.strptime(startdate,"%Y-%m-%d").date()
                targetdatetmp=datetime.strptime(targetdate,"%Y-%m-%d").date()
                
                if targetdatetmp>startdatetmp:
                    self.__startdate=startdatetmp
                    self.__targetdate=targetdatetmp
                    
                    if self.__discard_nonbusinessdays:
                        ndiscardeddays=self.getnonbusinessdays(self.__startdate,
                                                               self.__targetdate)
                    else:
                        ndiscardeddays=0
                    
                    self.__days2target=(self.__targetdate-
                                        self.__startdate).days-ndiscardeddays
                else:
                    raise ValueError("Start date cannot be after the target date!")
            except:
                print("Start date and target date must be provided in 'YYYY-MM-DD' format!")
                
                return
        else:
            self.__days2target=days2targetdate
            
        for i in range(len(strategy)):
            if "type" in strategy[i].keys():
                self.__type.append(strategy[i]["type"])
            else:
                raise KeyError("Key 'type' is missing!")
            
            if strategy[i]["type"] in ["call","put"]:
                if "strike" in strategy[i].keys():
                    self.__strike.append(float(strategy[i]["strike"]))
                else:
                    raise KeyError("Key 'strike' is missing!")
                    
                if "premium" in strategy[i].keys():
                    self.__premium.append(float(strategy[i]["premium"]))
                else:
                    raise KeyError("Key 'premium' is missing!")
                    
                if "n" in strategy[i].keys():
                    self.__n.append(int(strategy[i]["n"]))
                else:
                    raise KeyError("Key 'n' is missing!")
                    
                if "action" in strategy[i].keys():
                    self.__action.append(strategy[i]["action"])
                else:
                    raise KeyError("Key 'action' is missing!")
                    
                if "prevpos" in strategy[i].keys():
                    self.__prevpos.append(float(strategy[i]["prevpos"]))
                else:
                    self.__prevpos.append(0.0)
                    
                if "expiration" in strategy[i].keys():
                    if use_dates:
                        try:
                            expirationtmp=datetime.strptime(strategy[i]["expiration"],
                                                            "%Y-%m-%d").date()
                        except:
                            print("Expiration date must be provided in 'YYYY-MM-DD' format; input data may be corrupted!")
                    
                            return
                    else:
                        try:
                            days2maturitytmp=int(strategy[i]["expiration"])
                        except:
                            print("Days remaining to maturity must be provided as an integer!")
                            
                            return
                else:
                    if use_dates:
                        expirationtmp=self.__targetdate
                    else:
                        days2maturitytmp=self.__days2target
                
                if use_dates:
                    if expirationtmp>=self.__targetdate:
                        self.__expiration.append(expirationtmp)
                        
                        if self.__discard_nonbusinessdays:
                            ndiscardeddays=self.getnonbusinessdays(self.__startdate,
                                                                   expirationtmp)
                        else:
                            ndiscardeddays=0
                        
                        self.__days2maturity.append((expirationtmp-
                                                     self.__startdate).days-
                                                    ndiscardeddays)
                            
                        if expirationtmp==self.__targetdate:
                            self.__usebs.append(False)
                        else:
                            self.__usebs.append(True)
                    else:
                        raise ValueError("Expiration date must be after or equal to the target date!")
                else:
                    if days2maturitytmp>=self.__days2target:
                        self.__days2maturity.append(days2maturitytmp)
                        
                        if days2maturitytmp==self.__days2target:
                            self.__usebs.append(False)
                        else:
                            self.__usebs.append(True)
                    else:
                        raise ValueError("Days left to maturity must be greater than or equal to the number of days remaining to the target date!")                    
            elif strategy[i]["type"]=="stock":
                if "n" in strategy[i].keys():
                    self.__n.append(int(strategy[i]["n"]))
                else:
                    raise KeyError("Key 'n' is missing!")
                    
                if "action" in strategy[i].keys():
                    self.__action.append(strategy[i]["action"])
                else:
                    raise KeyError("Key 'action' is missing!")
                    
                if "prevpos" in strategy[i].keys():
                    self.__prevpos.append(float(strategy[i]["prevpos"]))
                else:
                    self.__prevpos.append(0.0)
                    
                self.__strike.append(0.0)
                self.__premium.append(0.0)
                self.__usebs.append(False)
                self.__days2maturity.append(-1)
                                   
                if use_dates:
                    self.__expiration.append(self.__targetdate)
                else:
                    self.__expiration.append(-1)
            else:
                raise ValueError("Type must be 'call', 'put' or 'stock'!")
                            
        self.__stockprice=stockprice
        self.__volatility=volatility
        self.__r=interestrate
        self.__minstock=minstock
        self.__maxstock=maxstock
        self.__profittarg=profittarg
        self.__losslimit=losslimit
        self.__optcommission=optcommission
        self.__stockcommission=stockcommission
        self.__distribution=distribution
        self.__nmcprices=nmcprices
        self.__compute_the_greeks=compute_the_greeks
        self.__compute_expectation=compute_expectation
        self.__use_dates=use_dates
        
        if s.shape[0]>0:
            self.__s=asarray(s)
            
        if s_mc.shape[0]>0:
            self.__s_mc=asarray(s_mc)
                    
    cpdef void getdatafromjson(self,str jsonstring):
        '''
        getdatafromjson -> reads the stock and otpions data from a JSON string.
        
        Parameters
        ----------
        jsonstring : string
            String in JSON format.
            
        Returns
        -------
        None.
        '''
        cdef dict d
        cdef list expirationtmp=[]
        cdef Py_ssize_t i
        cdef int ndiscardeddays
        
        self.__expiration=[]
        self.__days2maturity=[]
        self.__usebs=[]
        
        try:
            d=json.loads(jsonstring)
            
            if "UseDatesInCalculations" in d.keys():
                self.__use_dates=d["UseDatesInCalculations"]
            else:
                self.__use_dates=True
                
            if "DiscardNonBusinessDays" in d.keys():
                self.__discard_nonbusinessdays=d["DiscardNonBusinessDays"]
            else:
                self.__discard_nonbusinessdays=True
                
            if self.__discard_nonbusinessdays:
                self.__daysinyear=252.0
            else:
                self.__daysinyear=365.0
            
            self.__stockprice=float(d["StockPrice"])
            self.__volatility=float(d["Volatility"])
            
            if self.__use_dates:
                self.__startdate=datetime.strptime(d["StartDate"],
                                                     "%Y-%m-%d").date()
                self.__targetdate=datetime.strptime(d["TargetDate"],
                                                      "%Y-%m-%d").date()
                
                if self.__discard_nonbusinessdays:
                    ndiscardeddays=self.getnonbusinessdays(self.__startdate,
                                                           self.__targetdate)
                else:
                    ndiscardeddays=0
                
                self.__days2target=(self.__targetdate-
                                    self.__startdate).days-ndiscardeddays
                expirationtmp=d["Expiration"].copy()
            
                for i in range(len(expirationtmp)):
                    self.__expiration.append(datetime.strptime(expirationtmp[i],
                                                               "%Y-%m-%d").date())
                    
                    if self.__discard_nonbusinessdays:
                        ndiscardeddays=self.getnonbusinessdays(self.__startdate,
                                                               self.__expiration[i])
                    else:
                        ndiscardeddays=0
                    
                    self.__days2maturity.append((self.__expiration[i]-
                                                self.__startdate).days-
                                                ndiscardeddays)
                
                    if self.__expiration[i]==self.__targetdate:
                        self.__usebs.append(False)
                    else:
                        self.__usebs.append(True)
            else:
                self.__days2target=d["DaysToTargetDate"]
                self.__days2maturity=d["DaysToMaturity"].copy()
                
                for i in range(len(self.__days2maturity)):
                    if self.__days2maturity[i]==self.__days2target:
                        self.__usebs.append(False)
                    else:
                        self.__usebs.append(True)
                
            self.__r=float(d["InterestRate"])
            self.__minstock=float(d["StockPriceRangeBounds"][0])
            self.__maxstock=float(d["StockPriceRangeBounds"][1])
            
            if "OptionCommission" in d.keys():
                self.__optcommission=float(d["OptionCommission"])
            else:
                self.__optcommission=0.0
                
            if "StockCommission" in d.keys():
                self.__stockcommission=float(d["StockCommission"])
            else:
                self.__stockcommission=0.0
                
            if "ProfitTarget" in d.keys():
                self.__profittarg=float(d["ProfitTarget"])
            else:
                self.__profittarg=-1.0
                
            if "LossLimit" in d.keys():
                self.__losslimit=float(d["LossLimit"])
            else:
                self.__losslimit=1.0
                
            self.__type=d["Type"].copy()
            self.__strike=d["Strike"].copy()
            self.__premium=d["Premium"].copy()
            self.__n=d["N"].copy()
            self.__action=d["Action"].copy()
            self.__prevpos=d["PreviousPosition"].copy()
            self.__compute_the_greeks=d["ComputeTheGreeks"]
            self.__compute_expectation=d["ComputeExpectationProfitAndLoss"]
            self.__distribution=d["Distribution"]
            self.__nmcprices=d["NumberOfMCPrices"]
                    
            if len(d["ListOfStockPrices"])>0 and \
                ((int(max(d["ListOfStockPrices"])-
                      min(d["ListOfStockPrices"]))*100+1)==
                 len(d["ListOfStockPrices"])):
                self.__s=asarray(d["ListOfStockPrices"])
                
            if len(d["ListOfStockPricesFromMC"])>0:
                self.__s_mc=asarray(d["ListOfStockPricesFromMC"])
        except:
            print("Malformed JSON string, check if there are input data missing!")
                    
    cpdef void run(self):
        '''
        run -> runs a simulation that computes the payoff and probabilites
            of profit of an options trading strategy.
            
        Returns
        -------
        None.
        '''
        if len(self.__type)==0:
            raise RuntimeError("No legs in the strategy! Nothing to do!")
            
        if not self.__distribution in ["normal","laplace","normal-risk-neutral",
                                       "mc","montecarlo"]:
            raise ValueError("Invalid distribution!")
            
        if self.__distribution in ["mc","montecarlo"] and \
            self.__s_mc.shape[0]==0:
            raise RuntimeError("No terminal stock prices from MC simulations! Nothing to do!")
                        
        cdef Py_ssize_t i
        cdef double time2maturity,stockpos,opval,balancetmp
        cdef double time2target=self.__days2target/self.__daysinyear
        cdef double calldelta,putdelta,calltheta,puttheta,gamma,vega,\
            callitmprob,putitmprob
        
        if self.__s.shape[0]==0:
            self.__s=getsequentialprices(self.__minstock,self.__maxstock)

        self.profit=zeros((len(self.__type),self.__s.shape[0]))            
        self.totprofit=zeros(self.__s.shape[0])                   
        
        if self.__compute_expectation and self.__s_mc.shape[0]==0:             
            self.__s_mc=getrandomprices(self.__stockprice,
                                        self.__volatility,
                                        time2target,
                                        self.__r,
                                        self.__distribution,
                                        self.__nmcprices)

        if self.__s_mc.shape[0]>0:                
            self.profit_mc=zeros((len(self.__type),self.__s_mc.shape[0]))
            self.totprofit_mc=zeros(self.__s_mc.shape[0])
                        
        for i in range(len(self.__type)):
            self.balance.append(0.0)
                     
            if self.__type[i] in ["call","put"]:
                if self.__compute_the_greeks and self.__prevpos[i]>=0.0:
                    time2maturity=self.__days2maturity[i]/self.__daysinyear                    
                    calldelta,putdelta,calltheta,puttheta,gamma,vega,\
                        callitmprob,putitmprob=\
                        getBSinfo(self.__stockprice,self.__strike[i],
                                  self.__r,self.__volatility,
                                  time2maturity)[2:]               
     
                    self.gamma.append(gamma)
                    self.vega.append(vega)
                                    
                    if self.__type[i]=="call":                                    
                        self.impvol.append(getimpliedvol("call",
                                                         self.__premium[i],
                                                         self.__stockprice,
                                                         self.__strike[i],
                                                         self.__r,
                                                         time2maturity))
                        self.delta.append(calldelta)
                        self.itmprob.append(callitmprob)
                        
                        if self.__action[i]=="buy":
                            self.theta.append(calltheta/self.__daysinyear)
                        else:
                            self.theta.append(-calltheta/self.__daysinyear)
                    else:
                        self.impvol.append(getimpliedvol("put",
                                                         self.__premium[i],
                                                         self.__stockprice,
                                                         self.__strike[i],
                                                         self.__r,
                                                         time2maturity))
                        self.delta.append(putdelta)
                        self.itmprob.append(putitmprob)
                        
                        if self.__action[i]=="buy":
                            self.theta.append(puttheta/self.__daysinyear)
                        else:
                            self.theta.append(-puttheta/self.__daysinyear)
                else:
                    self.impvol.append(0.0)
                    self.itmprob.append(0.0)
                    self.delta.append(0.0)
                    self.gamma.append(0.0)
                    self.vega.append(0.0)
                    self.theta.append(0.0)
                    
                if self.__prevpos[i]<0.0: # Previous position is closed
                    balancetmp=(self.__premium[i]+
                                self.__prevpos[i])*self.__n[i]
                    
                    if self.__action[i]=="buy":
                        balancetmp*=-1.0
                        
                    self.balance[i]=balancetmp
                    self.profit[i]=balancetmp
                    
                    if self.__compute_expectation:
                        self.profit_mc[i]=balancetmp
                else:
                    if self.__prevpos[i]>0.0: # Premium of the open position
                        opval=self.__prevpos[i]
                    else:   # Current premium
                        opval=self.__premium[i]
                    
                    if self.__usebs[i]:
                        self.profit[i],self.balance[i]=\
                            getPLprofileBS(self.__type[i],
                                           self.__action[i],
                                           self.__strike[i],
                                           opval,
                                           self.__r,
                                           (self.__days2maturity[i]-
                                            self.__days2target)/self.__daysinyear,
                                           self.__volatility,
                                           self.__n[i],
                                           self.__s,
                                           self.__optcommission)
                        
                        if self.__compute_expectation or \
                            self.__distribution in ["mc","montecarlo"]:
                            self.profit_mc[i]=\
                                getPLprofileBS(self.__type[i],
                                               self.__action[i],
                                               self.__strike[i],
                                               opval,
                                               self.__r,
                                               (self.__days2maturity[i]-
                                                self.__days2target)/self.__daysinyear,
                                               self.__volatility,
                                               self.__n[i],
                                               self.__s_mc,
                                               self.__optcommission)[0]
                    else:                       
                        self.profit[i],self.balance[i]=\
                            getPLprofile(self.__type[i],
                                         self.__action[i],
                                         self.__strike[i],
                                         opval,
                                         self.__n[i],
                                         self.__s,
                                         self.__optcommission)
                        
                        if self.__compute_expectation or \
                            self.__distribution in ["mc","montecarlo"]:
                            self.profit_mc[i]=\
                                getPLprofile(self.__type[i],
                                             self.__action[i],
                                             self.__strike[i],
                                             opval,
                                             self.__n[i],
                                             self.__s_mc,
                                             self.__optcommission)[0]
                            
                    if self.__prevpos[i]>0.0:
                        balancetmp=(self.__premium[i]-
                                    self.__prevpos[i])*self.__n[i]
                        
                        if self.__action[i]=="sell":
                            balancetmp*=-1.0
                            
                        self.balance[i]=balancetmp                      
            else:
                self.impvol.append(0.0)
                self.itmprob.append(1.0)
                self.delta.append(1.0)
                self.gamma.append(0.0)
                self.vega.append(0.0)
                self.theta.append(0.0)
                                
                if self.__prevpos[i]<0.0: # Previous position is closed
                    balancetmp=(self.__stockprice+
                                self.__prevpos[i])*self.__n[i]
                        
                    if self.__action[i]=="buy":
                        balancetmp*=-1.0                    
                    
                    self.balance[i]=balancetmp
                    self.profit[i]=balancetmp
                    
                    if self.__compute_expectation:
                        self.profit_mc[i]=balancetmp
                else:
                    if self.__prevpos[i]>0.0: # Stock price at previous position
                        stockpos=self.__prevpos[i]
                    else:   # Spot price of the stock at start date
                        stockpos=self.__stockprice
                
                    self.profit[i],self.balance[i]=\
                        getPLprofilestock(stockpos,
                                          self.__action[i],
                                          self.__n[i],
                                          self.__s,
                                          self.__stockcommission)
                    
                    if self.__compute_expectation or \
                        self.__distribution in ["mc","montecarlo"]:
                        self.profit_mc[i]=\
                            getPLprofilestock(stockpos,
                                              self.__action[i],
                                              self.__n[i],
                                              self.__s_mc,
                                              self.__stockcommission)[0]
                        
                    if self.__prevpos[i]>0.0:
                        balancetmp=(self.__stockprice-
                                    self.__prevpos[i])*self.__n[i]
                        
                        if self.__action[i]=="sell":
                            balancetmp*=-1.0
                            
                        self.balance[i]=balancetmp
            
            self.totprofit+=self.profit[i]
            self.totbalance+=self.balance[i]
            
            if self.__compute_expectation or \
                self.__distribution in ["mc","montecarlo"]:
                self.totprofit_mc+=self.profit_mc[i]
            
        self.__profitranges=getprofitrange(self.__s,self.totprofit)
        
        if self.__profitranges:
            if self.__distribution in ["normal","laplace","normal-risk-neutral"]:           
                self.profitprob=getPoP(self.__profitranges,
                                       self.__distribution,
                                       {"stockprice":self.__stockprice,
                                        "volatility":self.__volatility,
                                        "time2maturity":time2target,
                                        "interestrate":self.__r})
            elif self.__distribution in ["mc","montecarlo"]:
                self.profitprob=getPoP(self.__profitranges,
                                       self.__distribution,
                                       {"pricearray":self.__s_mc})

        if self.__profittarg>=0.0:
            self.__profittargrange=getprofitrange(self.__s,self.totprofit,
                                                  self.__profittarg)
            
            if self.__profittargrange:
                if self.__distribution in ["normal","laplace",
                                           "normal-risk-neutral"]:                    
                    self.profittargprob=getPoP(self.__profittargrange,
                                               self.__distribution,
                                               {"stockprice":self.__stockprice,
                                                "volatility":self.__volatility,
                                                "time2maturity":time2target,
                                                "interestrate":self.__r})
                elif self.__distribution in ["mc","montecarlo"]:
                    self.profittargprob=getPoP(self.__profittargrange,
                                               self.__distribution,
                                               {"pricearray":self.__s_mc})

        if self.__losslimit<0.0:
            self.__losslimitranges=getprofitrange(self.__s,self.totprofit,
                                                  self.__losslimit+0.01)

            if self.__losslimitranges:
                if self.__distribution in ["normal","laplace",
                                           "normal-risk-neutral"]:
                    self.losslimitprob=1.0-getPoP(self.__losslimitranges,
                                                  self.__distribution,
                                                  {"stockprice":self.__stockprice,
                                                   "volatility":self.__volatility,
                                                   "time2maturity":time2target,
                                                   "interestrate":self.__r})
                elif self.__distribution in ["mc","montecarlo"]:
                    self.losslimitprob=1.0-getPoP(self.__losslimitranges,
                                                  self.__distribution,
                                                  {"pricearray":self.__s_mc})

    cpdef int getnonbusinessdays(self,date startdate,date enddate):
        '''
        getnonbusinessdays -> returns the number of non-business days between 
        the start and end date.
        
        Parameters
        ----------
        startdate : date
            Initial date in the range.
        enddate : date
            End date in the range.
        
        Returns
        -------
        nonbusinessdays : int
            Number of non-business days in the range of dates.
        '''
        cdef int ndays=(enddate-startdate).days
        cdef int nonbusinessdays=0
        cdef date currdate
        cdef Py_ssize_t i
        
        if enddate<startdate:
            raise ValueError("End date must be after start date!")
        
        for i in range(ndays):
            currdate=startdate+timedelta(days=i)
            
            if currdate.weekday()>=5:
                nonbusinessdays+=1
                
        return nonbusinessdays
            
    cpdef void csvpayoff(self,str filename="payoff.csv",int leg=-1):
        '''
        csvpayoff -> saves the payoff data to a .csv file.

        Parameters
        ----------
        filename : string, optional
            Name of the .csv file. The default is 'payoff.csv'.
        leg : integer, optional
            Index of the leg. The default is -1 (whole strategy).

        Returns
        -------
        None.
        '''
        cdef np.ndarray[np.float64_t,ndim=2] arr
        
        if self.profit.size>0 and leg>=0 and leg<self.profit.shape[0]:
            arr=stack((self.__s,self.profit[leg]))
        else:
            arr=stack((self.__s,self.totprofit))
        
        savetxt(filename,arr.transpose(),delimiter=",",
                header="StockPrice,Profit/Loss")
        