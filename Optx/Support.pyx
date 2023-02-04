from scipy import stats
from numpy import array,exp,zeros,abs,round,diff,flatnonzero,arange,inf
from numpy.random import normal,laplace
from numpy.lib.scimath import log,sqrt
from cpython.datetime cimport date,timedelta
cimport numpy as np

cpdef np.ndarray[np.float64_t,ndim=1] getpayoff(str optype,
                                                np.ndarray[np.float64_t,
                                                           ndim=1] s,
                                                double x):
    '''
    getpayoff(optype,s,x) -> returns the payoff of an option trade at expiration.
    
    Arguments:
    ----------    
    optype: option type (either 'call' or 'put').
    s: a numpy array of stock prices.
    x: strike price.
    '''
    if optype=="call":
        return (s-x+abs(s-x))/2.0
    elif optype=="put":
        return (x-s+abs(x-s))/2.0
    else:
        raise ValueError("Option type must be either 'call' or 'put'!")

cpdef np.ndarray[np.float64_t,ndim=1] getPLoption(str optype,
                                                  double opvalue,
                                                  str action,
                                                  np.ndarray[np.float64_t,
                                                             ndim=1] s,
                                                  double x):
    '''
    getPLoption(optype,opvalue,action,s,x) -> returns the profit (P) or loss 
    (L) per option of an option trade at expiration.
    
    Arguments:
    ----------
    optype: option type (either 'call' or 'put').
    opvalue: option price.
    action: either 'buy' or 'sell' the option.
    s: a numpy array of stock prices.
    x: strike price.
    '''
    if action=="sell":
        return opvalue-getpayoff(optype,s,x)
    elif action=="buy":
        return getpayoff(optype,s,x)-opvalue
    else:
        raise ValueError("Action must be either 'sell' or 'buy'!")
    
cpdef np.ndarray[np.float64_t,ndim=1] getPLstock(double s0,
                                                 str action,
                                                 np.ndarray[np.float64_t,
                                                            ndim=1] s):
    '''
    getPLstock(s0,action,s) -> returns the profit (P) or loss (L) of a stock
    position.
    
    Arguments:
    ----------
    s0: initial stock price.
    action: either 'buy' or 'sell' the stock.
    s: a numpy array of stock prices.
    '''
    if action=="sell":
        return s0-s
    elif action=="buy":
        return s-s0
    else:
        raise ValueError("Action must be either 'sell' or 'buy'!")
        
cpdef list getPLprofile(str optype,
                        str action,
                        double x,
                        double val,
                        int n,
                        np.ndarray[np.float64_t,ndim=1] s,
                        double commission=0.0):
    '''
    getPLprofile(optype,action,x,val,n,s,commision) -> returns the profit/loss 
    profile of an option trade at expiration, including the number of options 
    and commission.
    
    Arguments:
    ----------
    optype: option type ('call' or 'put').
    action: either 'buy' or 'sell' the option.
    x: strike price.
    val: option price.
    n: number of options.
    s: a numpy array of stock prices.
    comission: per transaction commission charged by the broker (0.0 is the 
               default).
    '''
    cdef double cost
    
    if action=="buy":
        cost=-val
    elif action=="sell":
        cost=val
    else:
        raise ValueError("Action must be either 'buy' or 'sell'!")

    if optype in ["call","put"]: 
        return [n*getPLoption(optype,val,action,s,x)-commission,
                n*cost+commission]
    else:
        raise ValueError("Option type must be either 'call' or 'put'!")

cpdef list getPLprofilestock(double s0,
                             str action,
                             int n,
                             np.ndarray[np.float64_t,ndim=1] s,
                             double commission=0.0):
    '''
    getPLprofilestock(s0,action,n,s,commission) -> returns the profit/loss 
    profile of a stock trade including number of shares and commission.
    
    Arguments:
    ----------
    s0: initial stock price.
    action: either 'buy' or 'sell' the shares.
    n: number of shares.
    s: a numpy array of stock prices.
    comission: per transaction commission charged by the broker (0.0 is the 
               default).
    '''
    cdef double cost
    
    if action=="buy":
        cost=-s0
    elif action=="sell":
        cost=s0
    else:        
        raise ValueError("Action must be either 'buy' or 'sell'!")
   
    return [n*getPLstock(s0,action,s)-commission,n*cost+commission]
        
cpdef list getPLprofileBS(str optype,
                          str action,
                          double x,
                          double val,
                          double r,
                          double targ2maturity,
                          double volatility,
                          int n,
                          np.ndarray[np.float64_t,ndim=1] s,
                          double commission=0.0):
    '''
    getPLprofileBS(optype,action,x,val,r,targ2maturity,volatility,n,s,
    commission) -> returns the profit/loss profile of an option trade at a 
    target date before maturity using the Black-Scholes model given a list of
    stock prices.
    
    Arguments:
    ----------
    optype: option type (either 'call' or 'put').
    action: either 'buy' or 'sell' the option.
    x: strike.
    val: actual option price.
    r: risk-free rate.
    targ2maturity: time remaining to maturity from the target date.
    volatility: annualized volatility of the underlying asset.
    n: number of options.
    s: a numpy array of stock prices.
    comission: per transaction commission charged by the broker (0.0 is the 
               default).
    '''
    cdef int fac
    cdef double cost
    
    if action=="buy":
        cost=-val
        fac=1
    elif action=="sell":
        cost=val
        fac=-1
    else:
        raise ValueError("Action must be either 'buy' or 'sell'!")  

    cdef np.ndarray[np.float64_t,ndim=1] d1=(log(s/x)+
                                             (r+volatility*volatility/2.0)*
                                             targ2maturity)/(volatility*
                                                             sqrt(targ2maturity))
    cdef np.ndarray[np.float64_t,ndim=1] calcprice
    
    if optype=="call":
        calcprice=round((s*stats.norm.cdf(d1)-x*exp(-r*targ2maturity)*
                         stats.norm.cdf(d1-volatility*sqrt(targ2maturity))),2)
    elif optype=="put":
        calcprice=round((x*exp(-r*targ2maturity)*
                         stats.norm.cdf(-d1+volatility*sqrt(targ2maturity))-
                                        s*stats.norm.cdf(-d1)),2)
    else:
        raise ValueError("Option type must be either 'call' or 'put'!")
        
    return [fac*n*(calcprice-val)-commission,n*cost+commission]

cpdef int getnonbusinessdays(date startdate,date enddate):
    '''
    getnonbusinessdays -> returns the number of non-business days between 
    the start and end date, both provided as date objects.
    
    Arguments
    ---------
    startdate: Initial date in the range.
    enddate: End date in the range.
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

cpdef np.ndarray[np.float64_t,ndim=1] getsequentialprices(double minprice,
                                                          double maxprice):
    '''
    getsequentialprices(minprice,maxprice) -> generates a sequence of numbers 
    from 'minprice' to 'maxprice' with increment 0.01.
    
    Arguments:
    ----------
    minprice: minimum stock price in the range.
    maxprice: maximum stock price in the range.
    '''
    if maxprice>minprice:
        return round((arange(int(maxprice-minprice)*100+1)*0.01+minprice),2)
    else:
        raise ValueError("Maximum price cannot be less than minimum price!")
      
cpdef np.ndarray[np.float64_t,ndim=1] getrandomprices(double s0,
                                                      double volatility,
                                                      double time2maturity,
                                                      double r=0.0,
                                                      str distribution="normal",
                                                      int n=100000):
    '''
    getrandomprices(s0,volatility,time2maturity,r,distribution,n) -> generates 
    a series of random prices of the stock according to a statistical 
    distribution.
    
    Arguments:
    ----------
    s0: spot price of the stock.
    volatility: annualized volatility.
    time2maturity: time left to maturity in units of year.
    r: annualized risk-free interest rate. Used only if distribution is 
       'normal-risk-neutral'.
    distribution: statistical distribution to be used when sampling prices of 
                  the stock at maturity. It can be 'normal' (default), 
                  'normal-risk-neutral' or 'laplace'.
    n: number of randomly generated prices.
    '''
    cdef double drift
       
    if distribution=="normal":
        return exp(normal(log(s0),volatility*sqrt(time2maturity),n))
    elif distribution=="normal-risk-neutral":
        drift=(r-0.5*volatility*volatility)*time2maturity
        
        return exp(normal((log(s0)+drift),volatility*sqrt(time2maturity),n))
    elif distribution=="laplace":
        return exp(laplace(log(s0),
                           (volatility*sqrt(time2maturity))/sqrt(2.0),n))
    else:
        raise ValueError("Distribution must be 'normal', 'normal-risk-neutral' or 'laplace'!")
                                    
cpdef list getprofitrange(np.ndarray[np.float64_t,ndim=1] s,
                          np.ndarray[np.float64_t,ndim=1] profit,
                          double target=0.01):
    '''
    getprofitrange(s,profit,target) -> returns pairs of stock prices, as a list,
    for which an option trade is expected to get the desired profit in between.
    
    Arguments:
    ----------
    s: a numpy array of stock prices.
    profit: a numpy array containing the profit (or loss) of the trade for each
            stock price in the stock price array.
    target: profit target ($0.01 is the default).
    '''
    cdef np.ndarray[np.float64_t,ndim=1] t
    cdef np.ndarray[np.longlong_t,ndim=1] maxi
    cdef np.ndarray[np.uint8_t,ndim=1,cast=True] mask1,mask2
    cdef list profitrange=[]     
    cdef Py_ssize_t i
    
    t=s[profit>=target]
    
    if t.shape[0]==0:
        return profitrange
    
    mask1=diff(t)<=target+0.001
    mask2=diff(t)>target+0.001
    maxi=flatnonzero(mask1[:-1] & mask2[1:])+1
    
    for i in range(maxi.shape[0]+1):
        profitrange.append([])
        
        if i==0:
            if t[0]==s[0]:
                profitrange[0].append(0.0)
            else:
                profitrange[0].append(t[0])
        else:
            profitrange[i].append(t[maxi[i-1]+1])
            
        if i==maxi.shape[0]:
            if t[t.shape[0]-1]==s[s.shape[0]-1]:
                profitrange[maxi.shape[0]].append(inf)
            else:
                profitrange[maxi.shape[0]].append(t[t.shape[0]-1])
        else:
            profitrange[i].append(t[maxi[i]])

    return profitrange

cpdef double getPoP(list profitranges,str source="normal",dict kwargs={}):
    '''
    getPoP(profitranges,source,kwargs) -> estimates the probability of 
    profit (PoP) of a trade. 
    
    Arguments:
    ----------
    profitranges: a Python list containing the ranges of stock prices, 
                  as given by 'getprofitrange()', for which a trade results 
                  in profit.    
    source: a string indicating the origin of data used to estimate the 
            probability of profit (see next).
    kwargs: a Python dictionary. The data that has to be provided depend on 
            the data source and are determined by setting the value of the  
            'source' argument:
    
            * For 'source="normal"' (default) or 'source="laplace"': the 
            probability is calculated assuming either a (log)normal or a 
            (log)Laplace distribution. The keywords 'stockprice', 'volatility' 
            and 'time2maturity' must be set.
            
            * For 'source="normal-risk-neutral"': the probability is calculated 
            assuming a (log)normal distribution with risk neutrality as 
            implemented in the Black-Scholes formula. The keywords 'stockprice', 
            'volatility', 'interestrate' and 'time2maturity' must be set.
    
            * For 'source="pricearray"' or 'source="mc"' or 
            'source="montecarlo"': the probability is calculated from a numpy 
            array of stock prices typically at maturity generated by a Monte 
            Carlo simulation; this numpy array has to be assigned to the 
            'pricearray' keyword.
    '''
    if not bool(kwargs):
        raise ValueError("'kwargs' is empty, nothing to do!")
            
    cdef Py_ssize_t i
    cdef double pop=0.0
    cdef double stockprice,volatility,r,time2maturity,sigma,beta,lval,hval
    cdef double drift=0.0
    cdef np.ndarray[np.float64_t,ndim=1] stocks,tmp1,tmp2
 
    if len(profitranges)==0:        
        return pop
                
    if source in ["normal","laplace","normal-risk-neutral"]:
        if "stockprice" in kwargs.keys():
            stockprice=float(kwargs["stockprice"])
                
            if stockprice<=0.0:
                raise ValueError("Stock price must be greater than zero!")                    
        else:
            raise ValueError("Stock price must be provided!")
                    
        if "volatility" in kwargs.keys():
            volatility=float(kwargs["volatility"])
            
            if volatility<=0.0:
                raise ValueError("Volatility must be greater than zero!")                    
        else:
            raise ValueError("Volatility must be provided!")
                    
        if "time2maturity" in kwargs.keys():
            time2maturity=float(kwargs["time2maturity"])
            
            if time2maturity<0.0:
                raise ValueError("Time left to expiration must be a positive number!")                    
        else:
            raise ValueError("Time left to expiration must be provided!")
            
        if source=="normal-risk-neutral":
            if "interestrate" in kwargs.keys():
                r=float(kwargs["interestrate"])
                
                if r<0.0:
                    raise ValueError("Risk-free interest rate must be a positive number!")
            else:
                raise ValueError("Risk-free interest rate must be provided!")
                
            drift=(r-0.5*volatility*volatility)*time2maturity
                    
        sigma=volatility*sqrt(time2maturity)
               
        if sigma==0.0:
            sigma=1e-10
            
        if source=="laplace":
            beta=sigma/sqrt(2.0)
                    
        for i in range(len(profitranges)):
            lval=profitranges[i][0]
            
            if lval<=0.0:
                lval=1e-10
            
            hval=profitranges[i][1]
            
            if source in ["normal","normal-risk-neutral"]:
                pop+=(stats.norm.cdf((log(hval/stockprice)-drift)/sigma)-
                      stats.norm.cdf((log(lval/stockprice)-drift)/sigma))
            else:
                pop+=(stats.laplace.cdf(log(hval/stockprice)/beta)-
                      stats.laplace.cdf(log(lval/stockprice)/beta))
    
    elif source in ["pricearray","mc","montecarlo"]:
        if "pricearray" in kwargs.keys():
            if len(kwargs["pricearray"])>0:
                stocks=kwargs["pricearray"]
            else:
                raise ValueError("The array of stock prices is empty!")
        else:
            raise ValueError("An array of stock prices must be provided!")
                            
        for i in range(len(profitranges)):
            tmp1=stocks[stocks>=profitranges[i][0]]
            tmp2=tmp1[tmp1<=profitranges[i][1]]
            pop+=tmp2.shape[0]
                        
        pop=pop/stocks.shape[0]
    else:
        raise ValueError("Data source not supported yet!")

    return pop