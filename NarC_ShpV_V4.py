//+------------------------------------------------------------------+
//|                                                     NarC_ShpV.py |
//|                                                      Shiying Cui |
//+------------------------------------------------------------------+
#property copyright "Shiying Cui"
#property version   "4.0"
#property private!!!
from __future__ import division

import datetime
import math
# import numpy as np
# import pandas as pd
# import statsmodels.api as sm
# import statsmodels.tsa.stattools as ts
# from numpy.linalg import inv, eig, cholesky as chol
# from statsmodels.regression.linear_model import OLS
# from numpy import zeros, ones, flipud, log

from robot_trader.base_strategy import BaseStrategy, strategy_main
from strategy_plugin.cs_plugin import CandlestickPlugin
from utils.tick_info_db import t_round
from utils.types_enum.order_types import LIMIT
from utils.types_enum.valid_types import GTC
from utils.product_identifier import identify_product_name
from utils.main import datestr2date

from utils.trade_cal_hour import is_trading_hour, get_next_symbol, get_next_trade_session, get_futures_active_contract, \
    convert_product_name_2_dest_dt, get_symbol_expiry_day, get_order_trade_date, get_last_trade_session_on_trade_date, \
    get_first_trade_session_on_trade_date, \
    get_previous_trading_day, get_next_expiry_day, get_last_active_trade_session, get_expiry_trade_session


# Exponentially Weighted Moving Standand Variation
# recurrence formula 
class EStd:
    def __init__(self):
        self.PreXEma = None
        self.PreX2Ema = None
        self.value = None

    def CalEma(self, alpha, PreEma, newdata):
        if PreEma is None:
            Ema = newdata;
        else:
            Ema = alpha * newdata + (1 - alpha) * PreEma;
        return Ema

    def CalEStd(self, alpha, newdata):
        self.PreXEma = self.CalEma(alpha, self.PreXEma, newdata)
        self.PreX2Ema = self.CalEma(alpha, self.PreX2Ema, newdata ** 2)
        EStd = (self.PreX2Ema - (self.PreXEma) ** 2) ** 0.5;
        self.value = EStd
        return EStd


# Exponentially Weighted Moving Average
# recurrence formula
class EMA:
    def __init__(self):
        self.PreEma = None
        self.value = None

    def CalEma(self, alpha, newdata):
        if self.PreEma is None:
            self.PreEma = newdata;
        else:
            self.PreEma = alpha * newdata + (1 - alpha) * self.PreEma;
        Ema = self.PreEma
        self.value = Ema
        return Ema

# Function for True range
def CalTR(high, low, prelast):
    TH = max(high, prelast)
    TL = min(low, prelast)
    TR = TH - TL
    return TR

# An interface for recording narrow channel
class NC:
    def __init__(self, nc_time, upper_limit, lower_limit):
        self.nc_time = nc_time
        self.upper_limit = upper_limit
        self.lower_limit = lower_limit
        self.isBreakout = False
        self.entry_px_L = float('inf')
        self.entry_px_S = float('-inf')
        self.break_time = None
        self.isNCValid = True
        self.isBreakValid = True

# An interface for recording shape V
class ShpV:
    def __init__(self, leftV_time, upper_limit, lower_limit, V_or_A):
        self.leftV_time = leftV_time
        self.withdraw_Time = None
        self.upper_limit = upper_limit
        self.lower_limit = lower_limit
        self.V_or_A = V_or_A
        self.isleftVValid = True
        self.isWithdraw = False
        self.isWithdrawValid = True

# MMAE Function for stop loss 
def MaxFavorAndAdverExcur(oHH, oLL, oMFE, oMAE, EntryPrice, MP, H, L, IsPositionChange):
    if IsPositionChange:
        oHH = 0
        oLL = 999999
        oMFE = 0
        oMAE = 0
    if MP > 0:
        oHH = max(oHH, H)
        oLL = min(oLL, L)
        oMFE = oHH - EntryPrice
        oMAE = oLL - EntryPrice
    elif MP < 0:
        oLL = min(oLL, L)
        oHH = max(oHH, H)
        oMFE = EntryPrice - oLL
        oMAE = EntryPrice - oHH
    else:
        oHH = 0
        oLL = 999999
        oMFE = 0
        oMAE = 0
    return oHH, oLL, oMFE, oMAE


# -------------------------------------------------------------------------------------------------
# narrow channel breakout strategy:
# -------------------------------------------------------------------------------------------------
class NarChaBreak:
    def __init__(self, NCThr,NCDis,NCLen):
        self.NCThr = NCThr
        self.NCLen = NCLen
        self.ATRTimes = NCDis

        self.nc_valid_time = 0.2*self.ATRTimes*self.NCLen
        self.break_valid_time = 0.1*self.ATRTimes*self.NCLen
        self.stdLastPx = EStd()
        self.eATR = EMA()
        self.alpha = 2 / (self.NCLen + 1)
        self.len_effect = 1 / (self.NCLen**0.5)
        self.lasts = []
        self.highs = []
        self.lows = []
        self.vols = []
        self.NCs = []
        self.min_entry_L = float('+inf')
        self.max_entry_S = float('-inf')
        self.nc_indicator = None
        self.vol_indicator = None 
    def run(self, time, last, high, low, vol, ThrReady):

        # ------------------------------record terminal data-------------------------------------
        
        self.lasts.append(last)
        self.highs.append(high)
        self.lows.append(low)
        self.vols.append(vol)
        if len(self.lasts) > self.NCLen:
            del (self.lasts[0])
            del (self.highs[0])
            del (self.lows[0])
            del (self.vols[0])
        # Record history data 

        if len(self.lasts) >= 2 :
            prelast = self.lasts[-2]
            self.stdLastPx.CalEStd(self.alpha, last)
            self.eATR.CalEma(self.alpha, CalTR(high, low, prelast))
        # Calculate eATR and stdLastPx
        if len(self.lasts) < self.NCLen:
            return 0
        # If there is not enough data for eATR, or stdLastPx, skip the rest.
        
    
        if self.eATR.value > 0:
            nc_indicator = self.stdLastPx.value / self.eATR.value * self.len_effect
        else:
            nc_indicator = float('inf')
        self.nc_indicator = nc_indicator
        self.vol_indicator =  sum(self.vols) / self.NCLen
        # Calculate indicator using stdLastPx/eATR
        if not ThrReady:
            return

        # ---------------------------recognize narrow channel-------------------------------------
        
        print 'NCvol/min: %s' % self.vol_indicator
        # Set a threshold for volume
        #if nc_indicator < self.NCThr and temp > 250 and sum(self.vols[-5:])/5 > temp * 1.5:
        if nc_indicator < self.NCThr and self.vol_indicator > self.VolThr:
            nC = NC(time, max(self.highs), min(self.lows))
            self.NCs.append(nC)

        # ---------------------------recognize break and set entry price--------------------------
        min_entry_L = float('+inf')
        max_entry_S = float('-inf')
        for nC in self.NCs:
            if (time - nC.nc_time).seconds > self.nc_valid_time * 60:
                nC.isNCValid = False
            else:
                if high > nC.upper_limit:
                    nC.isBreakout = True
                    break_long_dis = self.eATR.value * self.ATRTimes
                    nC.entry_px_L = nC.upper_limit + break_long_dis
                    nC.break_time = time
                    # if breakout happened (upwards), set the threshold for entry 

                if low < nC.lower_limit:
                    nC.isBreakout = True
                    break_short_dis = self.eATR.value * self.ATRTimes
                    nC.entry_px_S = nC   .lower_limit - break_short_dis
                    nC.break_time = time
                    # if breakout happened (downwards), set the threshold for entry 

            if nC.break_time is not None and (time - nC.break_time).seconds > self.break_valid_time * 60:
                nC.isBreakValid = False

            if nC.isBreakout and nC.isBreakValid:
                min_entry_L = min((min_entry_L, nC.entry_px_L))
                max_entry_S = max((max_entry_S, nC.entry_px_S))

        self.min_entry_L = min_entry_L
        self.max_entry_S = max_entry_S

        temp = [nC for nC in self.NCs if (nC.isNCValid or (nC.isBreakout and nC.isBreakValid))]
        self.NCs = temp
        # Delete invalid narrow channels 
        print "len of NC lasts: %s" % len(self.lasts) 
        print "len of NC vols: %s" % len(self.vols)


# -------------------------------------------------------------------------------------------------
# shape V strategy: Different with the previous one
# -------------------------------------------------------------------------------------------------
class ShapeV:
    def __init__(self, ShpVThr, ShpVDis, ShpVLen):
        self.ShpVThr = ShpVThr
        self.ShpVDis = ShpVDis
        self.ShpVLen = ShpVLen 

        self.WithdrawValidTime = self.ShpVLen * ShpVDis * 2
        self.LeftVValidTime = self.ShpVLen

        self.stdLastPx = EStd()
        self.eATR = EMA()

        self.alpha = 2 / (self.ShpVLen + 1)
        self.len_effect = 1 / (self.ShpVLen**0.5)
        self.entry_L = float('inf')
        self.entry_S = float('-inf')
        self.lasts = []
        self.highs = []
        self.lows = []
        self.vols = []
        self.shpV = None
        self.shp_indicator = None
        self.vol_indicator = None

    def run(self, time, last, high, low, vol, ThrReady):
        
        self.lasts.append(last)
        self.highs.append(high)
        self.lows.append(low)
        self.vols.append(vol)
        if len(self.lasts) > self.ShpVLen:
            del (self.lasts[0])
            del (self.highs[0])
            del (self.lows[0])
            del (self.vols[0])
        # Record history data 

        if len(self.lasts) >= 2:
            prelast = self.lasts[-2]
            self.stdLastPx.CalEStd(self.alpha, last)
            self.eATR.CalEma(self.alpha, CalTR(high, low, prelast))
            #Calculate eATR and stdLastPx 
        if len(self.lasts) < self.ShpVLen:
            return  
            # If not enough data, skip the rest
    

        if self.eATR.value > 0:
            shp_indicator = self.stdLastPx.value / self.eATR.value * self.len_effect
        else:
            shp_indicator = float('-inf')
        self.shp_indicator = shp_indicator
        # Calculate shape V indicator
        self.vol_indicator = sum(self.vols)/self.ShpVLen

        if not ThrReady:
            return

        
        # Set an threshold for volume
        #if shp_indicator > self.ShpVThr and  temp > 250  and sum(self.vols[-5:])/5 > temp * 1.5:
        if shp_indicator > self.ShpVThr and  self.vol_indicator > self.VolThr:
            if high >= max(self.highs):
                self.shpV = ShpV(time, max(self.highs), min(self.lows), 'A')
            elif low <= min(self.lows):
                self.shpV = ShpV(time, max(self.highs), min(self.lows), 'V')
        
        # Recognize left A (sharp rise) or left V (sharp drop)

        if self.shpV is not None:
            if (self.shpV.V_or_A == 'A' and last < prelast) or (self.shpV.V_or_A == 'V' and last > prelast):
                self.shpV.isWithdraw = True
                self.shpV.withdraw_Time = time

        self.entry_L = float('inf')
        self.entry_S = float('-inf')

        if self.shpV is not None and self.shpV.withdraw_Time is not None:
            if (time - self.shpV.withdraw_Time).seconds > self.WithdrawValidTime * 60:
                self.shpV.isWithdrawValid = False
            elif self.shpV.V_or_A == 'V':
                self.entry_L = (1 - self.ShpVDis) * self.shpV.lower_limit + self.ShpVDis * self.shpV.upper_limit
            elif self.shpV.V_or_A == 'A':
                self.entry_S = (1 - self.ShpVDis) * self.shpV.upper_limit + self.ShpVDis * self.shpV.lower_limit
            # if recognize withdraw (right side of V or A), set an threshold price for entry

        if self.shpV is not None and (time - self.shpV.leftV_time).seconds > self.LeftVValidTime * 60:
            self.shpV = None

       


class NarC_ShpV(BaseStrategy):
    def init_daily(self):
        print 'init daily'

    def set_parameters(self, **kwargs):
        # print 'kwargs: %s' %kwargs
        self.narChaBreak = NarChaBreak(float(kwargs["NCThr"]), float(kwargs["NCDis"]), float(kwargs["NCLen"]))
        self.shapeV = ShapeV(float(kwargs["ShpVThr"]), float(kwargs["ShpVDis"]), float(kwargs["ShpVLen"]))
        self.TradeProduct = kwargs["TradeProduct"]
        self.TradeShares = float(kwargs["TradeShares"])

    def setup(self):
        print 'setting things up...'

        # loss control
        self.Pml_PT = 16
        self.Pml_TS = 16
        self.Pml_SL = 6
        self.MP = 0
        self.entry_price = None

        self.HHAE = float('-inf')
        self.LLAE = float('inf')
        self.MFE = float('-inf')
        self.MAE = float('-inf')
        self.SLRatio = self.Pml_SL * 0.001
        self.TSRatio = self.Pml_TS * 0.001
        self.PTRatio = self.Pml_PT * 0.001
        self.must_out = False
        self.can_in = True
        self.current_dest_date = None
        self.current_trade_session = None
        self.incoming_dest_dt = None
        self.pre_dt = None
        self.dt = None
        self.isMPChange = False

        self.lasts = []
        self.vols = []
        self.IndicatorsRecord = {"NC":[],"ShpV":[],"NCVol":[],"ShpVVol":[]}

    def _send_order(self, symbol, price, volume, remarks=None):
        if abs(volume) > 0:
            if volume > 0:
                p = t_round(self.current_dest_date, symbol, price, 50)
            elif volume < 0:
                p = t_round(self.current_dest_date, symbol, price, -50)
            self.order_submit(symbol, p, volume, LIMIT, GTC, False, remarks)

    def minuteLevel(self, time, high, low, last,vol, symbol, ThrReady):


        self.narChaBreak.run(time, last, high, low, vol,ThrReady)
        self.shapeV.run(time, last, high, low, vol, ThrReady)

        if self.must_out:
            self.shapeV.highs = []
            self.shapeV.lasts = []
            self.shapeV.lows = []
            self.shapeV.vols = []
            self.narChaBreak.highs = []
            self.narChaBreak.lows = []
            self.narChaBreak.lasts = []
            self.narChaBreak.vols = []
        

        print '-------------------------------%s--------------------------------------------'%time
        print 'last: %s'%last
        
        # print self.shapeV.ShpVLen
        # print self.shapeV.ShpVDis
        # print self.shapeV.ShpVThr
        # print 'alpha: %s' % self.shapeV.alpha
        # print 'len_effect: %s' % self.shapeV.len_effect
        print 'shp std: %s'%self.shapeV.stdLastPx.value
        print 'shp eATR: %s'%self.shapeV.eATR.value 
        print 'shp_indicator: %s' % self.shapeV.shp_indicator
        print 'nc_indicator: %s' % self.narChaBreak.nc_indicator
        print 'NCs size: %s' % len(self.narChaBreak.NCs)
        
        if self.shapeV.shpV is not None:
            print self.shapeV.shpV.leftV_time
            print self.shapeV.shpV.withdraw_Time
            print self.shapeV.shpV.V_or_A
            print self.shapeV.shpV.upper_limit
            print self.shapeV.shpV.lower_limit
 
        

        price = last
        condi_holiday = (self.next_trade_session[0] - time).days > 1 and (self.current_trade_session[
                                                                           1] - time).seconds < 60 * 60
        #print 'self.next_trade_session: %s' %self.next_trade_session[0]
        #print 'self.current_trade_session: %s'%self.current_trade_session[0]
        #print 'self.incoming_dest_dt: %s' %self.incoming_dest_dt

        # close position 60 minutes before a holiday come
        condi_expiry = (self.current_active_futures_expiry_day - time.date()).days <= 1
        # close position 1 day before contract expire
        time_10min_later = time + datetime.timedelta(seconds=60 * 60)
        self.next_active_hsi_future = get_futures_active_contract(self.TradeProduct, time_10min_later)
        condi_contract_change = self.next_active_hsi_future != self.current_active_hsi_futures
        # close position 60 minutes before contract change

        if condi_contract_change or condi_expiry or condi_holiday:
            self.must_out = True
        else:
            self.must_out = False
        # must_out means that we must close the position immediately;

        condi_holiday = (self.next_trade_session[0] - time).days > 1 and (self.current_trade_session[
                                                                           1] - time).seconds < 240 * 60
        # stop trade-in 4h before holiday
        condi_expiry = (self.current_active_futures_expiry_day - time.date()).days <= 1
        # stop trade-in 2 days before expire\
        time_120min_later = time + datetime.timedelta(seconds=120 * 60)
        self.next_active_hsi_future = get_futures_active_contract(self.TradeProduct, time_120min_later)
        condi_contract_change = self.next_active_hsi_future != self.current_active_hsi_futures
        # stop trade-in 120 minutes before contract change

        if condi_holiday or condi_expiry or condi_contract_change:
            self.can_in = False
        else:
            self.can_in = True
        # Can_in means that we allow trade-in; 
        # the combination for can_in can must_out consists time control for trading




        self.HHAE,self.LLAE,self.MFE,self.MAE = MaxFavorAndAdverExcur(self.HHAE,self.LLAE,self.MFE,self.MAE,self.entry_price,self.MP,high,low,self.isMPChange)
        # calculate variables for stop loss and stop profit


        action = False
        self.isMPChange = False
        # ----------------------------------------buy-------------------------------------
        isEntryLong = (last > self.narChaBreak.min_entry_L or last > self.shapeV.entry_L) and self.MP == 0
        isReverToLong = (last > self.narChaBreak.min_entry_L or last > self.shapeV.entry_L) and self.MP < 0
        isTimeCtrlCoverS = self.must_out and self.MP < 0
        isStopLossCoverS = self.entry_price is not None and last > self.entry_price * (1 + self.SLRatio) and self.MP < 0
        isStopProfitCoverS = self.entry_price is not None and self.MFE >= self.entry_price * self.PTRatio and last > self.LLAE + self.entry_price * self.TSRatio and self.MP < 0
        if (isEntryLong and self.can_in) or isTimeCtrlCoverS or isStopLossCoverS or isStopProfitCoverS or (
                    isReverToLong and not self.can_in):
            self._send_order(symbol, price, self.TradeShares)
            action = True
            print 'buy 1'
            
        elif isReverToLong and self.can_in:
            self._send_order(symbol, price, 2 * self.TradeShares)
            print 'buy 2'
            action = True

        

        # ----------------------------------------sell-------------------------------------
        isEntryShort = (last < self.narChaBreak.max_entry_S or last < self.shapeV.entry_S) and self.MP == 0
        isReverToShort = (last < self.narChaBreak.max_entry_S or last < self.shapeV.entry_S) and self.MP > 0
        isTimeCtrlSellL = self.must_out and self.MP > 0
        isStopLossSellL = self.entry_price is not None and last < self.entry_price * (1 - self.SLRatio) and self.MP > 0
        isStopProfitSellL = self.entry_price is not None and self.MFE >= self.entry_price * self.PTRatio and last < self.HHAE - self.entry_price * self.TSRatio and self.MP > 0
        if (isEntryShort and self.can_in) or isTimeCtrlSellL or isStopLossSellL or isStopProfitSellL or (
                    isReverToShort and not self.can_in):
            self._send_order(symbol, price, -1 * self.TradeShares)
            print 'sell 1'
            action = True
        elif isReverToShort and self.can_in:
            self._send_order(symbol, price, -2 * self.TradeShares)
            print 'sell 2'
            action = True

        if action:
            self.entry_price = price
            self.isMPChange = True
            self.MP = self._theoretical_position(self.current_active_hsi_futures)
            print 'entry_price: %s' %self.entry_price
            print 'MP: %s' %self.MP

    def process_terminal_data(self, terminal_data):
        # print 'self.last_refresh_dt %s' % self.last_refresh_dt
        if terminal_data["data_type"] == "local_server_dt":
            self.incoming_dest_dt = convert_product_name_2_dest_dt(self.TradeProduct, self.last_refresh_dt)
            # print 'self.incoming_dest_dt:  %s' % self.incoming_dest_dt
        if self.incoming_dest_dt is None:
            print 'Aware: self.incoming_dest_dt is None'
            return
        if self.current_dest_date is None or self.incoming_dest_dt.date() > self.current_dest_date:
            self.current_dest_date = self.incoming_dest_dt.date()
            self.current_active_futures_expiry_day = get_next_expiry_day(self.TradeProduct, self.current_dest_date, 0, True)
            #print 'self.current_active_futures_expiry_day  %s' % self.current_active_futures_expiry_day
            # self.current_active_hsi_futures = get_next_symbol(self.TradeProduct, self.current_active_futures_expiry_day, 0, False)
            self.current_active_hsi_futures = get_futures_active_contract(self.TradeProduct, self.incoming_dest_dt)
            self.subscribe_symbol(self.current_active_hsi_futures)



        #print 'terminal_data[data_type]:%s' % terminal_data['data_type']
        if terminal_data['data_type'] == 'tick':
            #print 'tick came: %s' % terminal_data['data_content']
            #print 'last_price of %s: %s' % (
            #    self.current_active_hsi_futures, self.last_price(self.current_active_hsi_futures))
            data_content = terminal_data['data_content']
            #print data_content
            self.lasts.append(data_content[ 'last_traded_price'])
            self.vols.append(data_content['last_traded_volume'])
            #print data_content['dt']
            self.dt = datetime.datetime.strptime(data_content['dt'],'%Y-%m-%d %H:%M:%S')
            #print self.dt


            if self.pre_dt is not None and self.dt is not None and self.dt.minute != self.pre_dt.minute:# new minute
                # calculate high ,low, last on minute level
                high = max(self.lasts)
                low = min(self.lasts)
                last = self.lasts[-1]
                vol = sum(self.vols)
                self.lasts = []
                self.vols = []
                self.last = last
                self.vol = vol
                self.IndicatorsRecord["NC"].append(self.narChaBreak.nc_indicator)
                self.IndicatorsRecord["ShpV"].append(self.shapeV.shp_indicator)
                self.IndicatorsRecord["NCVol"].append(self.narChaBreak.vol_indicator)
                self.IndicatorsRecord["ShpVVol"].append(self.shapeV.vol_indicator)
                
            

                self.current_trade_session = get_next_trade_session(self.TradeProduct, self.incoming_dest_dt, False, 0)
                self.next_trade_session = get_next_trade_session(self.TradeProduct, self.incoming_dest_dt, False, 2) # to skip night trade
                self.MP = self._theoretical_position(self.current_active_hsi_futures)
                

                LenForEsti = 300

                if len(self.IndicatorsRecord["NC"]) > LenForEsti:
                    # del(self.IndicatorsRecord["NC"][:-LenForEsti])
                    # del(self.IndicatorsRecord["ShpV"][:-LenForEsti])
                    # del(self.IndicatorsRecord["NCVol"][:-LenForEsti])
                    # del(self.IndicatorsRecord["ShpVVol"][:-LenForEsti])
                    self.IndicatorsRecord["NC"].sort()
                    self.narChaBreak.NCThr = self.IndicatorsRecord["NC"][int(LenForEsti*0.10)]
                    self.IndicatorsRecord["ShpV"].sort()
                    self.shapeV.ShpVThr = self.IndicatorsRecord["ShpV"][-int(LenForEsti*0.10)]
                    self.IndicatorsRecord["NCVol"].sort()
                    self.narChaBreak.VolThr = self.IndicatorsRecord["NCVol"][int(0.75*LenForEsti)]
                    self.IndicatorsRecord["ShpVVol"].sort()
                    self.shapeV.VolThr = self.IndicatorsRecord["ShpVVol"][int(0.75*LenForEsti)]
                    print [self.narChaBreak.NCThr,self.shapeV.ShpVThr, self.narChaBreak.VolThr,len(self.IndicatorsRecord["NC"])]
                    self.IndicatorsRecord = {"NC":[],"ShpV":[],"NCVol":[],"ShpVVol":[]}
                    

                    
                    ThrReady = True
                else:
                    ThrReady = False

                # active contract symbol
                self.minuteLevel(self.dt, high, low, last, vol,self.current_active_hsi_futures,ThrReady)
                # execute minute level strategy

                
        
            if self.pre_dt is not None and self.dt is not None and self.dt.hour != self.pre_dt.hour: # new hour
                print 'days_until_expire: %s' % (self.current_active_futures_expiry_day - self.incoming_dest_dt.date()).days
                print 'new hour comes: %s' % self.dt
                print 'self.must_out: %s'%self.must_out
                print 'self.can_in: %s'%self.can_in
                print 'self.last: %s'% self.last
                print 'self.narChaBreak.min_entry_L: %s'%self.narChaBreak.min_entry_L
                print 'self.narChaBreak.max_entry_S: %s' %self.narChaBreak.max_entry_S
                print 'self.shapeV.entry_L: %s'% self.shapeV.entry_L
                print 'self.shapeV.entry_S: %s'%self.shapeV.entry_S

            self.pre_dt = self.dt
            
            
def main():
    strategy_main(NarC_ShpV)


if __name__ == '__main__':
    main()
