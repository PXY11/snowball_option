# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 14:21:29 2021

@author: Mr.P
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import *
from joblib import Parallel, delayed

mc_para = {}
mc_para['r'] = 0.05
mc_para['vol'] = 0.23
mc_para['T'] = 1
mc_para['N'] = 250
mc_para['simulations'] = 1000
mc_para['dt'] = mc_para['T']/mc_para['N']
mc_para['mu'] = 0

basic_para = {}
basic_para['s0'] = 10000
basic_para['ki'] = basic_para['s0']*0.8
basic_para['ko'] = basic_para['s0']*1
basic_para['coupon_rate'] = 0.2 #敲出票息（年化）
basic_para['dividend'] = 0.2 #红利票息（年化）
basic_para['period'] = 40 #两个月观察一次，一个月20个交易日
basic_para['observe_point'] = [k for k in range(40,mc_para['N'],basic_para['period'])]
basic_para['observe_point_copy'] = [k for k in range(40,mc_para['N'],basic_para['period'])]
basic_para['calender'] = [k for k in range(mc_para['N'])]
basic_para['buy_cost'] = 0.0005  #delta对冲买入现货手续费 
basic_para['sell_cost'] = 0.0009 #delta对冲卖出现货手续费
basic_para['F'] = 10000 #面值
basic_para['init_net_value'] = 100000000 #计算对冲收益时初始账户净值
basic_para['capital_cost_rate'] = 0.05 #资金使用成本--百分比
basic_para['trade_cost_rate'] = 0.00015 #交易成本--绝对值
basic_para['path_count'] = 0
np.random.seed(10)

class AutoCall():
    def __init__(self,basic_para,mc_para):
        self.mc_parameter = mc_para   #蒙特卡洛参数
        self.basic_parameter = basic_para  #基础参数
        self.stock_path_1 = pd.DataFrame()   #用于保存股票价格路径的dataframe
        self.stock_path_2 = pd.DataFrame()   #用于保存股票价格路径的dataframe
        self.cum_rtn_1 = pd.DataFrame()    #用于保存累计收益率路径的dataframe
        self.cum_rtn_2 = pd.DataFrame()    #用于保存累计收益率路径的dataframe
        self.cum_rtn_1_copy = pd.DataFrame()    #用于保存累计收益率路径的dataframe
        self.cum_rtn_2_copy = pd.DataFrame()    #用于保存累计收益率路径的dataframe
        self.random_r_cum_1 = np.zeros([self.mc_parameter['N'],self.mc_parameter['simulations']])
        self.random_r_cum_2 = np.zeros([self.mc_parameter['N'],self.mc_parameter['simulations']])
        self.return_lst = []
        self.ob_copy = []
        self.ob_copy.extend(self.basic_parameter['observe_point'])
    def monte_carlo(self):
        N = self.mc_parameter['N']
        simulations = self.mc_parameter['simulations']
        vol = self.mc_parameter['vol']
        dt = self.mc_parameter['dt']
        mu = self.mc_parameter['mu']
        random_r = np.random.normal(0, 1, (N, simulations))
        panduan_up = (0.1 - mu * dt) / (vol * np.sqrt(dt))  # 将收益率序列用涨跌幅规范
        panduan_low = (-0.1 - mu * dt) / (vol * np.sqrt(dt))
        random_r[np.where(random_r > panduan_up)] = panduan_up
        random_r[np.where(random_r < panduan_low)] = panduan_low
        random_r_cum_1 = np.cumprod(1 + mu * dt + vol * np.sqrt(dt) * random_r, axis=0)
        random_r_cum_2 = np.cumprod(1 + mu * dt + vol * np.sqrt(dt) * -random_r, axis=0)
        #加上第一日收益率
        head = np.array([[1]*simulations])
        random_r_cum_1 = np.vstack((head,random_r_cum_1))
        random_r_cum_2 = np.vstack((head,random_r_cum_2))
        self.random_r_cum_1 = random_r_cum_1
        self.random_r_cum_2 = random_r_cum_2
        df1 = pd.DataFrame(random_r_cum_1)
        df2 = pd.DataFrame(random_r_cum_2)
        self.cum_rtn_1 = df1.copy(deep=True)
        self.cum_rtn_2 = df2.copy(deep=True)
        self.cum_rtn_1_copy = df1.copy(deep=True)
        self.cum_rtn_2_copy = df2.copy(deep=True)
    def get_stock_path(self,s0):
        self.stock_path_1 = s0*self.cum_rtn_1
        self.stock_path_2 = s0*self.cum_rtn_2
    
    def spath_expect_payoff(self,spath):
        '''
        参数准备
        '''
        s0 = basic_para['s0']
        r = self.mc_parameter['r']
        N = self.mc_parameter['N']
        dt = self.mc_parameter['dt']
        dividend = self.basic_parameter['dividend']
        coupon_rate = self.basic_parameter['coupon_rate']
        T = self.mc_parameter['T']
        F = self.basic_parameter['F']
        discount_factor = np.exp(-r*N*dt)
        simulations = self.mc_parameter['simulations']
        ki = self.basic_parameter['ki']
        ko = self.basic_parameter['ko']
        #######################################################################
        ob = self.basic_parameter['observe_point']
        spath_value = spath.values
        path_ind = spath.index.tolist()
#        print(type(path_value))
        path_col = [i for i in range(simulations)]
#        ob_point_price = spath_value[ob]
        ob_point_price = spath.loc[ob].values
#        print(ob)
        if len(ob) != 0:
        #######################################################################
        #spath_expect_payoff要加一个判断ob_point是不是空的功能，为了配合get_se_deltas()
        #计算最后一个观察日至到期日之间的收益
        #最后一个观察日至到期日之间，肯定不会有敲出了，故在else情况中ever_out_payoff设置为0
        #######################################################################
            ever_out = list(np.where(np.max(ob_point_price,axis=0)>=ko)[0])
#            path_ob_point = pd.DataFrame(spath_value).iloc[ob,ever_out]
#            path_ob_point = pd.DataFrame(spath_value).loc[ob,ever_out]
            path_ob_point = spath.loc[ob,ever_out]
            path_ob_point = path_ob_point > ko
            first_out_date = path_ob_point.idxmax()
            out_discount_factor = np.exp(-r*first_out_date*dt)
            payoff_ever_out = (1+first_out_date/N*coupon_rate)*F*out_discount_factor
            payoff_ever_out = payoff_ever_out.sum()
            
            ever_in = np.where(np.min(spath_value,axis=0)<=ki)[0]
            ever_io = list(set(ever_in)&set(ever_out))
            only_in = list(set(ever_in)^set(ever_io))
            payoff_only_in = np.exp(-r*T)*spath_value[-1,only_in].sum()
            noi = list(set(path_col)^set(ever_out)^set(only_in))
            payoff_noi = len(noi)*(1+dividend)*F*np.exp(-r*T)
            payoff_expect = (payoff_ever_out + payoff_noi + payoff_only_in)/simulations
        else:
            payoff_ever_out = 0
            ever_out = []
            ever_in = np.where(np.min(spath_value,axis=0)<=ki)[0]
            ever_io = list(set(ever_in)&set(ever_out))
            only_in = list(set(ever_in)^set(ever_io))
            payoff_only_in = np.exp(-r*T)*spath_value[-1,only_in].sum()
            noi = list(set(path_col)^set(ever_out)^set(only_in))
            payoff_noi = len(noi)*(1+dividend)*F*np.exp(-r*T)
            payoff_expect = (payoff_ever_out + payoff_noi + payoff_only_in)/simulations
        return {'expect_payoff':payoff_expect}
        
    def customer_expect_payoff(self):
        sp_1 = self.stock_path_1
        ep_1 = self.spath_expect_payoff(sp_1)['expect_payoff']
        sp_2 = self.stock_path_2
        ep_2 = self.spath_expect_payoff(sp_2)['expect_payoff']
        #######################################################################
        expect_payoff = (ep_1 + ep_2)/2
#        print('ep_1:',ep_1,'ep_2:',ep_2,'expect_payoff',expect_payoff)
        return {'expect_payoff':expect_payoff}
    
    
    def get_delta(self,s,ds):
        su = s+ds
        self.get_stock_path(su)
        pup = self.customer_expect_payoff()['expect_payoff']
        sd = s-ds
        self.get_stock_path(sd)
        pdown = self.customer_expect_payoff()['expect_payoff']
        delta = (pup-pdown)/(2*ds)
        return {'delta':delta,'su':su,'sd':sd,'pup':pup,'pdown':pdown}

    def get_gamma(self,s,ds):
        self.get_stock_path(s)
        p = self.customer_expect_payoff()['expect_payoff']
        su = s+ds
        self.get_stock_path(su)
        pup = self.customer_expect_payoff()['expect_payoff']
        sd = s-ds
        self.get_stock_path(sd)
        pdown = self.customer_expect_payoff()['expect_payoff']
        gamma = (pup+pdown-2*p)/(ds**2)
        return {'gamma':gamma,'s':s,'su':su,'sd':sd,'p':p,'pup':pup,'pdown':pdown}
    def get_se_deltas(self,se,ds):
        deltas_lst = []
        ob_point = self.ob_copy
        #由于每次调用，会删改self.basic_parameter['observe_point']，所以每次调用前要重置观察点列表 
        if len(self.basic_parameter['observe_point'])!=len(self.ob_copy):
            self.basic_parameter['observe_point'].clear()
            self.basic_parameter['observe_point'].extend(self.ob_copy)
        for i in range(len(se)):
            sprice = se[i]
            np.random.seed(10)
#            self.monte_carlo()
            self.cum_rtn_1 = self.cum_rtn_1_copy.copy(deep=True)
            self.cum_rtn_2 = self.cum_rtn_2_copy.copy(deep=True)
            #最后一个观察日后，到期日之前，此时self.basic_parameter['observe_point']已经空了，但还要继续计算delta
            if len(self.basic_parameter['observe_point']) !=0:
                if i == self.basic_parameter['observe_point'][0] and sprice < self.basic_parameter['ko']:
                    self.basic_parameter['observe_point'].pop(0)
#            print(self.basic_parameter['observe_point'])
            #第五天则从第五天开始截取收益率矩阵，股价矩阵也对等的只有245天的数据
            self.cum_rtn_1 = self.cum_rtn_1.iloc[i:,:]
            self.cum_rtn_2 = self.cum_rtn_2.iloc[i:,:]
            tic1 = time()
            delta = self.get_delta(sprice,ds)['delta']
            tic2 = time()
#            print('计算单日delta耗时：',tic2-tic1)
#            print('delta%d=%.4f计算完毕'%(i,delta))
            deltas_lst.append(delta)
            #i%40==0 和 i == self.basic_parameter['observe_point'][0]都能用来判断是否是敲出观察日
            #如果观察日敲出了，后续delta全部赋值为0不用再计算了,第一日不是观察日所以要加上i!=0
            #最后一个观察日后，到期日之前，此时self.basic_parameter['observe_point']已经空了，但还要继续计算delta
            if len(self.basic_parameter['observe_point']) !=0:
                #如果i是观察日并且i这天敲出了后面就不用再计算了
                if i in self.basic_parameter['observe_point'] and sprice > self.basic_parameter['ko'] and i!=0:
                    remain_deltas = [0 for i in range(len(se)-len(deltas_lst))]
                    deltas_lst += remain_deltas
#                    ob_ind = ob_point.index(i)
#                    ob_day = int(ob_ind)
#                    ob_day += 1
#                    print('第%d个观察日敲出了，后续deltas为0'%ob_day)
                    break
        print('单路径delta计算完毕')
        return deltas_lst
    
    def get_df_deltas(self,df,ds):
        deltas_dic = {}
        for i in df.columns.tolist():
            tick1 = time()
            se = df[i]
            temp = self.get_se_deltas(se,ds)
            deltas_dic[i]=temp
            ith = int(i)
            ith += 1
            tick2 = time()
            time_cost = tick2 - tick1
            print('第%d条路径delta计算完毕，耗时：%.6f'%(ith,time_cost))
        deltas_df = pd.DataFrame(deltas_dic)
        return deltas_df

    def cal_se_hedge(self,stock_se,delta_se):
        '''
        存数据所需list初始化
        '''
        stock_lst = stock_se.values.tolist()
        delta_lst = delta_se.values.tolist()
        net_value_lst = [] #净值
        delta_worth_lst = [] #delta市值
        stock_num_lst = [] # 对冲所需股票数
        trade_cost_lst = [] #交易费用
        trade_chg_lst = [] #交易净流出
        stock_value_lst = [] #股票价值
        cash_lst = [] #现金
        present_worth_lst = []
        capital_use = []
        '''
        第一日参数计算
        '''
        init_net_value = self.basic_parameter['init_net_value']
        trade_cost_rate = self.basic_parameter['trade_cost_rate']
        init_present_worth = init_net_value
        init_delta_worth = delta_lst[0]*init_net_value
        init_stock_num = init_delta_worth/stock_lst[0]
        init_trade_cost = np.abs(init_stock_num*stock_lst[0]*trade_cost_rate)
        init_trade_chg = init_stock_num*stock_lst[0] + init_trade_cost
        init_stock_value =  init_stock_num*stock_lst[0]
        init_cash = init_net_value - init_trade_chg
        init_net_value = init_cash + init_stock_value
        init_capital_use = init_stock_num*stock_lst[0] + init_trade_cost
        '''
        第一日参数加入列表
        '''
        net_value_lst.append(init_net_value)
        present_worth_lst.append(init_present_worth)
        delta_worth_lst.append(init_delta_worth)
        stock_num_lst.append(init_stock_num)
        trade_cost_lst.append(init_trade_cost)
        trade_chg_lst.append(init_trade_chg)
        stock_value_lst.append(init_stock_value)
        cash_lst.append(init_cash)
        capital_use.append(init_capital_use)
        '''
        第i日参数计算
        '''
        for i in range(1,251):
            present_worth_temp = net_value_lst[i-1] + stock_num_lst[i-1]*(stock_lst[i]-stock_lst[i-1])
            delta_worth_temp = delta_lst[i]*present_worth_temp
            stock_num_temp = delta_worth_temp/stock_lst[i]
            trade_cost_temp =np.abs( (stock_num_temp - stock_num_lst[i-1])*stock_lst[i]*trade_cost_rate)
            trade_chg_temp = (stock_num_temp - stock_num_lst[i-1])*stock_lst[i] + trade_cost_temp
            stock_value_temp = stock_num_temp*stock_lst[i]
            cash_temp = cash_lst[i-1] - trade_chg_temp
            net_value_temp = cash_temp + stock_value_temp
            capital_use_temp = capital_use[i-1] + trade_cost_temp + (stock_num_temp-stock_num_lst[i-1])*stock_lst[i] 
            
            net_value_lst.append(net_value_temp)
            present_worth_lst.append(present_worth_temp)
            delta_worth_lst.append(delta_worth_temp)
            stock_num_lst.append(stock_num_temp)
            trade_cost_lst.append(trade_cost_temp)
            trade_chg_lst.append(trade_chg_temp)
            stock_value_lst.append(stock_value_temp)
            cash_lst.append(cash_temp)
            capital_use.append(capital_use_temp)
            
        res_dic = {
                    'stock_lst':stock_lst,
                    'delta_lst':delta_lst,
                    'net_value_lst':net_value_lst,
                    'present_worth_lst':present_worth_lst,
                    'delta_worth_lst':delta_worth_lst,
                    'stock_num_lst':stock_num_lst,
                    'trade_cost_lst':trade_cost_lst,
                    'trade_chg_lst':trade_chg_lst,
                    'cash_lst':cash_lst,
                    'stock_value_lst':stock_value_lst,
                    'capital_use':capital_use
                }
        res_df = pd.DataFrame(res_dic)
        capital_cost_rate = self.basic_parameter['capital_cost_rate']
        res_df['capital_use_cost'] = res_df['capital_use']*capital_cost_rate/365
        payoff = (res_df['net_value_lst'].iloc[-1] - res_df['capital_use_cost'].iloc[-1])
        return {'payoff':payoff,'res_df':res_df}

    def cal_df_hedge(self,stock_df,deltas_df):
        payoffs = []
        for i in stock_df.columns.tolist():
            se_payoff = self.cal_se_hedge(stock_df.loc[:,i],deltas_df.loc[:,i])['payoff']
            payoffs.append(se_payoff)
        payoffs = pd.Series(payoffs)
        return payoffs

t1 = time()
atc = AutoCall(basic_para,mc_para)
t2 = time()
atc.monte_carlo()
t3 = time()
s0 = atc.basic_parameter['s0']
t4 = time()
atc.get_stock_path(s0)
t5 = time()
cum_rtn_1 = atc.cum_rtn_1
cum_rtn_2 = atc.cum_rtn_2
t6 = time()
spath_1 = atc.stock_path_1
spath_2 = atc.stock_path_2
t7 = time()
customer_expect_payoff = atc.customer_expect_payoff()
t8 = time()
delta0 = atc.get_delta(6520,100)
t9 = time()
se = spath_1.iloc[:,2]
deltas_lst = atc.get_se_deltas(se,100)
#final_observe_point = atc.basic_parameter['observe_point'] 
#final_observe_point = atc.ob_copy
#ob_copy = atc.ob_copy
t10 = time()
#deltas_df0 = atc.get_df_deltas(spath_1,100)
t11 = time()
#deltas_df1 = atc.get_df_deltas(spath_1,100)
t12 = time()
print('蒙特卡洛模拟次数:',mc_para['simulations'])
print('构建对象耗时：',t2-t1)
print('蒙特卡洛耗时：',t3-t2)
print('取单个参数耗时：',t4-t3)
print('获取股票路径耗时：',t5-t4)
print('获取收益率矩阵耗时：',t6-t5)
print('获取股票矩阵耗时：',t7-t6)
print('获取客户收益耗时：',t8-t7)
print('计算delta耗时：',t9-t8)
print('计算单路径deltas耗时：',t10-t9)
print('第一次计算所有路径deltas耗时',t11-t10)
print('第二次计算所有路径deltas耗时',t12-t11)
print('总耗时：',t12-t1)

#
#t13 = time()
#deltas_lst_2 = Parallel(n_jobs=4)(delayed(atc.get_se_deltas)(spath_1[i],100) for i in spath_1.columns.tolist())
#deltas_df2 = pd.DataFrame(deltas_lst_2)
#deltas_df2 = deltas_df2.T
#t14 = time()
#print('第一次并行计算所有列deltas耗时：',t14-t13)
#deltas_lst_3 = Parallel(n_jobs=4)(delayed(atc.get_se_deltas)(spath_1[i],100) for i in spath_1.columns.tolist())
#deltas_df3 = pd.DataFrame(deltas_lst_3)
#deltas_df3 = deltas_df3.T
#t15 = time()
#print('第二次并行计算所有列deltas耗时：',t15-t14)
#
##deltas_df = pd.read_csv('deltas_cal_2.csv')
##Astock_path = pd.read_csv('spath_1_cal_2.csv')
#deltas_df = deltas_df2
#stock_path = spath_1
#t16 = time()
#hedge_res_0 = atc.cal_df_hedge(stock_path,deltas_df)
#t17 = time()
#print('第一次无并行计算所有列hedge_payoff耗时：',t17-t16)
#hedge_res_1 = Parallel(n_jobs=9)(delayed(atc.cal_se_hedge)(stock_path[i],deltas_df[i]) for i in stock_path.columns.tolist())
#t18 = time()
#print('第二次并行计算所有列hedge_payoff耗时：',t18-t17)

