# fx_explore.py
import pandas as pd















# data spec : http://www.histdata.com/f-a-q/data-files-detailed-specification/
names_fx = ['DateTime_Stamp','Open','High','Low','close','volume']
usdcad_m1_2016 = pd.read_csv('D:/Data/forex_currency_2016/DAT_ASCII_USDCAD_M1_2016.csv', sep = ';', header = None, names = names_fx)


print usdcad_m1_2016
