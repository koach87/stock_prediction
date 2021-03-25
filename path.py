
import os
import pandas as pd

def list_data():
    reg = []
    for dirname, _, filenames in os.walk('.\Data'):
        for i, filename in enumerate(filenames):
            reg.append(filename)
            os.path.join(dirname,filename)
    print(reg)

def find_path(symbol, date):
    for dirname, _, filenames in os.walk('.\Data'):
        for i, filename in enumerate(filenames):
            if(symbol in filename):
                return os.path.join(dirname,filename)
    return 'Not found : ' + symbol

if __name__ == '__main__':
    dates = pd.date_range('2010-01-02','2017-10-11',freq='B')
    print(find_path('aaaa',dates))
