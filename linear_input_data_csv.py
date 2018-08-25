import numpy
import pandas
import matplotlib.pyplot as plt
from mlxtend.preprocessing import one_hot

def import_data():
    dataframe = pandas.read_csv('/home/deka/Desktop/datasets/example.csv')
    return dataframe

def info_data():
    data = import_data()

    empty_df = pandas.DataFrame(columns=['gas_fuel_flow','hpc_eta','ngp','npt'])

    empty_df.gas_fuel_flow=numpy.array(data.gas_fuel_flow.rolling(window=207).mean())
    plt.plot(empty_df.gas_fuel_flow, 'red');
    plt.title("Gas Flow smoothing trend");
    plt.show()

    empty_df.hpc_eta=numpy.array(data.hpc_eta.rolling(window=125).mean())
    plt.plot(empty_df.hpc_eta, 'blue');
    plt.title("HPC ETA smoothing trend");
    plt.show()

    empty_df.ngp=numpy.array(data.ngp.rolling(window=208).mean())
    plt.plot(empty_df.ngp, 'green');
    plt.title("NGP smoothing trend");
    plt.show()

    empty_df.npt=numpy.array(data.npt.rolling(window=1000).mean())
    plt.plot(empty_df.npt, 'pink');
    plt.title("NPT smoothing trend");
    plt.show()

    return empty_df

def input_data():
    df = info_data() # values
    divide = 14500 # value for divide data on train and test
    df.dropna(inplace=True)
    print(df)
    print(df.head(10),"\n") # print first 10 raws
    print(df.info(),"\n") # print info about dataframe
    print(df.shape,"\n") # print dataframe shape
    print(df.describe(),"\n") # print info about values
    print(df.corr(),"\n") # dataframe correlation

    # Divide data on train and test without shuffle
    train_X = numpy.array(df.values[:divide,0:3])
    train_Y = numpy.array(df.values[:divide,3:])
    test_X = numpy.array(df.values[divide:,0:3])
    test_Y = numpy.array(df.values[divide:,3:])

    return train_X, test_X, train_Y, test_Y

input_data()
