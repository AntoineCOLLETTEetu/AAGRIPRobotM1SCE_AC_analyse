import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import signal
import random
from scipy.stats import linregress
from scipy.stats import ttest_1samp

NAME = ['M1_fall.csv',
        'M1_hold.csv',
        'M1_peaks.csv',
        'M2_fall.csv',
        'M2_hold.csv',
        'M2_peaks.csv',
        'M3.csv','M4_Marker.csv']

COLUMNS = ['Time (s)', 'Fz (N)', 'AI7 (V)', 'AI8 (V)']
DATA = {}
FILTERA, FILTERB = signal.butter(4, 15, fs=200, btype='low', analog=False)

#### Début du code
def start():
    Import_Data()

#### On import les données
def Import_Data():
    for name in NAME:
        DATA[name] = pd.read_csv(name)[COLUMNS].dropna()
        DATA[name][COLUMNS[1]] = - DATA[name][COLUMNS[1]]

#### On montre le signal
def Show_Signal(x, y):
    num_rows = 2
    num_cols = int(len(NAME)/2)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 8))

    axes = axes.flatten()

    # Itérer sur les paires (nom, DataFrame) du dictionnaire
    for i, (name, df) in enumerate(DATA.items()):
        # Sélectionner l'axe correspondant pour le subplot actuel
        ax = axes[i]
        
        # Utiliser Seaborn pour visualiser le DataFrame sur l'axe du subplot
        sns.lineplot(data=df, x=x, y=y, ax=ax)
        
        # Définir le titre du subplot
        ax.set_title(name)
    
    fig.tight_layout()
    plt.show()

### Ces deux fonctions resample le signal
def Resample_All_Signal(target_fr):
    for name in NAME:
        if DATA[name][COLUMNS[0]].iloc[1] != 1/target_fr:
            DATA[name] = Resample_Signal(DATA[name], target_fr)

def Resample_Signal(DF_to_resample, target_fr):
    column = COLUMNS[0]
    # Met le timer sous forme de DateTime
    DF_to_resample[column] = pd.to_datetime(DF_to_resample[column], unit='s')
    # Resample le signal
    DF_Resampled = DF_to_resample.set_index(column).resample(f"{int(1000/target_fr)}L").mean()
    # Reset l'index
    DF_Resampled = DF_Resampled.reset_index()
    # remet le signal en seconde
    DF_Resampled[column] = pd.to_numeric(DF_Resampled[column])/10**9

    return DF_Resampled

### Ces deux fonctions filtre le signal
def Filter_All_Signal():
    for name in NAME:
        DATA[name][COLUMNS[1]] = Filter_Signal(DATA[name][COLUMNS[1]])

def Filter_Signal(sery):
    filtered = signal.filtfilt(FILTERA, FILTERB, sery)
    return filtered


### Ici, on cherche les trigger
def Search_Trigger():
    df_arduino = DATA[NAME[7]].iloc[:,:]
    echantillonnage = 1/200
    df_arduino = arduino_ping_bool(df_arduino, 'AI7 (V)', 'AI7-2')
    df_arduino['A7'] = 0
    df_arduino['A7'] = arduino_ping(df_arduino.loc[df_arduino[df_arduino['AI7-2']].index, 'A7'])
    df_arduino['A7'] = np.where(df_arduino['A7'].isna(), 0, df_arduino['A7'])
    df_arduino['Code-7'] = arduino_ping_translation(df_arduino['A7'][df_arduino['A7'] != 0], echantillonnage)
    df_arduino['Code-7'] = np.where(df_arduino['Code-7'].isna(), '', df_arduino['Code-7'])

    df_arduino['dT'] = df_arduino['Time (s)'][df_arduino['A7'] != 0].diff()
    df_arduino['Code-7'][df_arduino['A7'] != 0]
    tableau = {}
    for index, i in df_arduino[(df_arduino['A7'] != 0) & ((df_arduino['dT']>0.9) | (df_arduino['dT'].isna()))].iterrows():
        tableau[index] = i
    
    df = pd.DataFrame()
    array = np.arange(0,1,echantillonnage)
    df['Time (s)'] = array
    name = 'name'
    i = 0
    for index in tableau.keys():   
        if tableau[index]['Time (s)'] >= 0.100:
            index_start = index - int((200/echantillonnage)/1000)
            index_end = index  + int((800/echantillonnage)/1000)
            df[name + str(i)] = df_arduino['Fz (N)'].iloc[index_start:index_end+1].copy().reset_index(drop = True)
            i += 1

    return df


def arduino_ping(df):

    i = 1
    j = 0
    
    taille = 1
    taille_echantillon = df.tail(1).index[0]
    echantillon = df.index[i]
    debut_pic = df.index[j]
    
    while echantillon < taille_echantillon:
        if echantillon - taille == debut_pic:
            taille = taille + 1
        else:
            df.iloc[j] = taille
            j = i
            debut_pic = df.index[j]
            taille = 1

        i = i+1
        echantillon = df.index[i]
    
    df.iloc[j] = taille
    j = i
    taille = 1 
    
    return df

def arduino_ping_bool(df, name, nc):
    LIMIT_ON = 0.5
    df[nc] = df[name].iloc[:]
    df[nc] = np.where(df[nc] > LIMIT_ON, True, False)
    return df

def arduino_ping_translation(df, fr):

    for i in range(len(df)):

        code = df.iloc[i] * fr
        if code <= 15/1000:
            df.iloc[i] = 'Err'
        elif code <= 25/1000 and code > 15/1000:
            df.iloc[i] = 'F'
        elif code <= 35/1000:
            df.iloc[i] = 'R'
        elif code <= 45/1000:
            df.iloc[i] = 'M'
        elif code <= 55/1000:
            df.iloc[i] = 'T'
        elif code <= 65/1000:
            df.iloc[i] = 'S'
        else:
            df.iloc[i] = 'Err'
        
    return df

def Show_Trigger(a):
    columns = a.columns.array
    columns = columns[range(1,len(columns))].to_numpy()
    temp = pd.DataFrame()
    temp['Time (s)'] = a.loc[:,'Time (s)']
    fig = plt.figure()
    for name in columns:
        if name == 'condition1' or name == 'condition2':
            ax = sns.lineplot(data=a, x='Time (s)', y=name, alpha=0.5, label=name)
        else:
            ax = sns.lineplot(data=a, x='Time (s)', y=name, alpha=0.5)
    temp['mean'] = a[columns].mean(axis=1)
    ax = sns.lineplot(data=temp, x='Time (s)', y='mean', color='black', label='Moyenne')
    ax.set(xlabel='Time (s)', ylabel='Fz (N)')

    plt.vlines(x=[0.2, 0.7], 
           ymin=a[columns].min().min(),
           ymax=a[columns].max().max(),
           color=['red', 'black'], alpha=.5)
    
    
def Create_Trigger(df):
    i = len(df.columns) - 1
    fr = 200
    length = int(fr)
    dictionnary = {}
    for name in NAME[0:len(NAME)-1]:
        dataF = DATA[name].loc[:,['Time (s)','Fz (N)']]
        loop = True
        index = 0
        while loop:
            if (index+length) in dataF.index:
                dictionnary['name'+str(i)] = dataF.loc[range(index, index+length),'Fz (N)'].reset_index(drop=True)
                index += length
                i += 1
            else:
                loop = False
    df = pd.concat([df, pd.DataFrame(dictionnary)], axis=1)
    return df

def Baseline(df):
    fr = 200
    index_start = 0
    index_end = int(200*fr/1000)
    DFBL = df.copy(deep=True)
    for name in DFBL.iloc[:,range(1,len(DFBL.columns))]:
        moyenne = DFBL[name][index_start:index_end].mean()
        DFBL[name] = DFBL[name] - moyenne
    return DFBL

def Drop_Pics_BL(df):
    DFDP = df.copy(deep=True)
    columnsToDrop = []
    for name in DFDP.iloc[:,range(1,len(DFDP.columns))]:
        Conditional_peaks = np.where(abs(DFDP[name]) > 0.2)[0]
        if len(Conditional_peaks) > 0:
            columnsToDrop.append(name)
    # print(columnsToDrop)
    DFDP = DFDP.drop(columnsToDrop, axis=1)

    return DFDP

def Drop_Pics_Min_Max(df):
    DFDP = df.copy(deep=True)
    columnsToDrop = []
    fr = 200
    length = int(100*fr/1000)
    for name in DFDP.iloc[:,range(1,len(DFDP.columns))]:
        for index, Force in enumerate(DFDP[name]):
            # print(name, len(DFBL[name]), index, Force)
            indexMin = index
            indexMax = min(index+length, len(DFDP[name]))
            MaxMinusMin = abs(DFDP[name][indexMin:indexMax].max() - DFDP[name][indexMin:indexMax].min())
            conditionnalValue = MaxMinusMin > 0.1
            if conditionnalValue:
                columnsToDrop.append(name)
                break
    # print(columnsToDrop)
    DFDP = DFDP.drop(columnsToDrop, axis=1)
    return DFDP

def RandomSeparateMeasures(df):
    allMeasures = df.iloc[:,range(1,len(df))]
    columns1 = allMeasures.sample(frac=.5, replace=False, axis=1, random_state=42).columns
    columns2 = allMeasures.columns.difference(columns1)
    condition1 = df.loc[:,'Time (s)']
    condition1 = pd.concat([condition1, allMeasures[columns1]], axis=1)
    condition2 = df.loc[:,'Time (s)']
    condition2 = pd.concat([condition2, allMeasures[columns2]], axis=1)
    condition1 = condition1.reindex(columns=df.columns).dropna(axis=1)
    condition2 = condition2.reindex(columns=df.columns).dropna(axis=1)

    dfoutput = pd.DataFrame()
    dfoutput['Time (s)'] = df.loc[:,'Time (s)']
    dfoutput['condition1'] = condition1.iloc[:,range(1,len(condition1.columns))].mean(axis=1)
    dfoutput['condition2'] = condition2.iloc[:,range(1,len(condition2.columns))].mean(axis=1)

    return dfoutput, columns1, columns2