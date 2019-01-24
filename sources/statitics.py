import pandas as pd
import numpy as np
from sklearn import linear_model
import sklearn
import matplotlib.pyplot as plt

# test_feature_df = pd.read_csv(r'\\homedrive.login.htw-berlin.de\s0554617\WI-Profile\Desktop\Downloads\CSVFiles\test_feature_df.csv')
# test_label_df =pd.read_csv(r'\\homedrive.login.htw-berlin.de\s0554617\WI-Profile\Desktop\Downloads\CSVFiles\test_label_df.csv')
# test_df =pd.read_csv(r'\\homedrive.login.htw-berlin.de\s0554617\WI-Profile\Desktop\Downloads\CSVFiles\test_df.csv')
# train_feature_df =pd.read_csv(r'\\homedrive.login.htw-berlin.de\s0554617\WI-Profile\Desktop\Downloads\CSVFiles\train_feature_df.csv')
# train_label_df =pd.read_csv(r'\\homedrive.login.htw-berlin.de\s0554617\WI-Profile\Desktop\Downloads\CSVFiles\train_label_df.csv')
#
##Unwichtige Spalte wird gelöscht
# test_feature_df= test_feature_df.drop(['Unnamed: 0'],axis=1)
# test_label_df= test_label_df.drop(['Unnamed: 0'],axis=1)
# test_df= test_df.drop(['Unnamed: 0'],axis=1)
# train_feature_df= train_feature_df.drop(['Unnamed: 0'],axis=1)
# train_label_df = train_label_df.drop(['Unnamed: 0'],axis=1)

ergebnis_df = pd.read_csv(r'/home/tahir/Documents/code/tahir/SALT_USWS/sources/ergebnis_df_onlyDistanceToMitte')
ergebnis_df = ergebnis_df.drop(['Unnamed: 0'], axis=1)

print(np.var(ergebnis_df["score"]))
print(np.var(ergebnis_df["predictedscore"]))
print(np.mean(ergebnis_df["score"]))

plt.hist(ergebnis_df["score"])
plt.hist(ergebnis_df["predictedscore"])
plt.show

count = 0
count1 = 0

for x in range(len(ergebnis_df)):
    dif = 0
    dif = abs((ergebnis_df.at[x, "score"]) - (ergebnis_df.at[x, "predictedscore"]))
    if (dif < 1):
        count = count + 1



print(count)
