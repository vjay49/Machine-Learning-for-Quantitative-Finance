# Import libraries

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
import datetime
from numpy.random import seed
from tensorflow.random import set_seed
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers

# Change the number of rows and columns to display
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.precision', 5)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# Define a path for import and export

path = 'E:\\UM\Courses\\FIN427\\FIN427 Winter 2024\\'

# Import data
returns1 = pd.read_excel(path + 'Aggregate PE Ranks 20240408_2308.xlsx', sheet_name='pe07')
print(returns1.head())
print(returns1.columns)

# Analysis that uses training and validation datasets.
# The dataset has 324 months of returns from April 1995 to March 2022.
datecut = datetime.datetime(2016, 12, 31)
print(datecut)
returns_train = returns1[returns1['retmonth'] <= datecut]
returns_test  = returns1[returns1['retmonth'] >  datecut]
print(returns_train.head(5))
print(returns_train.tail(5))
print(returns_test.head(5))
print(returns_test.tail(5))

y = returns1['ey1tr']
x = returns1[['lnmktcapadj', 'bidaskadj', 'turnadj', 'putcalladj', 'bmaadj', 'int_ebitadj',
'empgadj', 'revgradj', 'roeadj', 'cradj', 'capex_salesadj', 'lnpatentsadj', 'cfi_salesadj', 'insideradj',
'payoutadj', 'sentimentadj', 'deadj',
'bidaskmiss', 'putcallmiss', 'bmamiss', 'int_ebitmiss', 'empgmiss', 'revgrmiss', 'roemiss', 'crmiss',
'capex_salesmiss', 'lnpatentsmiss', 'cfi_salesmiss', 'insidermiss', 'payoutmiss', 'sentimentmiss', 'demiss']]


y_train = returns_train['ey1tr']
x_train = returns_train[['lnmktcapadj', 'bidaskadj', 'turnadj', 'putcalladj', 'bmaadj', 'int_ebitadj',
'empgadj', 'revgradj', 'roeadj', 'cradj', 'capex_salesadj', 'lnpatentsadj', 'cfi_salesadj', 'insideradj',
'payoutadj', 'sentimentadj', 'deadj',
'bidaskmiss', 'putcallmiss', 'bmamiss', 'int_ebitmiss', 'empgmiss', 'revgrmiss', 'roemiss', 'crmiss',
'capex_salesmiss', 'lnpatentsmiss', 'cfi_salesmiss', 'insidermiss', 'payoutmiss', 'sentimentmiss', 'demiss']]

y_test = returns_test['ey1tr']
x_test = returns_test[['lnmktcapadj', 'bidaskadj', 'turnadj', 'putcalladj', 'bmaadj', 'int_ebitadj',
'empgadj', 'revgradj', 'roeadj', 'cradj', 'capex_salesadj', 'lnpatentsadj', 'cfi_salesadj', 'insideradj',
'payoutadj', 'sentimentadj', 'deadj',
'bidaskmiss', 'putcallmiss', 'bmamiss', 'int_ebitmiss', 'empgmiss', 'revgrmiss', 'roemiss', 'crmiss',
'capex_salesmiss', 'lnpatentsmiss', 'cfi_salesmiss', 'insidermiss', 'payoutmiss', 'sentimentmiss', 'demiss']]

# Initialize the random seed for the neural network. Nets select random weights to start the process.
# Need to set the numpy seed, which initializes the tensorflow seed.
seed(24754)
set_seed(11610)

# Neural network code for the full dataset appears, followed by analysis on training and test sets
nnet0 = Sequential()

# Add a layer of 32 neurons with ReLU activation.
# It is possible to add a penalty function.
# In the absence of a penalty factor the neural network can overfit to the training data and therefore perform
# poorly in predicting returns in the test sample.
nnet0.add(Dense(32, input_dim=x_train.shape[1], activation='relu', kernel_regularizer=regularizers.l1(0.0001)))
# nnet0.add(Dense(32, input_dim=x_train.shape[1], activation='relu'))
# You can add additional hidden layers by switching on the lines below.
# nnet0.add(Dense(3, input_dim=x.shape[1], activation='relu'))
# nnet0.add(Dense(3, input_dim=x.shape[1], activation='relu'))
# nnet0.add(Dense(3, input_dim=x.shape[1], activation='relu'))

# Add an output layer with one output and linear activation
nnet0.add(Dense(1, activation='linear'))

# Compile the net using the adam ("adaptive moment estimation") optimizer and mean squared error as the loss function.
nnet0.compile(optimizer='adam', loss='mse')

# The default learning rate is 0.001 and the learning rate ranges from 0 to 1.
# Switch on the two lines below to specify a different learning rate.
# A fast learning rate might converge to a local optimum. A slow learning rate might not converge.
# opt = keras.optimizers.Adam(learning_rate=0.001)
# nnet0.compile(optimizer=opt, loss='mse')

# Fit the data
history0 = nnet0.fit(x, y, epochs=3)

# Each epoch can be split into batches of different sizes. Use the line below instead of the line above
# history0 = nnet0.fit(x, y, epochs=50, batch_size=10000)

# Calculate R-squared score
preds = nnet0.predict(x)
print('R-squared full:', r2_score(y, preds))

# Plot the loss function history
plt.plot(history0.history['loss'])
plt.show()

# Put data together for review
dfx = x.rename_axis('oldindex').reset_index()
dfy = pd.DataFrame(y).rename_axis('oldindex').reset_index()
predictions = pd.DataFrame(preds, columns=['pred_ey1tr'])
agg1 = pd.merge(dfx, dfy)
print(agg1)
agg2 = agg1.merge(predictions, how='inner', left_index=True, right_index=True)
print(agg2)

# Export results to Excel (use sparingly as the file is large)
with pd.ExcelWriter(path + 'Neural networks output full 20240408_2318.xlsx') as writer:
    agg2.to_excel(writer, sheet_name='Agg2')

# OLS regression for a quick and dirty view on what characteristics are being used in the neural network under the
# assumption of a linear relationship between features and predicted abnormal returns.
deletexlist = ['oldindex']
print(dfx)
x0 = dfx.drop(deletexlist, axis=1)
x0 = sm.add_constant(x0)
y0 = predictions
model = sm.OLS(y0, x0).fit()
print_model = model.summary()
ols_coef = model.params
ols_rsq = model.rsquared
print(print_model)
print(f'R-squared: {model.rsquared:.5f}')
print(ols_coef)

# We can also use a decision tree to demonstrate what characteristics are most important in the network

dt = DecisionTreeRegressor(max_depth=10, min_weight_fraction_leaf=0.01)
x0tree = x0.drop('const', axis=1)
dtmodel = dt.fit(x0tree, y0)
dtresult = dt.score(x0tree, y0)
dfscore = pd.DataFrame([dtresult])
print(dfscore)

fn = ['const', 'lnmktcapadj', 'bidaskadj', 'turnadj', 'pegadj', 'putcalladj', 'bmaadj', 'int_ebitadj', 'empgadj',
'revgradj', 'roeadj', 'cradj', 'capex_salesadj', 'lnpatentsadj', 'cfi_salesadj', 'insideradj', 'payoutadj',
'sentimentadj', 'deadj',
'bidaskmiss', 'pegmiss', 'putcallmiss', 'bmamiss', 'int_ebitmiss', 'empgmiss', 'revgrmiss', 'roemiss', 'crmiss',
'capex_salesmiss', 'lnpatentsmiss', 'cfi_salesmiss', 'insidermiss', 'payoutmiss', 'sentimentmiss', 'demiss']

plot_tree(dtmodel, feature_names=fn)
plt.savefig(path + 'Net tree explanation full 20240408_2324.pdf')
plt.show()

# Analysis with training and test datasets

# Initialize the neural net
nnet1 = Sequential()

# Add a layer of 32 neurons with ReLU activation.
# It is possible to add a penalty function.
# In the absence of a penalty factor the neural network can overfit to the training data and therefore perform
# poorly in predicting returns in the test sample.
nnet1.add(Dense(32, input_dim=x_train.shape[1], activation='relu', kernel_regularizer=regularizers.l1(0.0001)))

# You can add additional hidden layers by switching on the lines below.
# nnet1.add(Dense(3, input_dim=x_train.shape[1], activation='relu'))
# nnet1.add(Dense(3, input_dim=x_train.shape[1], activation='relu'))
# nnet1.add(Dense(3, input_dim=x_train.shape[1], activation='relu'))

# Add an output layer with one output and linear activation
nnet1.add(Dense(1, activation='linear'))

# Compile the net using the adam ("adaptive moment estimation") optimizer and mean squared error as the loss function.
nnet1.compile(optimizer='adam', loss='mse')

# The default learning rate is 0.001 and the learning rate ranges from 0 to 1.
# Switch on the two lines below to specify a different learning rate.
# A fast learning rate might converge to a local optimum. A slow learning rate might not converge.
# opt = keras.optimizers.Adam(learning_rate=0.001)
# nnet1.compile(optimizer=opt, loss='mse')

# Fit the data to the full sample and the training set.
history1 = nnet1.fit(x_train, y_train, epochs=10)

# Each epoch can be split into batches of different sizes. Use the line below instead of the line above
# history1 = nnet1.fit(x_train, y_train, epochs=50, batch_size=10000)

# Calculate R-squared score for training and test sets
train_preds = nnet1.predict(x_train)
test_preds = nnet1.predict(x_test)
print('R-squared train:', r2_score(y_train, train_preds))
print('R-squared test:', r2_score(y_test, test_preds))

# Plot the loss function history
plt.plot(history1.history['loss'])
plt.show()

# Put data together for review
dfx_train = x_train.rename_axis('oldindex').reset_index()
dfy_train = pd.DataFrame(y_train).rename_axis('oldindex').reset_index()
predictions_train = pd.DataFrame(train_preds, columns=['pred_ey1tr'])
aggtrain1 = pd.merge(dfx_train, dfy_train)
print(aggtrain1)
aggtrain2 = aggtrain1.merge(predictions_train, how='inner', left_index=True, right_index=True)
print(aggtrain2)

dfx_test = x_test.rename_axis('oldindex').reset_index()
dfy_test = pd.DataFrame(y_test).rename_axis('oldindex').reset_index()
predictions_test = pd.DataFrame(test_preds, columns=['pred_ey1tr'])
aggtest1 = pd.merge(dfx_test, dfy_test)
print(aggtest1)
aggtest2 = aggtest1.merge(predictions_test, how='inner', left_index=True, right_index=True)
print(aggtest2)

# Export results to Excel (use sparingly as the file is large)
with pd.ExcelWriter(path + 'Neural networks output train 20240408_2332.xlsx') as writer:
    aggtrain2.to_excel(writer, sheet_name='Aggtrain2')

# OLS regression for a quick and dirty view on what characteristics are being used in the neural network under the
# assumption of a linear relationship between features and predicted abnormal returns.
deletexlist = ['oldindex']
print(dfx_train)
x1 = dfx_train.drop(deletexlist, axis=1)
x1 = sm.add_constant(x1)
y1 = predictions_train
model = sm.OLS(y1, x1).fit()
print_model = model.summary()
ols_coef = model.params
ols_rsq = model.rsquared
print(print_model)
print(f'R-squared: {model.rsquared:.5f}')
print(ols_coef)

# We can also use a decision tree to demonstrate what characteristics are most important in the network

dt = DecisionTreeRegressor(max_depth=10, min_weight_fraction_leaf=0.01)
xtree = x1.drop('const', axis=1)
dtmodel = dt.fit(xtree, y1)
dtresult_train = dt.score(xtree, y1)
dfscore_train = pd.DataFrame([dtresult_train])
print(dfscore_train)

fn = ['const', 'lnmktcapadj', 'bidaskadj', 'turnadj', 'pegadj', 'putcalladj', 'bmaadj', 'int_ebitadj', 'empgadj',
'revgradj', 'roeadj', 'cradj', 'capex_salesadj', 'lnpatentsadj', 'cfi_salesadj', 'insideradj', 'payoutadj',
'sentimentadj', 'deadj',
'bidaskmiss', 'pegmiss', 'putcallmiss', 'bmamiss', 'int_ebitmiss', 'empgmiss', 'revgrmiss', 'roemiss', 'crmiss',
'capex_salesmiss', 'lnpatentsmiss', 'cfi_salesmiss', 'insidermiss', 'payoutmiss', 'sentimentmiss', 'demiss']

plot_tree(dtmodel, feature_names=fn)
plt.savefig(path + 'Net tree explanation train 20240331_1835.pdf')
plt.show()
