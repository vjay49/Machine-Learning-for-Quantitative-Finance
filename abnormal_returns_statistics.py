# Import libraries
import pandas as pd
import statsmodels.api as sm
import numpy as np
import datetime
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
from sklearn.model_selection import ParameterGrid

# Change the number of rows and columns to display
pd.set_option('display.max_rows', 200)
pd.set_option('display.min_rows', 200)
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_colwidth', 50)
pd.set_option('display.precision', 5)
pd.options.display.float_format = '{:.5f}'.format

# Define a path for import and export
path = 'E:\\UM\Courses\\FIN427\\FIN427 Winter 2024\\'

# Import data
returns1 = pd.read_excel(path + 'Aggregate data 20240322_1515.xlsx', sheet_name='returns23')

print(returns1.head())
print(returns1.columns)

# Compile descriptive statistics
mean = returns1[['rettr', 'abrettr', 'indabrettr', 'mktcapreal', 'lnmktcaptr', 'bidasktr', 'turntr', 'pegtr',
                 'putcalltr', 'bmatr', 'int_ebittr', 'empgtr', 'revgrtr', 'roetr',
                 'crtr', 'capex_salestr', 'patentstr', 'lnpatentstr', 'cfi_salestr', 'insidertr',
                 'payouttr','sentimenttr']].mean()
stddev = returns1[['rettr', 'abrettr', 'indabrettr', 'mktcapreal', 'lnmktcaptr', 'bidasktr', 'turntr', 'pegtr',
                 'putcalltr', 'bmatr', 'int_ebittr', 'empgtr', 'revgrtr', 'roetr',
                 'crtr', 'capex_salestr', 'patentstr', 'lnpatentstr', 'cfi_salestr', 'insidertr',
                 'payouttr','sentimenttr']].std()
percentiles = returns1[['rettr', 'abrettr', 'indabrettr', 'mktcapreal', 'lnmktcaptr', 'bidasktr', 'turntr', 'pegtr',
                 'putcalltr', 'bmatr', 'int_ebittr', 'empgtr', 'revgrtr', 'roetr',
                 'crtr', 'capex_salestr', 'patentstr', 'lnpatentstr', 'cfi_salestr', 'insidertr',
                 'payouttr','sentimenttr']].quantile([0, 0.125, 0.500, 0.875, 1])
print(mean)
print(stddev)
print(percentiles)

# What are the correlations between variables
correlation_matrix = returns1[['rettr', 'abrettr', 'indabrettr', 'lnmktcaptr', 'bidasktr', 'turntr', 'pegtr',
                 'putcalltr', 'bmatr', 'int_ebittr', 'empgtr', 'revgrtr', 'roetr',
                 'crtr', 'capex_salestr', 'patentstr', 'lnpatentstr', 'cfi_salestr', 'insidertr',
                 'payouttr','sentimenttr']].corr()
print(correlation_matrix)

# Regression using statsmodels
y = returns1['indabrettr']
x = returns1[['lnmktcapadj', 'bidaskadj', 'turnadj', 'pegadj', 'putcalladj', 'bmaadj', 'int_ebitadj',
'empgadj', 'revgradj', 'roeadj', 'cradj', 'capex_salesadj', 'lnpatentsadj', 'cfi_salesadj', 'insideradj',
'payoutadj', 'sentimentadj',
'bidaskmiss', 'pegmiss', 'putcallmiss', 'bmamiss', 'int_ebitmiss', 'empgmiss', 'revgrmiss', 'roemiss', 'crmiss',
'capex_salesmiss', 'lnpatentsmiss', 'cfi_salesmiss', 'insidermiss', 'payoutmiss', 'sentimentmiss']]

x = sm.add_constant(x)
model = sm.OLS(y, x).fit()
# To generate clustered standard errors use the line below
# model = sm.OLS(y, x).fit(cov_type='cluster', cov_kwds={'groups': returns1['ISIN']})
predictions = model.predict(x)
print_model = model.summary()
b_coef = model.params
b_err = model.bse

influence = model.get_influence()
cooks_d = influence.cooks_distance[0]
analysis = pd.DataFrame({'predictions': predictions, 'cooks_d': cooks_d})
analysis = analysis.sort_values(by='cooks_d', ascending=False)

print(print_model)
print(f'R-squared: {model.rsquared:.4f}')
print(b_coef)
print(b_err)
print(analysis.columns)
print(analysis)

# Merge analysis with original dataset to run the regression again after removing
# influential observations

analysis = analysis.sort_index(ascending=True)
returns2 = returns1.join(analysis, how='inner')
returns2 = returns2.sort_values(by='cooks_d', ascending=False)
print(returns2[['retmonth', 'Name', 'indabrettr', 'predictions', 'cooks_d']])

# Run the regression again after removing observations for which Cook's D > 0.003

# Regression using statsmodels
returns3 = returns2[returns2['cooks_d'] <= 0.003]
y3 = returns3['indabrettr']
x3 = returns3[['lnmktcapadj', 'bidaskadj', 'turnadj', 'pegadj', 'putcalladj', 'bmaadj', 'int_ebitadj',
'empgadj', 'revgradj', 'roeadj', 'cradj', 'capex_salesadj', 'lnpatentsadj', 'cfi_salesadj', 'insideradj',
'payoutadj', 'sentimentadj',
'bidaskmiss', 'pegmiss', 'putcallmiss', 'bmamiss', 'int_ebitmiss', 'empgmiss', 'revgrmiss', 'roemiss', 'crmiss',
'capex_salesmiss', 'lnpatentsmiss', 'cfi_salesmiss', 'insidermiss', 'payoutmiss', 'sentimentmiss']]
x3 = sm.add_constant(x3)
model3 = sm.OLS(y3, x3).fit()
# To generate clustered standard errors use the line below
# model = sm.OLS(y, x).fit(cov_type='cluster', cov_kwds={'groups': returns1['ISIN']})
predictions3 = model3.predict(x3)
print_model3 = model3.summary()
b_coef3 = model3.params
b_err3 = model3.bse

influence3 = model3.get_influence()
cooks_d3 = influence.cooks_distance[0]
analysis3 = pd.DataFrame({'predictions': predictions, 'cooks_d': cooks_d})
analysis3 = analysis3.sort_values(by='cooks_d', ascending=False)

print(print_model3)
print(f'R-squared: {model3.rsquared:.4f}')
print(b_coef3)
print(b_err3)
print(analysis3.columns)
print(analysis3)

# Export to Excel
with pd.ExcelWriter('Project 20240322_1529.xlsx') as writer:
    mean.to_excel(writer, sheet_name='mean')
    stddev.to_excel(writer, sheet_name='stddev')
    percentiles.to_excel(writer, sheet_name='percentiles')
    correlation_matrix.to_excel(writer, sheet_name='correlation_matrix')
    b_coef.to_excel(writer, sheet_name='b_coef')
    b_err.to_excel(writer, sheet_name='b_err')

# OLS using training and validation datasets
# The dataset has 336 months of returns from January 1995 to December 2022
datecut = datetime.datetime(2016, 12, 31)
print(datecut)
returns_train = returns1[returns1['retmonth'] <= datecut]
returns_test  = returns1[returns1['retmonth'] >  datecut]
print(returns_train.head(5))
print(returns_train.tail(5))
print(returns_test.head(5))
print(returns_test.tail(5))

y_train = returns_train['indabrettr']
x_train = returns_train[['lnmktcapadj', 'bidaskadj', 'turnadj', 'pegadj', 'putcalladj', 'bmaadj',
                         'int_ebitadj',
'empgadj', 'revgradj', 'roeadj', 'cradj', 'capex_salesadj', 'lnpatentsadj', 'cfi_salesadj', 'insideradj',
'payoutadj', 'sentimentadj',
'bidaskmiss', 'pegmiss', 'putcallmiss', 'bmamiss', 'int_ebitmiss', 'empgmiss', 'revgrmiss', 'roemiss', 'crmiss',
'capex_salesmiss', 'lnpatentsmiss', 'cfi_salesmiss', 'insidermiss', 'payoutmiss', 'sentimentmiss']]

y_test = returns_test['indabrettr']
x_test = returns_test[['lnmktcapadj', 'bidaskadj', 'turnadj', 'pegadj', 'putcalladj', 'bmaadj',
                       'int_ebitadj',
'empgadj', 'revgradj', 'roeadj', 'cradj', 'capex_salesadj', 'lnpatentsadj', 'cfi_salesadj', 'insideradj',
'payoutadj', 'sentimentadj',
'bidaskmiss', 'pegmiss', 'putcallmiss', 'bmamiss', 'int_ebitmiss', 'empgmiss', 'revgrmiss', 'roemiss', 'crmiss',
'capex_salesmiss', 'lnpatentsmiss', 'cfi_salesmiss', 'insidermiss', 'payoutmiss', 'sentimentmiss']]

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
print(x_train.columns)
print(x_test.columns)

# OLS regression with out-of-sample prediction
x_train = sm.add_constant(x_train)
x_test = sm.add_constant(x_test)
model = sm.OLS(y_train, x_train).fit()
print_model = model.summary()
ols_coef = model.params
ols_rsq = model.rsquared

y_pred_train = model.predict(x_train)
ssr_train = np.sum((y_train - y_pred_train)**2)
sst_train = np.sum((y_train - np.mean(y_train))**2)
rsq_train = 1 - (ssr_train/sst_train)
rmse_train = np.sqrt(np.mean((y_train - y_pred_train)**2))

y_pred_test = model.predict(x_test)
ssr_test = np.sum((y_test - y_pred_test)**2)
sst_test = np.sum((y_test - np.mean(y_test))**2)
rsq_test = 1 - (ssr_test/sst_test)
rmse_test = np.sqrt(np.mean((y_test - y_pred_test)**2))

print(print_model)
print(ols_coef)

print(f'R-squared in training sample from rsquared function: {model.rsquared:.5f}')
print(f'Sum of squared difference between y values and predicted y values in training sample (SSR): {ssr_train:.5f}')
print(f'Sum of squared difference between y values and average y values in training sample (SST): {sst_train:.5f}')
print(f'R-squared in training sample = 1 - SSR/SST: {rsq_train:.5f}')
print(f'Square root of the mean squared error in training sample: {rmse_train:.5f}')

print(f'Sum of squared difference between y values and predicted y values in test sample (SSR): {ssr_test:.5f}')
print(f'Sum of squared difference between y values and average y values in test sample (SST): {sst_test:.5f}')
print(f'R-squared in test sample = 1 - SSR/SST: {rsq_test:.5f}')
print(f'Square root of the mean squared error in test sample: {rmse_test:.5f}')

# Regression on test sample
model = sm.OLS(y_test, x_test).fit()
print_model = model.summary()
ols_coef = model.params
ols_rsq = model.rsquared
print(print_model)
print(ols_coef)

y_pred_test = model.predict(x_test)
ssr_test = np.sum((y_test - y_pred_test)**2)
sst_test = np.sum((y_test - np.mean(y_test))**2)
rsq_test = 1 - (ssr_test/sst_test)
rmse_test = np.sqrt(np.mean((y_test - y_pred_test)**2))

print(f'Sum of squared difference between y values and predicted y values in test sample (SSR): {ssr_test:.5f}')
print(f'Sum of squared difference between y values and average y values in test sample (SST): {sst_test:.5f}')
print(f'R-squared in test sample = 1 - SSR/SST: {rsq_test:.5f}')
print(f'Square root of the mean squared error in test sample: {rmse_test:.5f}')

# LASSO regression

# Specify the penalty factor
lasso = Lasso(alpha=0.0001)

# LASSO regression on the full sample
lasso.fit(x, y)
lasso_coef = lasso.fit(x, y).coef_
lasso_score = lasso.score(x, y)
print(lasso.intercept_)
print(lasso_coef)
print(x.columns)
print(pd.Series(lasso_coef, index=x.columns))
y_pred = lasso.predict(x)
ssr = np.sum((y - y_pred)**2)
sst = np.sum((y - np.mean(y))**2)
rsq = 1 - (ssr/sst)
rmse = np.sqrt(np.mean((y - y_pred)**2))
print(f'Sum of squared difference between y values and predicted y values (SSR): {ssr:.5f}')
print(f'Sum of squared difference between y values and average y values (SST): {sst:.5f}')
print(f'R-squared = 1 - SSR/SST: {rsq:.5f}')
print(f'Square root of the mean squared error: {rmse:.5f}')

# LASSO regression with out-of-sample prediction

lasso.fit(x_train, y_train)
lasso_coef_train = lasso.fit(x_train, y_train).coef_
lasso_score_train = lasso.score(x_train, y_train)
print(lasso.intercept_)
print(lasso_coef_train)
print(x_train.columns)
print(pd.Series(lasso_coef, index=x_train.columns))

y_pred_train = lasso.predict(x_train)
ssr_train = np.sum((y_train - y_pred_train)**2)
sst_train = np.sum((y_train - np.mean(y_train))**2)
rsq_train = 1 - (ssr_train/sst_train)
rmse_train = np.sqrt(np.mean((y_train - y_pred_train)**2))
print(f'Sum of squared difference between y values and predicted y values in training sample (SSR): {ssr_train:.5f}')
print(f'Sum of squared difference between y values and average y values in training sample (SST): {sst_train:.5f}')
print(f'R-squared in training sample = 1 - SSR/SST: {rsq_train:.5f}')
print(f'Square root of the mean squared error in training sample: {rmse_train:.5f}')

y_pred_test = lasso.predict(x_test)
ssr_test = np.sum((y_test - y_pred_test)**2)
sst_test = np.sum((y_test - np.mean(y_test))**2)
rsq_test = 1 - (ssr_test/sst_test)
rmse_test = np.sqrt(np.mean((y_test - y_pred_test)**2))
print(f'Sum of squared difference between y values and predicted y values in test sample (SSR): {ssr_test:.5f}')
print(f'Sum of squared difference between y values and average y values in test sample (SST): {sst_test:.5f}')
print(f'R-squared in test sample = 1 - SSR/SST: {rsq_test:.5f}')
print(f'Square root of the mean squared error in test sample: {rmse_test:.5f}')

lasso.fit(x_test, y_test)
lasso_coef_test = lasso.fit(x_test, y_test).coef_
lasso_score_test = lasso.score(x_test, y_test)
print(lasso.intercept_)
print(lasso_coef_test)
print(x_test.columns)
print(pd.Series(lasso_coef_test, index=x_test.columns))

y_pred_test = lasso.predict(x_test)
ssr_test = np.sum((y_test - y_pred_test)**2)
sst_test = np.sum((y_test - np.mean(y_test))**2)
rsq_test = 1 - (ssr_test/sst_test)
rmse_test = np.sqrt(np.mean((y_test - y_pred_test)**2))
print(f'Sum of squared difference between y values and predicted y values in test sample (SSR): {ssr_test:.5f}')
print(f'Sum of squared difference between y values and average y values in test sample (SST): {sst_test:.5f}')
print(f'R-squared in test sample = 1 - SSR/SST: {rsq_test:.5f}')
print(f'Square root of the mean squared error in test sample: {rmse_test:.5f}')

# What if we could run the program many times with different alpha?

alphalist_full = []
rsq_fullsum = []
alphalist = []
rsq_trainsum = []
rsq_testsum = []

for i in range(1, 11):
    alpha_cand_full = i/20000
    lasso = Lasso(alpha=alpha_cand_full)
    lfit = lasso.fit(x, y)

    y_pred_full = lasso.predict(x)
    ssr_full = np.sum((y - y_pred_full) ** 2)
    sst_full = np.sum((y - np.mean(y)) ** 2)
    rsq_full = 1 - (ssr_full / sst_full)

    alphalist_full.append(alpha_cand_full)
    rsq_fullsum.append(rsq_full)

lassosum_full = pd.DataFrame(list(zip(alphalist_full, rsq_fullsum)),
                        columns=['alphalist_full', 'rsq_fullsum'])
print(lassosum_full)

for i in range(1, 11):
    alpha_cand = i/20000
    lasso = Lasso(alpha=alpha_cand)
    lfit = lasso.fit(x_train, y_train)

    y_pred_train = lasso.predict(x_train)
    ssr_train = np.sum((y_train - y_pred_train) ** 2)
    sst_train = np.sum((y_train - np.mean(y_train)) ** 2)
    rsq_train = 1 - (ssr_train / sst_train)

    y_pred_test = lasso.predict(x_test)
    ssr_test = np.sum((y_test - y_pred_test) ** 2)
    sst_test = np.sum((y_test - np.mean(y_test)) ** 2)
    rsq_test = 1 - (ssr_test / sst_test)

    alphalist.append(alpha_cand)
    rsq_trainsum.append(rsq_train)
    rsq_testsum.append(rsq_test)

lassosum = pd.DataFrame(list(zip(alphalist, rsq_trainsum, rsq_testsum)),
                        columns=['alphalist', 'rsq_trainsum', 'rsq_testsum'])
print(lassosum)

with pd.ExcelWriter(path + 'Lasso penalty grid 20240322_1532.xlsx') as writer:
    lassosum_full.to_excel(writer, sheet_name='lassosum_full')
    lassosum.to_excel(writer, sheet_name='lassosum')

# Decision-tree analysis
# Code that relates to the full sample, training set and test set
dt = DecisionTreeRegressor(max_depth=10, min_weight_fraction_leaf=0.05)
fn = ['const', 'lnmktcapadj', 'bidaskadj', 'turnadj', 'pegadj', 'putcalladj', 'bmaadj', 'int_ebitadj', 'empgadj', 'revgradj',
'roeadj', 'cradj', 'capex_salesadj', 'lnpatentsadj', 'cfi_salesadj', 'insideradj', 'payoutadj', 'sentimentadj',
'bidaskmiss', 'pegmiss', 'putcallmiss', 'bmamiss', 'int_ebitmiss', 'empgmiss', 'revgrmiss', 'roemiss', 'crmiss',
'capex_salesmiss', 'lnpatentsmiss', 'cfi_salesmiss', 'insidermiss', 'payoutmiss', 'sentimentmiss']

# Decision-tree analysis on the full dataset
dtmodel = dt.fit(x, y)
dtresult = dt.score(x, y)
dtpredictions = dt.predict(x)
dfscore = pd.DataFrame([dtresult])
dfpredictions = pd.DataFrame([dtpredictions])
dfreview = pd.DataFrame({'Name': returns1['Name'], 'ISIN': returns1['ISIN'],
                         'retmonth': returns1['retmonth'], 'indabrettr': y, 'pred_indabrettr': dtpredictions})
print(dfreview)

plot_tree(dtmodel, feature_names=fn)
plt.savefig(path + 'Basic decision tree 20240322_1533.pdf')
plt.show()

# What is the relative importance of each feature in lowering mean squared error?
# There is tabular output and a chart, the chart does not display well because of the number of features.
# I have left the chart code in the file in case you want it for another project.
importances = dt.feature_importances_
sorted_index = np.argsort(importances)[::-1]
ximportance = range(len(importances))
labels = np.array(fn)[sorted_index]
plt.bar(ximportance, importances[sorted_index], tick_label=labels)
plt.xticks(rotation=90)
plt.savefig(path + 'Basic decision tree importance 20240318_1537.pdf')
plt.show()

dfimportance = pd.DataFrame(list(zip(labels, importances[sorted_index])), columns=['fn', 'importances'])

print('Decision-tree score:', dtresult)
print('Mean Absolute Error:', metrics.mean_absolute_error(y, dtpredictions))
print('Mean Squared Error:', metrics.mean_squared_error(y, dtpredictions))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y, dtpredictions)))
print('Features sorted by importance:', dfimportance)

# Decision-tree analysis with training and test datasets

dtmodel_train = dt.fit(x_train, y_train)
dtresult_train = dt.score(x_train, y_train)
dtpredictions_train = dt.predict(x_train)
dfscore_train = pd.DataFrame([dtresult_train])
dfpredictions_train = pd.DataFrame([dtpredictions_train])
dfreview_train = pd.DataFrame({'Name': returns_train['Name'], 'ISIN': returns_train['ISIN'],
'retmonth': returns_train['retmonth'], 'indabrettr': y_train, 'pred_indabrettr': dtpredictions_train})
plot_tree(dtmodel_train, feature_names=fn)
plt.savefig(path + 'Basic decision tree train 20240322_1533.pdf')
plt.show()

# What is the relative importance of each feature in lowering mean squared error?
# There is tabular output and a chart, the chart does not display well because of the number of features.
# I have left the chart code in the file in case you want it for another project.
importances_train = dt.feature_importances_
sorted_index_train = np.argsort(importances_train)[::-1]
ximportance_train = range(len(importances_train))
labels_train = np.array(fn)[sorted_index_train]
plt.bar(ximportance_train, importances_train[sorted_index_train], tick_label=labels)
plt.xticks(rotation=90)
plt.savefig(path + 'Basic decision tree importance train 20240318_1537.pdf')
plt.show()

dfimportance_train = pd.DataFrame(list(zip(labels, importances_train[sorted_index_train])),
                                  columns=['fn', 'importances_train'])

print('Decision-tree score:', dtresult_train)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_train, dtpredictions_train))
print('Mean Squared Error:', metrics.mean_squared_error(y_train, dtpredictions_train))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, dtpredictions_train)))
print('Features sorted by importance:', dfimportance_train)

dtresult_test = dt.score(x_test, y_test)
dtpredictions_test = dt.predict(x_test)
dfscore_test = pd.DataFrame([dtresult_test])
dfpredictions_test = pd.DataFrame([dtpredictions_test])
dfreview_test = pd.DataFrame({'Name': returns_test['Name'], 'ISIN': returns_test['ISIN'],
'retmonth': returns_test['retmonth'], 'indabrettr': y_test, 'pred_indabrettr': dtpredictions_test})

print('Decision-tree score:', dtresult_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, dtpredictions_test))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, dtpredictions_test))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, dtpredictions_test)))

# Consider many possibilities for tree depth and minimum weight in each final node with training and test datasets
grid = {'max_depth': [5, 10],
       'min_weight_fraction_leaf': [0.01, 0.05, 0.10]}
dt = DecisionTreeRegressor()
scores_full = []
scores_train = []
scores_test = []
glist_full = []
glist_split = []

for g in ParameterGrid(grid):
    dt.set_params(**g)  # ** is "unpacking" the dictionary
    dt.fit(x, y)
    glist_full.append(g)
    scores_full.append(dt.score(x, y))
results_full = pd.DataFrame(glist_full, scores_full)
print(results_full)

for g in ParameterGrid(grid):
    dt.set_params(**g)  # ** is "unpacking" the dictionary
    dt.fit(x_train, y_train)
    glist_split.append(g)
    scores_train.append(dt.score(x_train, y_train))
    scores_test.append(dt.score(x_test, y_test))
results_train = pd.DataFrame(glist_split, scores_train)
results_test = pd.DataFrame(glist_split, scores_test)
print(results_train)
print(results_test)

# Export results to Excel
with pd.ExcelWriter(path + 'Tree output many combinations 20240322_1541.xlsx') as writer:
    results_full.to_excel(writer, sheet_name='results_full')
    results_train.to_excel(writer, sheet_name='results_train')
    results_test.to_excel(writer, sheet_name='results_test')

# Export results to Excel
with pd.ExcelWriter(path + 'Decision-tree output 20240322_1541.xlsx') as writer:
    dfscore.to_excel(writer, sheet_name='Score')
    dfimportance.to_excel(writer, sheet_name='Importance')
    dfscore_train.to_excel(writer, sheet_name='Score_train')
    dfimportance_train.to_excel(writer, sheet_name='Importance_train')
    dfscore_test.to_excel(writer, sheet_name='Score_test')

# Analysis using z-scores on a monthly and industry basis.
# Transform dataset to z-scores on a monthly basis.
x1 = returns1[['retmonth', 'ICBIC', 'lnmktcapadj', 'bidaskadj', 'turnadj', 'pegadj', 'putcalladj', 'bmaadj', 'int_ebitadj',
'empgadj', 'revgradj', 'roeadj', 'cradj', 'capex_salesadj', 'lnpatentsadj', 'cfi_salesadj', 'insideradj',
'payoutadj', 'sentimentadj',
'bidaskmiss', 'pegmiss', 'putcallmiss', 'bmamiss', 'int_ebitmiss', 'empgmiss', 'revgrmiss', 'roemiss', 'crmiss',
'capex_salesmiss', 'lnpatentsmiss', 'cfi_salesmiss', 'insidermiss', 'payoutmiss', 'sentimentmiss']]
x1 = x1.groupby(['retmonth', 'ICBIC']).transform(lambda a: (a - a.mean()) / a.std())
x1.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
print(x1.head(100))
print(x1.shape)

x1 = sm.add_constant(x1)
model = sm.OLS(y, x1).fit()
# To generate clustered standard errors use the line below
# model = sm.OLS(y, x1).fit(cov_type='cluster', cov_kwds={'groups': returns1['ISIN']})
predictions = model.predict(x1)
print_model = model.summary()
b_coef = model.params
b_err = model.bse

influence = model.get_influence()
cooks_d = influence.cooks_distance[0]
analysis = pd.DataFrame({'predictions': predictions, 'cooks_d': cooks_d})
analysis = analysis.sort_values(by='cooks_d', ascending=False)

print(print_model)
print(f'R-squared: {model.rsquared:.4f}')
print(b_coef)
print(b_err)
print(analysis.columns)
print(analysis)

# OLS using training and validation datasets
x1_train = returns_train[['retmonth', 'ICBIC', 'lnmktcapadj', 'bidaskadj', 'turnadj', 'pegadj', 'putcalladj', 'bmaadj',
                         'int_ebitadj',
'empgadj', 'revgradj', 'roeadj', 'cradj', 'capex_salesadj', 'lnpatentsadj', 'cfi_salesadj', 'insideradj',
'payoutadj', 'sentimentadj',
'bidaskmiss', 'pegmiss', 'putcallmiss', 'bmamiss', 'int_ebitmiss', 'empgmiss', 'revgrmiss', 'roemiss', 'crmiss',
'capex_salesmiss', 'lnpatentsmiss', 'cfi_salesmiss', 'insidermiss', 'payoutmiss', 'sentimentmiss']]
x1_train = x1_train.groupby(['retmonth', 'ICBIC']).transform(lambda a: (a - a.mean()) / a.std())
x1_train.replace([np.inf, -np.inf, np.nan], 0, inplace=True)

x1_test = returns_test[['retmonth', 'ICBIC', 'lnmktcapadj', 'bidaskadj', 'turnadj', 'pegadj', 'putcalladj', 'bmaadj',
                       'int_ebitadj',
'empgadj', 'revgradj', 'roeadj', 'cradj', 'capex_salesadj', 'lnpatentsadj', 'cfi_salesadj', 'insideradj',
'payoutadj', 'sentimentadj',
'bidaskmiss', 'pegmiss', 'putcallmiss', 'bmamiss', 'int_ebitmiss', 'empgmiss', 'revgrmiss', 'roemiss', 'crmiss',
'capex_salesmiss', 'lnpatentsmiss', 'cfi_salesmiss', 'insidermiss', 'payoutmiss', 'sentimentmiss']]
x1_test = x1_test.groupby(['retmonth', 'ICBIC']).transform(lambda a: (a - a.mean()) / a.std())
x1_test.replace([np.inf, -np.inf, np.nan], 0, inplace=True)

# OLS regression with out-of-sample prediction
x1_train = sm.add_constant(x1_train)
x1_test = sm.add_constant(x1_test)
model = sm.OLS(y_train, x1_train).fit()
print_model = model.summary()
ols_coef = model.params
ols_rsq = model.rsquared

y_pred_train = model.predict(x1_train)
ssr_train = np.sum((y_train - y_pred_train)**2)
sst_train = np.sum((y_train - np.mean(y_train))**2)
rsq_train = 1 - (ssr_train/sst_train)
rmse_train = np.sqrt(np.mean((y_train - y_pred_train)**2))

y_pred_test = model.predict(x1_test)
ssr_test = np.sum((y_test - y_pred_test)**2)
sst_test = np.sum((y_test - np.mean(y_test))**2)
rsq_test = 1 - (ssr_test/sst_test)
rmse_test = np.sqrt(np.mean((y_test - y_pred_test)**2))

print(print_model)
print(ols_coef)

print(f'R-squared in training sample from rsquared function: {model.rsquared:.5f}')
print(f'Sum of squared difference between y values and predicted y values in training sample (SSR): {ssr_train:.5f}')
print(f'Sum of squared difference between y values and average y values in training sample (SST): {sst_train:.5f}')
print(f'R-squared in training sample = 1 - SSR/SST: {rsq_train:.5f}')
print(f'Square root of the mean squared error in training sample: {rmse_train:.5f}')

print(f'Sum of squared difference between y values and predicted y values in test sample (SSR): {ssr_test:.5f}')
print(f'Sum of squared difference between y values and average y values in test sample (SST): {sst_test:.5f}')
print(f'R-squared in test sample = 1 - SSR/SST: {rsq_test:.5f}')
print(f'Square root of the mean squared error in test sample: {rmse_test:.5f}')

# Regression on test sample
model = sm.OLS(y_test, x1_test).fit()
print_model = model.summary()
ols_coef = model.params
ols_rsq = model.rsquared
print(print_model)
print(ols_coef)

y_pred_test = model.predict(x1_test)
ssr_test = np.sum((y_test - y_pred_test)**2)
sst_test = np.sum((y_test - np.mean(y_test))**2)
rsq_test = 1 - (ssr_test/sst_test)
rmse_test = np.sqrt(np.mean((y_test - y_pred_test)**2))

print(f'Sum of squared difference between y values and predicted y values in test sample (SSR): {ssr_test:.5f}')
print(f'Sum of squared difference between y values and average y values in test sample (SST): {sst_test:.5f}')
print(f'R-squared in test sample = 1 - SSR/SST: {rsq_test:.5f}')
print(f'Square root of the mean squared error in test sample: {rmse_test:.5f}')

# LASSO regression

# Specify the penalty factor
lasso = Lasso(alpha=0.0001)

# LASSO regression on the full sample
lasso.fit(x1, y)
lasso_coef = lasso.fit(x1, y).coef_
lasso_score = lasso.score(x1, y)
print(lasso.intercept_)
print(lasso_coef)
print(x1.columns)
print(pd.Series(lasso_coef, index=x1.columns))
y_pred = lasso.predict(x1)
ssr = np.sum((y - y_pred)**2)
sst = np.sum((y - np.mean(y))**2)
rsq = 1 - (ssr/sst)
rmse = np.sqrt(np.mean((y - y_pred)**2))
print(f'Sum of squared difference between y values and predicted y values (SSR): {ssr:.5f}')
print(f'Sum of squared difference between y values and average y values (SST): {sst:.5f}')
print(f'R-squared = 1 - SSR/SST: {rsq:.5f}')
print(f'Square root of the mean squared error: {rmse:.5f}')

# LASSO regression with out-of-sample prediction

lasso.fit(x1_train, y_train)
lasso_coef_train = lasso.fit(x1_train, y_train).coef_
lasso_score_train = lasso.score(x1_train, y_train)
print(lasso.intercept_)
print(lasso_coef_train)
print(x1_train.columns)
print(pd.Series(lasso_coef, index=x1_train.columns))

y_pred_train = lasso.predict(x1_train)
ssr_train = np.sum((y_train - y_pred_train)**2)
sst_train = np.sum((y_train - np.mean(y_train))**2)
rsq_train = 1 - (ssr_train/sst_train)
rmse_train = np.sqrt(np.mean((y_train - y_pred_train)**2))
print(f'Sum of squared difference between y values and predicted y values in training sample (SSR): {ssr_train:.5f}')
print(f'Sum of squared difference between y values and average y values in training sample (SST): {sst_train:.5f}')
print(f'R-squared in training sample = 1 - SSR/SST: {rsq_train:.5f}')
print(f'Square root of the mean squared error in training sample: {rmse_train:.5f}')

y_pred_test = lasso.predict(x1_test)
ssr_test = np.sum((y_test - y_pred_test)**2)
sst_test = np.sum((y_test - np.mean(y_test))**2)
rsq_test = 1 - (ssr_test/sst_test)
rmse_test = np.sqrt(np.mean((y_test - y_pred_test)**2))
print(f'Sum of squared difference between y values and predicted y values in test sample (SSR): {ssr_test:.5f}')
print(f'Sum of squared difference between y values and average y values in test sample (SST): {sst_test:.5f}')
print(f'R-squared in test sample = 1 - SSR/SST: {rsq_test:.5f}')
print(f'Square root of the mean squared error in test sample: {rmse_test:.5f}')

lasso.fit(x1_test, y_test)
lasso_coef_test = lasso.fit(x1_test, y_test).coef_
lasso_score_test = lasso.score(x1_test, y_test)
print(lasso.intercept_)
print(lasso_coef_test)
print(x1_test.columns)
print(pd.Series(lasso_coef_test, index=x1_test.columns))

y_pred_test = lasso.predict(x1_test)
ssr_test = np.sum((y_test - y_pred_test)**2)
sst_test = np.sum((y_test - np.mean(y_test))**2)
rsq_test = 1 - (ssr_test/sst_test)
rmse_test = np.sqrt(np.mean((y_test - y_pred_test)**2))
print(f'Sum of squared difference between y values and predicted y values in test sample (SSR): {ssr_test:.5f}')
print(f'Sum of squared difference between y values and average y values in test sample (SST): {sst_test:.5f}')
print(f'R-squared in test sample = 1 - SSR/SST: {rsq_test:.5f}')
print(f'Square root of the mean squared error in test sample: {rmse_test:.5f}')

# What if we could run the program many times with different alpha?

alphalist_full = []
rsq_fullsum = []
alphalist = []
rsq_trainsum = []
rsq_testsum = []

for i in range(1, 11):
    alpha_cand_full = i/10000
    lasso = Lasso(alpha=alpha_cand_full)
    lfit = lasso.fit(x1, y)

    y_pred_full = lasso.predict(x1)
    ssr_full = np.sum((y - y_pred_full) ** 2)
    sst_full = np.sum((y - np.mean(y)) ** 2)
    rsq_full = 1 - (ssr_full / sst_full)

    alphalist_full.append(alpha_cand_full)
    rsq_fullsum.append(rsq_full)

lassosum_full = pd.DataFrame(list(zip(alphalist_full, rsq_fullsum)),
                        columns=['alphalist_full', 'rsq_fullsum'])
print(lassosum_full)

for i in range(1, 11):
    alpha_cand = i/10000
    lasso = Lasso(alpha=alpha_cand)
    lfit = lasso.fit(x1_train, y_train)

    y_pred_train = lasso.predict(x1_train)
    ssr_train = np.sum((y_train - y_pred_train) ** 2)
    sst_train = np.sum((y_train - np.mean(y_train)) ** 2)
    rsq_train = 1 - (ssr_train / sst_train)

    y_pred_test = lasso.predict(x1_test)
    ssr_test = np.sum((y_test - y_pred_test) ** 2)
    sst_test = np.sum((y_test - np.mean(y_test)) ** 2)
    rsq_test = 1 - (ssr_test / sst_test)

    alphalist.append(alpha_cand)
    rsq_trainsum.append(rsq_train)
    rsq_testsum.append(rsq_test)

lassosum = pd.DataFrame(list(zip(alphalist, rsq_trainsum, rsq_testsum)),
                        columns=['alphalist', 'rsq_trainsum', 'rsq_testsum'])
print(lassosum)

with pd.ExcelWriter(path + 'Lasso penalty grid with z scores 20240322_1617.xlsx') as writer:
    lassosum_full.to_excel(writer, sheet_name='lassosum_full')
    lassosum.to_excel(writer, sheet_name='lassosum')
