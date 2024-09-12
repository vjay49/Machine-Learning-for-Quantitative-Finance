# Import libraries
import pandas as pd
import statsmodels.api as sm

# Change the number of rows and columns to display
pd.set_option('display.max_rows', 200)
pd.set_option('display.min_rows', 200)
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_colwidth', 50)
pd.set_option('display.precision', 4)
pd.options.display.float_format = '{:.4f}'.format

# Define a path for import and export
path = 'E:\\UM\Courses\\FIN427\\FIN427 Winter 2024\\'

# Import data
returns1 = pd.read_excel(path + 'Aggregate data 20240304_2134.xlsx', sheet_name='returns20a')

print(returns1.head())
print(returns1.columns)

# Compile dsecriptive statistics
mean = returns1[['rettr', 'abrettr', 'lnmktcaptr', 'bidasktr', 'turntr', 'pegtr',
                 'putcalltr', 'bmatr', 'int_ebittr', 'empgtr', 'revgrtr', 'roetr',
                 'crtr', 'capex_salestr', 'patentstr', 'lnpatentstr', 'cfi_salestr', 'insidertr',
                 'payouttr','sentimenttr']].mean()
stddev = returns1[['rettr', 'abrettr', 'lnmktcaptr', 'bidasktr', 'turntr', 'pegtr',
                 'putcalltr', 'bmatr', 'int_ebittr', 'empgtr', 'revgrtr', 'roetr',
                 'crtr', 'capex_salestr', 'patentstr', 'lnpatentstr', 'cfi_salestr', 'insidertr',
                 'payouttr','sentimenttr']].std()
percentiles = returns1[['rettr', 'abrettr', 'lnmktcaptr', 'bidasktr', 'turntr', 'pegtr',
                 'putcalltr', 'bmatr', 'int_ebittr', 'empgtr', 'revgrtr', 'roetr',
                 'crtr', 'capex_salestr', 'patentstr', 'lnpatentstr', 'cfi_salestr', 'insidertr',
                 'payouttr','sentimenttr']].quantile([0.125, 0.500, 0.875])
print(mean)
print(stddev)
print(percentiles)

# What are the correlations between variables
correlation_matrix = returns1[['rettr', 'abrettr', 'lnmktcaptr', 'bidasktr', 'turntr', 'pegtr',
                 'putcalltr', 'bmatr', 'int_ebittr', 'empgtr', 'revgrtr', 'roetr',
                 'crtr', 'capex_salestr', 'patentstr', 'lnpatentstr', 'cfi_salestr', 'insidertr',
                 'payouttr','sentimenttr']].corr()
print(correlation_matrix)

# Regression using statsmodels
y = returns1['abretadj']
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
print(returns2[['retmonth', 'Name', 'abret', 'predictions', 'cooks_d']])

# Run the regression again after removing observations for which Cook's D > 0.003

# Regression using statsmodels
returns3 = returns2[returns2['cooks_d'] <= 0.003]
y3 = returns3['abretadj']
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
with pd.ExcelWriter('Project 20240304_2222.xlsx') as writer:
    mean.to_excel(writer, sheet_name='mean')
    stddev.to_excel(writer, sheet_name='stddev')
    percentiles.to_excel(writer, sheet_name='percentiles')
    b_coef.to_excel(writer, sheet_name='b_coef')
    b_err.to_excel(writer, sheet_name='b_err')
