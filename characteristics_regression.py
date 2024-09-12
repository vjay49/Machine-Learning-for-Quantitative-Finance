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
returns1 = pd.read_excel(path + 'Compiled stock returns and characteristics 20240131.xlsx', sheet_name='returns07')

print(returns1.head())
print(returns1.columns)

# Compile dsecriptive statistics
mean = returns1[['abret', 'mktcapreal', 'lnmktcap', 'bidaskalt', 'turn', 'bidaskmiss']].mean()
stddev = returns1[['abret', 'mktcapreal', 'lnmktcap', 'bidaskalt', 'turn', 'bidaskmiss']].std()
percentiles = returns1[['abret', 'mktcapreal','lnmktcap', 'bidaskalt', 'turn', 'bidaskmiss']].quantile([0.125, 0.500, 0.875])
print(mean)
print(stddev)
print(percentiles)

# What are the correlations between variables
correlation_matrix = returns1[['abret', 'lnmktcap', 'bidaskalt', 'turn', 'bidaskmiss']].corr()
print(correlation_matrix)

# Regression using statsmodels
# Abnormal stock returns regressed on LN mkt cap, bid-ask spread, turnover and an
# indicator variable for instances in which the bid-ask spread is missing.
y = returns1['abret']
x = returns1[['lnmktcap', 'bidaskalt', 'turn', 'bidaskmiss']]
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

print(returns1[['retmonth', 'Name', 'abret', 'lnmktcap', 'bidaskalt', 'turn', 'bidaskmiss']])
analysis = analysis.sort_index(ascending=True)
print(analysis)
returns2 = returns1.join(analysis, how='inner')
returns2 = returns2.sort_values(by='cooks_d', ascending=False)
print(returns2[['retmonth', 'Name', 'abret', 'lnmktcap', 'bidaskalt', 'turn', 'bidaskmiss',
                'predictions', 'cooks_d']])

# Run the regression again after removing observations for which Cook's D > 0.003

# Regression using statsmodels
# Abnormal stock returns regressed on LN mkt cap, bid-ask spread, turnover and an
# indicator variable for instances in which the bid-ask spread is missing.
returns3 = returns2[returns2['cooks_d'] <= 0.003]
# print(returns3)
y = returns3['abret']
x = returns3[['lnmktcap', 'bidaskalt', 'turn', 'bidaskmiss']]
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

# Export to Excel
with pd.ExcelWriter('Characteristics regression 20240131.xlsx') as writer:
    mean.to_excel(writer, sheet_name='mean')
    stddev.to_excel(writer, sheet_name='stddev')
    percentiles.to_excel(writer, sheet_name='percentiles')
    b_coef.to_excel(writer, sheet_name='b_coef')
    b_err.to_excel(writer, sheet_name='b_err')
