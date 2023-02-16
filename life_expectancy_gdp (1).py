#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Install seaborn library
pip install seaborn


# In[ ]:


#install statsmodels
pip install statsmodels


# In[ ]:


pip install scikit-learn


# In[1]:


#Import usefull libraries for the proyect
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# In[2]:


#Turn csv into a dataframe with a basic visualization
data = pd.read_csv("all_data.csv")
#print(data.head())
#display(data)
#print(data.columns)


# In[3]:


#rename due to practicity the life expectany column
data = data.rename(columns ={'Life expectancy at birth (years)': 'Life expec'})


# In[ ]:


#Explore counts and unique values
#print(data['Year'].unique())
#print(data['Year'].value_counts())
#print(data['Country'].unique())
#print(data['Country'].value_counts())


# In[4]:


#Statistical exploration of the data
#print(data.info())
#print(data.describe())
print(data['Year'])


# In[5]:


#Set time variable for time series
data['Year'] = data['Year'].apply(pd.to_datetime, format='%Y').dt.year
print(data['Year'].head())
print(data.columns)
display(data)


# In[6]:


#Create a subset of the IQR of the data to exclude outliers
data_var=data
Q1 = data_var['GDP'].quantile(0.25)
Q3 = data_var['GDP'].quantile(0.75)
IQR = Q3 - Q1
data_var = data_var[(data_var['GDP'] >= Q1 - 1.5*IQR) & (data_var['GDP'] <= Q3 + 1.5*IQR)]
Q1 = data_var['Life expec'].quantile(0.25)
Q3 = data_var['Life expec'].quantile(0.75)
IQR = Q3 - Q1
data_var = data_var[(data_var['Life expec'] >= Q1 - 1.5*IQR) & (data_var['Life expec'] <= Q3 + 1.5*IQR)]
Q1 = data_var['GDP'].quantile(0.25)
Q3 = data_var['GDP'].quantile(0.75)
IQR = Q3 - Q1
data_var = data_var[(data_var['GDP'] >= Q1 - 1.5*IQR) & (data_var['GDP'] <= Q3 + 1.5*IQR)]
Q1 = data_var['Life expec'].quantile(0.25)
Q3 = data_var['Life expec'].quantile(0.75)
IQR = Q3 - Q1
data_var = data_var[(data_var['Life expec'] >= Q1 - 1.5*IQR) & (data_var['Life expec'] <= Q3 + 1.5*IQR)]


# In[51]:


#Plot a histogram for GDP and Life expectancy with the new IQR data
plt.hist(data_var['GDP'], color='green', bins=20, alpha=0.5)
plt.title('GDP IQR')
plt.show()
plt.clf()
plt.hist(data_var['Life expec'], color='green', bins=20, alpha=0.5)
plt.title('Life expectancy IQR')
plt.show()


# In[72]:


#Plot a lineplot to visualize tendencies on GDP for all six countries
sns.lineplot(x='Year',y='GDP',hue='Country',data=data, palette='gist_earth')
sns.light_palette('seagreen')
plt.title("GDP tendencies")
plt.xlabel("Year")
plt.ylabel("GDP")
plt.xticks(data['Year'],rotation=45)
plt.legend(loc='upper', bbox_to_anchor=(1.0, 0.5))
sns.set_style('whitegrid')
plt.show()


# In[71]:


#Plot a lineplot to visualize the tendencies in life expectancy for all six countries
sns.lineplot(x='Year',y='Life expec', hue='Country', data=data, palette='gist_earth')
plt.title('Life expectancy tendencies')
plt.xticks(data['Year'], rotation=45)
plt.ylabel('Life Expectancy')
plt.xlabel('Year')
plt.legend(loc='upper left',bbox_to_anchor=(1.0, 0.5))
sns.set_style('whitegrid')
plt.show()


# In[10]:


#plot a scatterplot to visualize the relationship between GDP and life expectancy per year
g = sns.FacetGrid(data, col="Year", hue="Country", col_wrap = 4, palette='gist_earth')
g.map(sns.scatterplot, "GDP", "Life expec", alpha=.7)
g.add_legend()
#No significant tendencies found
#Dead end


# In[11]:


# Plot a series of scatter plot (x=GDP, y=Life expectancy, per country)
g = sns.FacetGrid(data, col="Country", col_wrap = 3, palette='gist_earth')
g.map(sns.scatterplot, "GDP", "Life expec", alpha=.7)
g.add_legend()


# In[12]:


#Create different dataframes, one for each country, to perform independent scatterplots and adjust the scales
country_groups = data.groupby('Country')
chile = country_groups.get_group('Chile')
china = country_groups.get_group('China')
germany = country_groups.get_group('Germany')
mexico = country_groups.get_group('Mexico')
USA = country_groups.get_group('United States of America')
zimbabwe = country_groups.get_group('Zimbabwe')


# In[13]:


#Explore the data as a country subset
country_list=[chile, china, germany, mexico, USA, zimbabwe]
for i in country_list:
    print(i.info())
    display(i)


# In[62]:


sns.violinplot(x='Year', y='Life expec', data=data, palette='gist_earth')
plt.xticks(rotation = 30)
plt.title('Violin plot of Life Expectancy')
plt.show()
plt.clf()
sns.violinplot(x='Year', y='GDP', data=data, palette='gist_earth')
plt.xticks(rotation=30)
plt.title('Violinplot fo GDP')
plt.show()


# In[18]:


#Create a scatterplot for every country to see the GDP to rilefe expectancy relationship
sns.scatterplot(x='GDP', y= 'Life expec', data = chile, color='green')
plt.title('GDP to life expectancy relationship in Chile (2000-2015)')
plt.show()
plt.clf()
sns.scatterplot(x='GDP', y='Life expec', data= china, color='green')
plt.title('GDP to life expectancy relationship in China (2000-2015')
plt.show()
plt.clf()
sns.scatterplot(x='GDP', y='Life expec', data=germany, color='green')
plt.title('GDP to life expectancy relationship in Germany (2000-2015)')
plt.show()
plt.clf()
sns.scatterplot(x='GDP', y='Life expec', data=mexico, color='green')
plt.title('GDP to life expectany relationship in Mexico (2000-2015)')
plt.show()
plt.clf()
sns.scatterplot(x='GDP', y='Life expec', data=USA, color='green')
plt.title('GDP to life expectancy relationship in USA (2000-2015)')
plt.show()
plt.clf()
sns.scatterplot(x='GDP', y='Life expec', data=zimbabwe, color='green')
plt.title('GDP to life expectancy relationship on Zimbabwe (2000-2015)')
plt.show()


# In[63]:


#A scatterplot visualizing the relationship between GDP and Life Expectancy
sns.scatterplot(x='GDP', y='Life expec',hue='Country', data=data, palette='gist_earth')
plt.show()

#A scatterplot visualizing the relationship between GDP and Life Expectancy using the IQR data followed by a regression line 
y = np.array(data_var['Life expec'])
x = np.array(data_var['GDP'])
sns.scatterplot(x='GDP', y='Life expec', hue='Country', data=data_var, palette='gist_earth')
model = LinearRegression()
X = x[:, np.newaxis]
model.fit(X, y)
y_pred = model.predict(X)
plt.plot(x, y_pred, color='red')
plt.title('Linear regression between GDP and Life Expectancy')
plt.show()


# In[64]:


#Regression model test cheking for normality and the homoscedasticity of residuals
import statsmodels.api as sm
model = sm.OLS(y, X).fit()

model = sm.OLS(y, X).fit()
resid = model.resid
plt.hist(resid, bins=20, color='green', alpha=0.5)
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.title('Cheking for the normality of residuals for GDP to Life Expectancy Relationship')
plt.show()

pred = model.predict(X)
plt.scatter(pred, resid, color='green', alpha=0.5)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Homoscedasticity of Residuals for GDP to Life Expectancy Relationship')
plt.show()
bp_test = sm.stats.diagnostic.het_breuschpagan(resid, X)
print('Lagrange multiplier statistic: {:.2f}'.format(bp_test[0]))
print('p-value: {:.4f}'.format(bp_test[1]))
print('f-value: {:.2f}'.format(bp_test[2]))
print('f p-value: {:.4f}'.format(bp_test[3]))
print('The data show a perfect separation, meaning that while each country may have a relationship between the variation of GDP and the variation of life expectancy, as a set , this separtion means no mathematically relationship can be found using linear regression')
print('Additionally, we cant perform a linear regression on each country because there is know enough data to find certanty')


# In[37]:


#Create a variable for the GDP and life expectancy change
data['GDP_var'] = data['GDP'].pct_change()
data['Life_expec_var'] = data['Life expec'].pct_change()
print(data.describe())

data_var['GDP_var'] = data_var['GDP'].pct_change()
data_var['Life_expec_var'] = data_var['Life expec'].pct_change()
data_var = data_var.dropna()
display(data_var)


# In[38]:


#graph the new variables
sns.histplot(x='GDP_var', data=data,hue='Country', palette='gist_earth')
plt.show()
plt.clf()
sns.histplot(x='Life_expec_var', data=data,hue='Country', palette='gist_earth')
plt.show()


# In[39]:


#GDP and life expectancy histograms using the IQR dataset
sns.histplot(x='GDP_var', data=data_var, hue='Country', palette='gist_earth', alpha=0.5)
plt.show()
plt.clf()
sns.histplot(x='Life_expec_var', data=data_var, hue='Country', palette='gist_earth', alpha=0.5)
plt.show()


# In[65]:


#create a scatterplot to visualize the relationship between the variation of the GDP and the variation of the life expectancy
y = np.array(data_var['Life_expec_var'])
x = np.array(data_var['GDP_var'])
sns.scatterplot(x='GDP_var', y='Life_expec_var', hue='Country', data=data_var, palette='gist_earth')
model = LinearRegression()
X = x[:, np.newaxis]
model.fit(X, y)
y_pred = model.predict(X)
plt.plot(x, y_pred, color='red')
plt.title('Linear regression for the GDP change to Life Expectancy Change relationship')
plt.show()


# In[66]:


import statsmodels.api as sm
model = sm.OLS(y, X).fit()

model = sm.OLS(y, X).fit()
resid = model.resid
plt.hist(resid, bins=20, color='green', alpha=0.5)
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.title('Cheking for the normality of residuals for the GDP change to Life Expectancy Chenge Relationship')
plt.show()

pred = model.predict(X)
plt.scatter(pred, resid, color='green', alpha=0.5)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Homoscedasticity of Residuals for the GDP change to Life Expectancy Chenge Relationship')
plt.show()
bp_test = sm.stats.diagnostic.het_breuschpagan(resid, X)
print('Lagrange multiplier statistic: {:.2f}'.format(bp_test[0]))
print('p-value: {:.4f}'.format(bp_test[1]))
print('f-value: {:.2f}'.format(bp_test[2]))
print('f p-value: {:.4f}'.format(bp_test[3]))

print('The data show a perfect separation, meaning that while each country may have a relationship between the variation of GDP and the variation of life expectancy, as a set , this separtion means no mathematically relationship can be found using linear regression')
print('Additionally, we cant perform a linear regression on each country because there is know enough data to find certanty')

