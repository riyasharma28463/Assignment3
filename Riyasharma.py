# Analysis of CO2 Emissions Data: Mean Values, Maximum and Minimum Emissions
# Riya sharma2
from scipy.optimize import curve_fit
import numpy as np
import itertools as iter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load the dataset
df = pd.read_csv('API_EN.ATM.CO2E.PC_DS2_en_csv_v2_5358914.csv')
df.describe()

# Select the columns to be used for clustering
cols = ['1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019']

# Impute NaN values with the mean of the corresponding column
df[cols] = df[cols].fillna(df[cols].mean())

# Normalize the data
scaler = StandardScaler()
df[cols] = scaler.fit_transform(df[cols])

# Replace NaN values with mean
df = df.fillna(df.mean())


# Alternatively, you can replace NaN values with 0 using the following code:
# df = df.fillna(0)

# Print the updated DataFrame
print(df)

# Use KMeans clustering algorithm to cluster the data
kmeans = KMeans(n_clusters=10, random_state=42)
df['Cluster'] = kmeans.fit_predict(df[cols])

df['Cluster'].value_counts()

# Use logical slicing to select the data for plotting
data = new_df.loc[df['Cluster'].isin([2, 0, 6,3,9,8, 7, 1, 5, 4])]

sns.scatterplot(data=data, x='2019', y='1991', hue='Cluster')
plt.scatter(x=kmeans.cluster_centers_[:, 18], y=kmeans.cluster_centers_[:, 28], marker='x', s=200, linewidths=3, color='black')
plt.xlabel('2019')
plt.ylabel('1991')
plt.title('KMeans Clustering')
plt.show()

print(df.columns)

# Find one country from each cluster
countries = []
for i in [2, 0, 6,3,9,8, 7, 1, 5, 4]:
    cluster_data = df[df['Cluster'] == i]
    if not cluster_data.empty:
        country = cluster_data.sample(1)['Country Code'].values[0]
        countries.append(country)
print(countries)

# Compare the countries from one cluster to find similarities and differences
cluster_data = df[df['Cluster'] == 0]  # replace 0 with the cluster number you want to compare
cluster_data = cluster_data.drop(['Country Code', 'Cluster'], axis=1)
mean_values = cluster_data.mean()
max_values = cluster_data.max()
min_values = cluster_data.min()

print("Countries selected from each cluster:", countries)
print("Mean values for the selected cluster:\n", mean_values)
print("Maximum values for the selected cluster:\n", max_values)
print("Minimum values for the selected cluster:\n", min_values)

cluster_0_data = df[df['Cluster'] == 0]
cluster_1_data = df[df['Cluster'] == 1]
cluster_2_data = df[df['Cluster'] == 2]
cluster_3_data = df[df['Cluster'] == 3]
cluster_4_data = df[df['Cluster'] == 4]
cluster_5_data = df[df['Cluster'] == 5]
cluster_6_data = df[df['Cluster'] == 6]
cluster_7_data = df[df['Cluster'] == 7]
cluster_8_data = df[df['Cluster'] == 8]
cluster_9_data = df[df['Cluster'] == 9]
# Extract the country codes for the data points in cluster 8
country_codes_0 = cluster_0_data['Country Code'].tolist()
country_codes_1 = cluster_1_data['Country Code'].tolist()
country_codes_2 = cluster_2_data['Country Code'].tolist()
country_codes_3 = cluster_3_data['Country Code'].tolist()
country_codes_4 = cluster_4_data['Country Code'].tolist()
country_codes_5 = cluster_5_data['Country Code'].tolist()
country_codes_6 = cluster_6_data['Country Code'].tolist()
country_codes_7 = cluster_7_data['Country Code'].tolist()
country_codes_8 = cluster_8_data['Country Code'].tolist()
country_codes_9 = cluster_9_data['Country Code'].tolist()


# Print the list of country codes
print("Country codes in cluster 0:", country_codes_0)
print("Country codes in cluster 1:", country_codes_1)
print("Country codes in cluster 2:", country_codes_2)
print("Country codes in cluster 3:", country_codes_3)
print("Country codes in cluster 4:", country_codes_4)
print("Country codes in cluster 5:", country_codes_5)
print("Country codes in cluster 6:", country_codes_6)
print("Country codes in cluster 7:", country_codes_7)
print("Country codes in cluster 8:", country_codes_8)
print("Country codes in cluster 9:", country_codes_9)

def err_ranges(x, func, param, sigma):
 
    # Calculate the function values for all combinations of +/- sigma
    values = []
    for s in [-sigma, sigma]:
        for p in np.meshgrid(*[[0, s]] * len(param)):
            values.append(func(*p))
    
    # Determine the minimum and maximum values
    min_val = np.min(values)
    max_val = np.max(values)
    
    # Return the upper and lower limits
    return min_val, max_val

sns.scatterplot(data=data, x='1991', y='2019', hue='Cluster')
plt.scatter(x=kmeans.cluster_centers_[:, 18], y=kmeans.cluster_centers_[:, 28], marker='x', s=200, linewidths=3, color='black')
plt.xlabel('1991')
plt.ylabel('2019')
plt.title('KMeans Clustering')

# Calculate the error ranges for the cluster centers
for i, center in enumerate(kmeans.cluster_centers_):
    print(f"Cluster {i} center: {center}")
    param = ['x', 'y']
    sigma = 1
    lower, upper = err_ranges(center, func=lambda x, y: x + y, param=param, sigma=sigma)
    print(f"Error range for Cluster {i} center: [{lower}, {upper}]")

plt.show()


# Load the dataset for Curve Fit
df = pd.read_csv('API_EN.ATM.CO2E.PC_DS2_en_csv_v2_5358914.csv')


# Select the columns to be used for modeling
cols = [ '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']

# Impute NaN values with the mean of the corresponding column
df[cols] = df[cols].fillna(df[cols].mean())

# Define a simple model function
def model_func(x, a, b, c):
    return a * np.exp(-b * x) + c

# Fit the model to the data
x_data = np.array(df['2019'])
y_data = np.array(df['1991'])
popt, pcov = curve_fit(model_func, x_data, y_data)

# Make predictions for future years
x_pred = np.arange(2022, 2031)
y_pred = model_func(x_pred, *popt)

# Plot the data and the model predictions
plt.scatter(x_data, y_data, label='Data')
plt.plot(x_pred, y_pred, label='Model')
plt.xlabel('2019')
plt.ylabel('1991')
plt.title('Simple Model Fitting')
plt.legend()
plt.show()