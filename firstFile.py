#
#  firstFile.py
#  Created by Raghunath Tripasuri on 29/03/25.
#
print("First File");
a=100
print(a);

import numpy as np;
import pandas as pd;

array1d = [1,2,3,4];
array2d = [[1,2,3,4],[5,6,7,8]];
#print(array1d);
#print(array2d);
data = np.random.rand(2,3,4);
print(data);

zeros = np.zeros((2,2,2));
#print(zeros);

full = np.full((2,2,2), 10);
#print(full);

ones = np.ones((2,3,2));
#print(ones);

arr = np.array([[1,4,5],[6,7,8]]);
#print(arr, arr.shape, arr.size, arr.dtype);

#print(data[0], data[0:2], data[-1], data[0][0][2]);

list1 = np.random.rand(10);
list2 = np.random.rand(10);

#print("list 1 is: ",list1, "\n\n list2 is : ",list2);

#print("\n Addition of list1 and list2 is: ", np.add(list1, list2));

#print("\n dot product of list1 and list2 is: ", np.dot(list1, list2));

#print("\n sorted values in data array: ", np.sort(data));

zeros = np.zeros((8));
#print("\n zeros array: ", zeros);

zeros = np.append(zeros, [15,19]);
#print("\n zeros array after append: ", zeros);

zeros = np.insert(zeros, 3, 9);
#print("\n zeros array after insert: ", zeros);

data = np.delete(data, 0, axis=1);
#print("\n data array after delete: ", data);

cars_df = pd.read_csv("/Users/raghunatht/Documents/Programming/Python/Regression_py/Data/Auto.csv");
print(cars_df.columns);
print(cars_df.dtypes);
print(cars_df.describe());
print(cars_df.describe(include='object'));

# get one column
print(cars_df.horsepower);

# get one column with space in the name
print(cars_df['origin country']);

# get multiple columns
print(cars_df[['mpg','horsepower']]);

# get unique values in a column
print(cars_df['year'].unique());

#filter data on rows - get all rows for the year 82
print(cars_df[cars_df['year'] == 82]);

#filter data on rows - get all rows for the year 82 and american cars - origin country = 1
print(cars_df[(cars_df['year'] == 82) & (cars_df['origin country'] == 1)]);

# get data from 22nd row to 39th row
print(cars_df[22:40]);

# get data from 15th row and 3 rd column
print(cars_df.iloc[15,3]);

#find the rows which have any null values
print(cars_df.isnull().sum());

#drop the rows which have any null values
cars_df.dropna(inplace = True);

#find the rows which have any null values
print(cars_df.isnull().sum());

#find the new shape of the data
print(cars_df.shape);

#drop displacement column from the data set
print(cars_df.drop('displacement', axis=1));

#add a new derived column to the data set, which is the sum of weight and acceleration
cars_df['weight_acceleration'] = cars_df['weight'] + cars_df['acceleration'];
print(cars_df.columns);

#set all values of the new column to 100
cars_df['weight_acceleration'] = 100;

print(cars_df.head);

#set first record of new column to 300 and remaining to 100
cars_df.iloc[0,-1] = 300;
cars_df.loc[1,'weight_acceleration'] = 450;
print(cars_df.head);

#apply method usage
cars_df['wt_boolean'] = cars_df['weight'].apply(lambda x: True if x > 3000 else False);
print(cars_df.head);


#map the country names based on origin country data
cars_df['country_name'] = cars_df['origin country'].map({1:'USA', 2:'Europe', 3:'Japan'});
print(cars_df.head);

#cars_df save it to csv format
cars_df.to_csv("/Users/raghunatht/Documents/Programming/Python/Regression_py/Data/Auto_modified.csv");

import matplotlib.pyplot as plt;

#check the relationship between horsepower and mpg
plt.scatter(cars_df['horsepower'], cars_df['mpg']);
plt.xlabel("Horsepower");
plt.ylabel("Miles per Gallon");
plt.savefig("/Users/raghunatht/Documents/Programming/Python/Regression_py/horsepower_mpg.png");
plt.close();

import seaborn as sn;
sn.catplot(data = cars_df, x='cylinders', kind='count');
plt.savefig("/Users/raghunatht/Documents/Programming/Python/Regression_py/cylinders_count.png");
plt.close();

#box plot for cylinders vs mpg
sn.boxplot(data = cars_df, x='cylinders', y='mpg');
plt.savefig("/Users/raghunatht/Documents/Programming/Python/Regression_py/bxPlt_cyl_mpg.png");
plt.close();

import plotly;
import plotly.graph_objects as go;
import plotly.express as px;

df = px.data.iris();
fig = px.scatter(df, x='sepal_width', y='sepal_length', color='species', size='petal_length', hover_data=['petal_width']);
fig.show();

# use plotly to create scatter plot for cars data using horsepower, mpg, cylinders and show names of the cars
fig = px.scatter(cars_df, x='horsepower', y='mpg', color='cylinders', hover_data=['name']);
fig.show();
plotly.offline.plot(fig, filename="/Users/raghunatht/Documents/Programming/Python/Regression_py/cars_scatter_plot.html");
