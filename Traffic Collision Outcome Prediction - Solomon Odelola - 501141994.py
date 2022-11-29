#!/usr/bin/env python
# coding: utf-8

# # **Prepare**

# ## Import

# In[2]:


import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt


import statsmodels.formula.api as smf
from scipy.stats import norm 
from scipy import stats, integrate

from IPython.display import HTML

get_ipython().system('pip install folium')
import folium
from folium.plugins import HeatMap
from math import sqrt


from sklearn import neighbors
from sklearn.preprocessing import *
from sklearn.impute import SimpleImputer

get_ipython().system('pip install xgboost')
from xgboost import XGBClassifier
from sklearn.metrics import *
from sklearn.pipeline import *
from sklearn.utils.validation import check_is_fitted
import seaborn as sns
import matplotlib.pyplot as plt

warnings.simplefilter(action="ignore", category=FutureWarning)


# In[3]:


def wrangle(path):
    # Read csv file into dataframe
    df = pd.read_csv(path)

    # Select the features
    # df = df[[ 'YEAR', 'DATE', 'TIME', 'HOUR', 'STREET1', 'STREET2', 'ROAD_CLASS', 'DISTRICT', 'LOCCOORD', 'ACCLOC', 'TRAFFCTL', 'VISIBILITY', 'LIGHT',
    # 'RDSFCOND', 'ACCLASS', 'IMPACTYPE', 'INVTYPE', 'INVAGE', 'INJURY', 'INITDIR', 'VEHTYPE', 'MANOEUVER', 'DRIVACT', 'DRIVCOND', 'PEDTYPE', 'PEDACT', 
    # 'PEDCOND',  'PEDESTRIAN', 'CYCLIST', 'AUTOMOBILE', 'MOTORCYCLE', 'TRUCK','TRSN_CITY_VEH', 'EMERG_VEH', 'PASSENGER', 'SPEEDING', 'AG_DRIV', 'REDLIGHT',
    # 'ALCOHOL', 'DISABILITY', 'POLICE_DIVISION','NEIGHBOURHOOD']]
    ## Dropping columns where missing values were greater than 80%
    df = df.drop(["PEDTYPE", "PEDACT", "PEDCOND"], axis=1)
    # Changing the property damage and non-fatal columns to Non-Fatal
    df['ACCLASS'] = np.where(df['ACCLASS'] == 'Property Damage Only', 'Non-Fatal', df['ACCLASS'])
    df['ACCLASS'] = np.where(df['ACCLASS'] == 'Non-Fatal Injury', 'Non-Fatal', df['ACCLASS'])
    
    df['MONTH'] = pd.to_datetime(df['DATE']).dt.month
    df['MONTH_NAME'] = pd.to_datetime(df['DATE']).dt.month_name()
    df['DAY'] = pd.to_datetime(df['DATE']).dt.day
    df['MINUTES'] = pd.to_datetime(df['DATE']).dt.minute
    df['WEEKDAY'] = pd.to_datetime(df['DATE']).dt.weekday


    return df    


# In[4]:


path = 'KSI DATA.csv'
df = wrangle(path)
print(df.shape)


# In[5]:


df.head()


# In[6]:


df.describe()


# ## Explore

# ### 1. Changing the property damage and non-fatal columns to Non-Fatal

# In[7]:


df['ACCLASS'] = np.where(df['ACCLASS'] == 'Property Damage Only', 'Non-Fatal', df['ACCLASS'])
df['ACCLASS'] = np.where(df['ACCLASS'] == 'Non-Fatal Injury', 'Non-Fatal', df['ACCLASS'])
df.ACCLASS.unique()


# ### 2. Accident numbers against years and months

# In[8]:


# #Number of Unique accidents by Year
Year_accident = df.groupby('YEAR')['ACCNUM'].nunique().sort_index(ascending=True)
Month_accident = df.groupby('MONTH_NAME')['ACCNUM'].nunique().sort_index(ascending=True)

fig, ax = plt.subplots(2,1,figsize=(19,4))


ax[1].set_title("Accidents caused in different years")
ax[1].set_ylabel('Number of Accidents (ACCNUM)')
ax[1].plot(Year_accident, color='blue')


ax[0].set_title("Accidents caused in different months")
ax[0].set_ylabel('Number of Accidents (ACCNUM)')
ax[0].plot(Month_accident, color='red')

plt.show()


# ### 3. Fatality Heatmap of those that where Fatally Injured

# In[9]:


df_Fatal = df[df['INJURY'] == 'Fatal']
df_Fatal = df_Fatal[['LATITUDE', 'LONGITUDE']]
lat_Toronto_1 = df_Fatal.describe().at['mean','LATITUDE']
lng_Toronto_1 = df_Fatal.describe().at['mean','LONGITUDE']
Toronto_location_F = [lat_Toronto_1, lng_Toronto_1]
Fatal_map_F = folium.Map(Toronto_location_F, zoom_start=10.255)
HeatMap(df_Fatal.values, min_opacity =0.3).add_to(Fatal_map_F)
Fatal_map_F


# ### 4. Fatality Heatmap of those that where not Fatally Injured

# In[10]:


df_Non_Fatal = df[df['INJURY'] != 'Fatal']
df_Non_Fatal = df_Non_Fatal[['LATITUDE', 'LONGITUDE']]
lat_Toronto_2 = df_Non_Fatal.describe().at['mean','LATITUDE']
lng_Toronto_2 = df_Non_Fatal.describe().at['mean','LONGITUDE']
Toronto_location_N = [lat_Toronto_2, lng_Toronto_2]
Fatal_map_N = folium.Map(Toronto_location_F, zoom_start=10.255)
HeatMap(df_Fatal.values, min_opacity =0.3).add_to(Fatal_map_N)
Fatal_map_N


# ### 5. Fatality over years (# of people died)

# In[11]:


#Lets look at Fatality over years (# of people died)
Fatality = df[df['INJURY'] =='Fatal']
Fatality = Fatality.groupby(df['YEAR']).count()
plt.figure(figsize=(12,6))


plt.ylabel('Number of Injury=Fatal')
Fatality['INJURY'].plot(kind='bar',color="y" , edgecolor='black')

plt.show()


# ### 6. Looking at area where accident happens
# 
# 

# In[12]:


Region_df = df['DISTRICT'].value_counts()
plt.figure(figsize=(12,6))
plt.ylabel('Number of Accidents')
Region_df.plot(kind='bar',color=list('rgbkmc') )
plt.show()


# ### 7. Top 10 Neighbourhood with Most collisions

# In[13]:


Hood_df = df['NEIGHBOURHOOD'].value_counts()
plt.figure(figsize=(12,6))
plt.ylabel('Number of Accidents')
Hood_df.nlargest(10).plot(kind='bar',color=list('rgbkmc') )
plt.show()


# 

# In[14]:


def conut_feature(df, xlabel, title):
    ax = sns.countplot(x=df)

    plt.xticks(size=12)
    plt.xlabel(xlabel, size=14)
    plt.yticks(size=12)
    plt.ylabel('Number of Traffic Collisions', size=12)
    plt.title(title, size=16)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

    total = len(df)
    for p in ax.patches:
        percentage = f'{100 * p.get_height() / total:.2f}%\n'
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()
        ax.annotate(percentage, (x, y), ha='center', va='center')
    plt.tight_layout()
    plt.show()


# In[15]:


# What is the percentage of OUTCOME?
conut_feature(df['ACCLASS'], 'Outcome', 'Collision Outcome')


# In[16]:



conut_feature(df['ROAD_CLASS'], 'level of road classes',"Road Collision")


# ### Data cleaning

# In[17]:


df_clean_data = df[['ACCNUM', 'YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTES', 'WEEKDAY', 'LATITUDE', 'LONGITUDE', 
'DISTRICT', 'VISIBILITY', 'LIGHT', 'RDSFCOND', 'PEDESTRIAN', 'CYCLIST', 'AUTOMOBILE', 'MOTORCYCLE', 'TRUCK', 
    'TRSN_CITY_VEH', 'EMERG_VEH', 'PASSENGER', 'SPEEDING', 'AG_DRIV', 'REDLIGHT', 'ALCOHOL', 'DISABILITY', 'ACCLASS']]

# Transforming the coordinates
df_clean_data['LATITUDE'] = df_clean_data['LATITUDE'].astype('int')
df_clean_data['LONGITUDE'] = df_clean_data['LATITUDE'].astype('int')


# In[18]:


print("Percentage of missing values in the KSI_CLEAN_data dataset")
df_clean_data.isna().sum()/len(df_clean_data)*100


# #### Encoding

# In[19]:


df_clean_data = pd.get_dummies(df_clean_data, columns=['VISIBILITY','RDSFCOND','LIGHT','DISTRICT','PEDESTRIAN','CYCLIST', 'AUTOMOBILE', 'MOTORCYCLE', 'TRUCK', 
    'TRSN_CITY_VEH', 'EMERG_VEH', 'PASSENGER', 'SPEEDING', 'AG_DRIV', 'REDLIGHT', 'ALCOHOL', 'DISABILITY'])

df_clean_data.info()


# ### Finding Important features

# In[20]:


X = df_clean_data.drop('ACCLASS', axis=1)
y = df_clean_data['ACCLASS']    #target column i.e price range

from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X,y)
 #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(15).plot(kind='barh')
plt.show()


# In[21]:


feat_importances.sort_values(ascending=False)[:10]


# In[22]:


df_clean_data.info()


# ## Split

# In[23]:


target = 'ACCLASS'
X = df_clean_data.drop(target, axis=1)
y = df_clean_data[target].map({'Fatal':1,'Non-Fatal':0})


# In[24]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


# # **Build Model**

# ## Model 1 - XGBoost Classifier

# In[25]:


model_1 = make_pipeline(
    StandardScaler(),
    SimpleImputer(),
    XGBClassifier(),
)
model_1.fit(X_train, y_train)


# In[26]:


model_1_pred = model_1.predict(X_test)
model_1_acc = accuracy_score(y_test, model_1_pred)
print(f'Model Accuracy: {model_1_acc}')
print(classification_report(y_test, model_1_pred))


# ## Model 2 - Support Vector Classifier

# In[27]:


from sklearn.svm import SVC
model_2 = make_pipeline(
    StandardScaler(),
    SimpleImputer(),
    SVC()
)
model_2.fit(X_train, y_train)


# In[28]:


model_2_pred = model_2.predict(X_test)
model_2_acc = accuracy_score(y_test, model_2_pred)
print(f'Model Accuracy: {model_2_acc}')
print(classification_report(y_test, model_2_pred))


# ## Evaluate the models

# In[29]:


names = ['XGBClassifier', 'Support Vector Classifier']
acc_score = [model_1_acc, model_2_acc]
models = pd.DataFrame()
models['Models'] = names
models['Accuracy'] = acc_score


# In[30]:


models.head()


# In[31]:


sns.barplot(x='Models', y='Accuracy', data=models)

