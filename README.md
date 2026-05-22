# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Scaling for the feature in the data set.

STEP 4:Apply Feature Selection for the feature in the data set.

STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:

```
# FEATURE SCALING

import pandas as pd
from scipy import stats
import numpy as np

df=pd.read_csv("bmi.csv")
df.head()
```


<img width="363" height="258" alt="image" src="https://github.com/user-attachments/assets/506c3e94-596d-4bc5-bcf9-9edb37717498" />



```
df_null_sum=df.isnull().sum()
df_null_sum
```


<img width="158" height="275" alt="image" src="https://github.com/user-attachments/assets/8a292a15-68e3-4b1e-872a-6205e0d43f39" />

```
df.dropna()
```


<img width="393" height="532" alt="image" src="https://github.com/user-attachments/assets/e292aaf4-f362-4d56-b700-8147c4761b4c" />

```
max_vals = np.max(np.abs(df[['Height', 'Weight']]), axis=0)
max_vals
```


<img width="176" height="180" alt="image" src="https://github.com/user-attachments/assets/997119a8-44f7-4fdc-94b9-1b7e93e14573" />

```
# Standard Scaling

from sklearn.preprocessing import StandardScaler
df1=pd.read_csv("bmi.csv")
df1.head()
```


<img width="377" height="257" alt="image" src="https://github.com/user-attachments/assets/b320e93b-a6d3-40ee-aaf3-677f031dc698" />

```
sc=StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df1.head(10)
```


<img width="392" height="462" alt="image" src="https://github.com/user-attachments/assets/9c9168b1-3415-4348-967c-6bde5d97fab7" />

```
#MIN-MAX SCALING

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)

```

<img width="397" height="462" alt="image" src="https://github.com/user-attachments/assets/70e0a0ec-48f5-411b-98cc-e513b195e39b" />

```
#MAXIMUM ABSOLUTE SCALING:

from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
df3=pd.read_csv("bmi.csv")
df3.head()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```


<img width="402" height="492" alt="image" src="https://github.com/user-attachments/assets/c65aa643-4269-4d61-b5b5-3f945d141342" />

```
#ROBUST SCALING

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3.head()

```


<img width="422" height="282" alt="image" src="https://github.com/user-attachments/assets/cc7f17fe-b751-4e84-8fbe-6603a84362ef" />

```
#FEATURE SELECTION:

df=pd.read_csv("income.csv")
df.info()

```



<img width="516" height="477" alt="image" src="https://github.com/user-attachments/assets/6f40eea8-777d-49f8-b9f6-6bb8800c6238" />

```
df_null_sum=df.isnull().sum()
df_null_sum
```


<img width="277" height="652" alt="image" src="https://github.com/user-attachments/assets/8f4f8388-db26-41e0-a96d-49c6a2343201" />

```
# Chi_Square
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]

```

<img width="1107" height="558" alt="image" src="https://github.com/user-attachments/assets/b45b9a32-89bf-47c7-a42d-b17dcd12770f" />

```
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```

<img width="972" height="552" alt="image" src="https://github.com/user-attachments/assets/4486ebf3-573f-495a-a0c5-9b8d6a95864d" />

```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```

<img width="430" height="95" alt="image" src="https://github.com/user-attachments/assets/206eec3c-03d2-436c-b534-88fe91c041ff" />

```
y_pred = rf.predict(X_test)
df=pd.read_csv("income.csv")
df.info()
```


<img width="455" height="465" alt="image" src="https://github.com/user-attachments/assets/d5a4b74d-d3dc-4611-8df8-bc8dcef638ee" />

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```


<img width="1067" height="532" alt="image" src="https://github.com/user-attachments/assets/2799ebe4-830f-4abd-89be-308a7eb79c0b" />

```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```

<img width="953" height="521" alt="image" src="https://github.com/user-attachments/assets/b888e963-3c40-4ed0-9a05-34d44b3eba80" />

```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
k_chi2 = 6
selector_chi2 = SelectKBest(score_func=chi2, k=k_chi2)
X_chi2 = selector_chi2.fit_transform(X, y)
selected_features_chi2 = X.columns[selector_chi2.get_support()]
print("Selected features using chi-square test:")
print(selected_features_chi2)
```

<img width="772" height="102" alt="image" src="https://github.com/user-attachments/assets/1b4dcfac-2a28-4203-ba83-1d241f30fc28" />

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import train_test_split # Importing the missing function
from sklearn.ensemble import RandomForestClassifier

selected_features = ['age', 'maritalstatus', 'relationship', 'capitalgain', 'capitalloss',
'hoursperweek']
X = df[selected_features]
y = df['SalStat']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```


<img width="431" height="87" alt="image" src="https://github.com/user-attachments/assets/6055adfd-aeb5-48d3-82f9-b167553b51ae" />

```
y_pred = rf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using selected features: {accuracy}")
```


<img width="691" height="37" alt="image" src="https://github.com/user-attachments/assets/d6e7191f-1a0c-49fa-bb0b-41a3c7e58945" />

```
!pip install skfeature-chappers
```

<img width="1527" height="376" alt="image" src="https://github.com/user-attachments/assets/4f501dbe-c992-47af-843d-2f19454f5299" />

```
import numpy as np
import pandas as pd
from skfeature.function.similarity_based import fisher_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
categorical_columns = [
    'JobType',
    'EdType',
    'maritalstatus',
    'occupation',
    'relationship',
    'race',
    'gender',
    'nativecountry'
]

df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
# @title
df[categorical_columns]
```

<img width="967" height="532" alt="image" src="https://github.com/user-attachments/assets/83278b61-5132-4860-bdcc-e4bab4190dc2" />

```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
k_anova = 5
selector_anova = SelectKBest(score_func=f_classif,k=k_anova)
X_anova = selector_anova.fit_transform(X, y)
selected_features_anova = X.columns[selector_anova.get_support()]
print("\nSelected features using ANOVA:")
print(selected_features_anova)
```


<img width="882" height="95" alt="image" src="https://github.com/user-attachments/assets/099587c5-80a7-4c37-8381-0a12d96e01dd" />

```
# Wrapper Method
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
df=pd.read_csv("income.csv")
categorical_columns = [
    'JobType',
    'EdType',
    'maritalstatus',
    'occupation',
    'relationship',
    'race',
    'gender',
    'nativecountry'
]

# Convert the categorical columns to category dtype
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```

<img width="947" height="543" alt="image" src="https://github.com/user-attachments/assets/c20776bb-7817-4ba4-80f8-f4bb2caae0b9" />

```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
logreg = LogisticRegression()
n_features_to_select =6
rfe = RFE(estimator=logreg, n_features_to_select=n_features_to_select)
rfe.fit(X, y)
```

<img width="330" height="196" alt="image" src="https://github.com/user-attachments/assets/4817b2d4-989a-42fc-a7c1-283add5be2ae" />

# RESULT:
Thus the program to read the given data and perform Feature Scaling and Feature Selection process and save the data to a file is been executed.
