# Churn-Analysis
The StayWithMe (SWM) Bank in concerned about the decrease in its customers since some of them are leaving their credit card service.Our main required task is to predict who is going to leave, an important information that could give the chance to offer better service and conditions to customers that are thinking to leave the bank. 

In order to proceed with our analysis, we generated a training set and a test set from the original data. Right after, we created and implemented different models to perform predictions, compare the models between themselves, and to evaluate our data. Our ultimate step was to discuss the obtained results from the analysis and the future implementation of the choosen model.

METHODS 
In order to implement and improve our model, we started from a meticulous observation of the given dataset. Thus, we proceeded with scanning  the information available and studying our variables.
As already mentioned in the introduction, the given dataset consists of 10127 observations of 17 variables. 
In particular the given variables includes:
* Basic info:
  * CLIENTNUM : Unique identifier for the customer holding the account.
* Target:
   * Attrition_Flag: Specifies whether the account was closed (Attrited Customer).
* Demographic Variables:
   * Customer_Age: Demographic variable - Customer's Age in Years.
   * Gender: Demographic variable - M=Male, F=Female.
   * Dependent_count: Demographic variable - Number of dependents.
   * Education_Level: Demographic variable - Educational Qualification of the account holder (example: high school, college graduate, etc.).
   * Marital_Status: Demographic variable - Married, Single, Divorced, Unknown. 
   * Income_Category: Demographic variable - Annual Income Category of the account holder (< 40K, 40K - 60K, 60K−80K, … ).
* Variables (Product):
   * Card_Category: Product Variable - Type of Card (Blue, Silver, Gold, Platinum).
   * Months_on_book: Period of relationship with bank.
   * Total_Relationship_Count: Total no. of products held by the customer
   * Months_Inactive_12_mon: No. of Months in the last 12 months.
   * Contacts_Count_12_mon: No. of Contacts in the last 12 months.
   * Credit_Limit: Credit Limit on the Credit Card.
   * Total_Trans_Amt: Total Transaction Amount (Last 12 months).
   * Total_Trans_Ct: Total Transaction Count (Last 12 months).
   * Avg_Utilization_Ratio: Average Card Utilization Ratio.

The next step was to clean our dataset, with a particular focus on the "Unknown" values and on their potential replacement or, considering a most extreme scenario, removal. 
Thanks to a data visualization approach, we then proceeded on with an analysis of our variables, also looking at their general behaviour and focusing on interactions between them. This approach included, as well, not only the development of two types of graphs (bar graphs and pie charts), but also the analysis of the correlation matrix. This latter step was crucial for the identification of potential elevated correlations between variables. 
The just described processes, led us to the delicate phase of selecting the best model to implement: it is foundamental in order to obtain the most accurate prevision. As for the output, we choose the attrition flag column (with 1 for existing customer and 0 for attrited customer). Furthermore the features were scaled and an encoding was applied to categorical variables. Plus, we applied a split between train and test with the 75% for the train and the 25% for the test. Then, we implemented four different models:
* Logistic Regression,
* Random Forest,
* K Nearest Neighbor,
* Support Vector Machine. 
The results obtained were then analyzed, with an examination of the score of the accuracy, precision, recall, f1, fbeta, and test roc curve. 
As a consequence of the last steps, we selected the Random Forest as the best model. After obtaining this information, we proceeded with the model evaluation analysis of the selected model: the latter analysis consisted of a cross validation and implementation of a confusion matrix.



CODE DESCRIPTION 
Libraries
Here follow the libraries used in order to solve the task requested:
1. Numpy: it is an important package that returns a multidimensional array object, various derived objects (e.g., matrices), and an assortment of routines for fast operations on arrays. [https://numpy.org/doc/stable/user/whatisnumpy.html ]
2. Matplotlib: is a comprehensive library for creating static, animated, and interactive visualizations in Python  [https://matplotlib.org ]
3. Scikit Learn: is a library that provides a selection of efficient tools for machine learning and statistical modeling, including classification, regression, clustering, and dimensionality reduction [https://www.tutorialspoint.com/scikit_learn/scikit_learn_introduction.htm ]
4. Pandas: a library that provides high-performance, easy-to-use data structures, and data analysis tools [https://pandas.pydata.org/docs/ ]
5. Seaborn: is a data visualization library based on matplotlib that provides high-level interface for drawing attractive and informative statistical graphs [https://seaborn.pydata.org ]
6. Plotly: is an interactive, open-source plotting library that supports over 40 unique chart types. [https://plotly.com/python/getting-started/ ].

Data Collection
For this step, we imported the dataset and analyzed the information related to it. Among these, we were able to identify the presence of integers, float, categorical variables (objects), and the absence of null variables. However some "Unknown" values came to light.
Data Cleaning 
We started our data preparation with the cleaning process. The first step was converting all the "Unknown" values into “Nan” type.
Then we defined some functions (“ that could give us relevant and precise quantitative information. after using them, we proceed to delete the rows that presented more than three Nan values; to do so, we defined another function.
For our next step, we draw our attention to the “Income Category” column. The latter, presented a range of values, concerning the income, that were not perfectly suitable for our model; thus we started right from those ranges to create four income values to assign to each client, based on its belonging range. We grouped the values for their education level and calculated the mean and the median of every one of them (thanks to a for cycle); as a result we noticed that the mean and (also the median too) registered values near the income range between 40 thousand, and 80 thousand (later converted to 60 thousand). Right after we replaced each of the missing values within the income range between 40 and 80 thousand, thus 60 thousand. Then, we counted “Nan” values one more time.
We managed to scale down the number “Nan” values, and thus we chose to delete all the rows containing Nan values greater or equal to one, thanks to the functions previously described. The sample of observation on which we decided to use as the base of our analysis, is of 7973 clients.
 
Data visualization 
The best way to avoid customer churn is to know your customers, and the best way to know your customer is through customer data.
As a further step, we wanted to inspect some characteristics regarding the bank’s customers, in order to detect a possible prototype of the average client. We found out that the majority of them are women in an age range between 41 and 53 years old. It is then inevitable to discover that 3999 individuals are married. Taking into account the education level, most clients graduated or attended high school, while almost 1400 of them were uneducated. Moreover, we have two major groups as for the income level: nearly 2800 customers earned annually less than $40 000 and just few more than 3400 a range of $40 000 and $80 000; on the other hand, only 572 individuals overtook $120 000 per year. We then moved on with the study of other two aspects. For what concerns the card category, we realized that the vast majority of the bank’s customers owned a blue card (7441), compared to very few ones that possessed a platinum card (16). Finally, inspecting the attrition flag variable, the clients were broken down in two categories: 84.2% of them is an existing customer, while 15.8% represents the ones who left the bank.
 
Categorical variables convertion 
Before proceeding with our analysis, we wanted to consider also the remaining categorical variables, applying to them a label encoding. The latter converts the data in machine-readable form, and it assigns a unique number (starting from zero) to each class of data.
As we can see also from the correlation matrix, the "Attrition flag" variable doesn't have an high correlation with the others: we can, therefore, maintain all the variables. In the next section we will be able to select the model that  will be used for our analysis.

 
Data Partitoning 
We decided to operate a train test split, imported from sklearn, with a split of  75% for training set and 25% for test set.

Feature Scaling
Once our features were numerical, we opted for feature standardization , that makes the values of each feature in the data have zero-mean (when subtracting the mean in the numerator) and unit-variance.

Model Selection
For each algorithms, we analyzed the obtained results through some scores, importable from sklearn. In particular we used:
* Accuracy score: is the most intuitive performance measure and it is simply a ratio of correctly predicted observation to the total observations;
* Precision score: is defined as the relationship between true positive and the sum of true and false positive
* Recall score: it measures the sensitivity of the model; it can be calculated through the relationship between the corrected prediction for a class, on the total of cases in which the prediction actually happens;
* F score: F1 score is an harmonic mean, thus the reciprocal of the arithmetic mean of the reciprocals. In particular, this score is the harmonic mean of precision and recall. Contrary to a conventional mean, the harmonic one gives a higher weight to smaller values;
* F beta score: is the weighted harmonic mean of precision and recall, reaching its optimal value at 1 and its worst value at 0;
* Test Roc Auc Curve: is the measure of the ability of a classifier to distinguish between classes and is used as a summary of the ROC curve. The higher the AUC, the better the performance of the model at distinguishing between the positive and negative classes.
The algoritm where: 
* Logistic Regression: the main goal of the model is to settle the probability with which an observation can generate one or the other value of the dependent variable; furthermore, it can be used to classify the observations, based on the features of them, into two main categories, just like in our case.
* Random Forest: it is a set classifier obtained by the aggregation among bagging of decision trees; its main goal is to reduce the overfitting occurring in decision tree. Before actualize it, we had to observe which was the number of best estimators for the mode, and in order to do so, we built a loop with the number of estimators = each; the ratio was to build a score graph and to see what was the corresponding point to the best score.
* K Nearest Neighbors: it is an algorithm used for the identification of patterns for the classification of objects. It is based on features of objects that are near the one considered. For this algorithm, we operated as we did for the previous one, looking for the best number of neighbors (analogously to what we did for Random Forest). number of estimators = each; the ratio was to build a score graph and to see what was the corresponding point to the best score.
* Support Vector Machine: it builds an hyperplane or a set of hyperplanes, in a multidimensional, or infinite-dimensional, space. The latter can be used for the classification.
 
As for now, we can assume that the most reliable  model is "Random Forest".

Model Evaluation 
For the Model Evaluation, we decided to proceed with a cross validation and a confusion matrix. Thanks to the first one, we were able to delete the overfitting issue, occuring in the training set; the second one enabled us to analyze the error made by our model.
Cross validation consists in dividing the total data set into k parts of equal numerosity. At each step, the k-th part of the dataset is validated, while the remaining parts always constitutes the training set. Thus, the model is trained for each of the k parts, thus avoiding problems of overfitting, but also of asymmetrical (and thus biased) sampling of the observed sample, which is a typical issue that could arise when splitting the data into only two parts.
 
As we can see, the cross validation score is 95% with a fluctuation of 0.01%.
In the confusion matrix, the resulting table consists of two rows and two columns, filled with four values: true positives, false positives, true negatives and false negatives. In the confusion matrix, there is a true positive where the observation is positive with a positive prediction. There is a false positive where the observation is negative with a positive prediction. There is a true negative where the observation is negative with a negative prediction and a false negative indicates a positive observation with a negative prediction. The assorted equations then show how to calculate accuracy and precision for a given project.
 
The confusion matrix score is about 94%, in line with the other score of the cross validation that we already have obtained.


RESULTS AND CONCLUSIONS 

Results collection
We created a data frame with all the obtained predicted test values and we compared them with the actual test values.
Then, we took the index values of the created matrix and compared them with the original dataset in order to get only the corresponding "client ids", and we added the column to the data frame.
We changed the columns positions and converted the 0s and 1s, respectively, in "Attrited Customer" and "Existing Customer". 
Finally, we created an excel with the obtained results.  
  

Analyzing our results, we noticed that the actual data and the test set data registered a discrepancy of approximately 3%, obtained by implementing the Random Forest model. However, only a small percentage of our dataset is considered to be at churn risk. 

Future Implementations
In order to provide a deeper analysis of the steps done so far, we had to consider the cons of our model implementation. As a matter of fact, the chosen model is not easily interpretable. Random Forest, provides the possibility to detect feature importance, but the same cannot be said when coming to the chance of providing a more complete visibility into the coefficients; also, it can be computationally intensive for large datasets. Random Forest can be considered as a “black box” algorithm, in which the control over the model’s action, is rather limited and scarce. 
We can state that the implementation of our model in the existing process, with the available data, should be easy to achieve, and inserting new data into our database, we could be able to have more accurate predictions. Perhaps, as time goes by, the algorithm may encounter some difficulties in handling and operating with large amount of data; for this reason, we recommend for future implementations a continuous monitoring of the whole process, to see if and how  it changes over time. Another recommendation could be a continuous study of the independent variables (both at the level of values, and at the level of total influence on the analysis), trying to exclude any type of bias, which could mislead the model. In order to face the main issue concerning the limitation of clients that are closing their accounts, we highlighted, as a possible solution, the possibility to fill out a Customer Satisfaction Survey. The latter could be sent to all the customers that can be classified as "risk customers", in order to have a deeper understanding about their opinion on the company and possible complaints. Discounts or other incentives can also be offered to the customers flagged by the algorithm in order to try and retain them.


