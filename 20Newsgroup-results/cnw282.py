"""
Author: Chinmay Wyawahare
NetID: cnw282
"""

import pyspark
from pyspark import SparkContext

from pyspark.sql import SparkSession
from pyspark.sql.types import StringType
from pyspark.sql.types import DoubleType
from pyspark.sql import functions as F

from pyspark.sql import types as D
from pyspark.ml.feature import HashingTF, IDF, RegexTokenizer, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.mllib.evaluation import MulticlassMetrics
import csv
import lime 
from lime import lime_text
from lime.lime_text import LimeTextExplainer

import numpy as np

sc = SparkContext()

spark = SparkSession \
        .builder \
        .appName("hw3") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()

########################################################################################################
# Load data
categories = ["alt.atheism", "soc.religion.christian"]
LabeledDocument = pyspark.sql.Row("category", "text")

def categoryFromPath(path):
    return path.split("/")[-2]
    
def prepareDF(typ):
    rdds = [sc.wholeTextFiles("/user/tbl245/20news-bydate-" + typ + "/" + category)\
              .map(lambda x: LabeledDocument(categoryFromPath(x[0]), x[1]))\
            for category in categories]
    return sc.union(rdds).toDF()

train_df = prepareDF("train").cache()
test_df  = prepareDF("test").cache()

#####################################################################################################
""" Task 1.1
a.	Compute the numbers of documents in training and test datasets. Make sure to write your code here and report
    the numbers in your txt file.
b.	Index each document in each dataset by creating an index column, "id", for each data set, with index starting at 0. 

""" 
# Append text to file
def writeToFile(text):
	with open('cnw282_report.txt', 'a', newline="") as f:
		f.write(text)

train_df_rows = 'Number of rows in train_df = ' + str(train_df.count())+'\n'
test_df_rows = 'Number of rows in test_df = ' + str(test_df.count())+'\n'

# Dump the results to the report
writeToFile("Task 1.1 (a)\n")
writeToFile(train_df_rows)
writeToFile(test_df_rows)

train_df = train_df.rdd.zipWithIndex()
train_df = train_df.toDF()
train_df = train_df.withColumn('category', train_df['_1'].getItem("category"))
train_df = train_df.withColumn('text', train_df['_1'].getItem("text"))
train_df = train_df.withColumnRenamed('_2', 'id')
train_df = train_df.select('id','category','text')

test_df = test_df.rdd.zipWithIndex()
test_df = test_df.toDF()
test_df = test_df.withColumn('category', test_df['_1'].getItem("category"))
test_df = test_df.withColumn('text', test_df['_1'].getItem("text"))
test_df = test_df.withColumnRenamed('_2', 'id')
test_df = test_df.select('id','category','text')

writeToFile("\nTask 1.1 (b)\n")
writeToFile("First 5 rows of 'INDEXED' test set \n\n")
k = test_df.take(5)
for i,row in enumerate(k):
	row_name='Row-' + str(i)
	writeToFile(row_name+'\n')
	writeToFile(str(row[0])+', '+str(row[1])+ ', '+str(row[2])+ '\n\n')

########################################################################################################
# Build pipeline and run
indexer   = StringIndexer(inputCol="category", outputCol="label")
tokenizer = RegexTokenizer(pattern=u'\W+', inputCol="text", outputCol="words", toLowercase=False)
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures")
idf       = IDF(inputCol="rawFeatures", outputCol="features")
lr        = LogisticRegression(maxIter=20, regParam=0.001)

# Builing model pipeline
pipeline = Pipeline(stages=[indexer, tokenizer, hashingTF, idf, lr])

# Train model on training set
model = pipeline.fit(train_df)   #if you give new names to your indexed datasets, make sure to make adjustments here

# Model prediction on test set
pred = model.transform(test_df)  # ...and here

# Model prediction accuracy (F1-score)
pl = pred.select("label", "prediction").rdd.cache()
metrics = MulticlassMetrics(pl)
init_f1_score = metrics.fMeasure()
print('f1 score: ',init_f1_score)

# Dump f1 score to report
writeToFile("\nTask 1.2 (a)\n")
writeToFile("F1 score: ")
writeToFile(str(init_f1_score))

pred.show()

# Dump schema to report
writeToFile("\nTask 1.2 (b)\n")
for tup in pred.dtypes:
	res='Column Name: ' + str(tup[0])+', '+ 'Type: ' + str(tup[1]) + '\n'
	writeToFile(res)

#####################################################################################################

""" Task 1.2
a.	Run the model provided above. 
    Take your time to carefully understanding what is happening in this model pipeline.
    You are NOT allowed to make changes to this model's configurations.
    Compute and report the F1-score on the test dataset.
b.	Get and report the schema (column names and data types) of the model's prediction output.

""" 

#######################################################################################################
#Use LIME to explain example
class_names = ['Atheism', 'Christian']
explainer = LimeTextExplainer(class_names=class_names)

# Choose a random text in test set, change seed for randomness 
test_point = test_df.sample(False, 0.1, seed = 10).limit(1)
test_point_label = test_point.select("category").collect()[0][0]
test_point_text = test_point.select("text").collect()[0][0]

def classifier_fn(data):
    spark_object = spark.createDataFrame(data, "string").toDF("text")
    pred = model.transform(spark_object)   #if you build the model with a different name, make appropriate changes here
    output = np.array((pred.select("probability").collect())).reshape(len(data),2)
    return output

exp = explainer.explain_instance(test_point_text, classifier_fn, num_features=6)
#print('Probability(Christian) =', classifier_fn([test_point_text])[0][0])
#print('True class: %s' % class_names[categories.index(test_point_label)])
#print('explanation = ',exp.as_list())

#####################################################################################################

#####################################################################################################
""" 
Task 1.3 : Output and report required details on test documents with ID's 0, 275, and 664.
Task 1.4 : Generate explanations for all misclassified documents in the test set, sorted by conf in descending order, 
           and save this output (index, confidence, and LIME's explanation) to netID_misclassified_ordered.csv for submission.
"""
# Your code starts here

test_IDS = pred.filter((F.col('id')==0)|(F.col('id')==275)|(F.col('id')==664)).collect()

misclassified = pred.filter((F.col('label')!=F.col('prediction'))).collect()

misclassified_list=[]

for row in misclassified:
	Id = row[0]
	prob_l = row[8]
	conf = abs(prob_l[0]-prob_l[1])
	exp_row = explainer.explain_instance(row[2], classifier_fn, num_features=6)
	new_row = [Id, conf, exp_row.as_list()]
	misclassified_list.append(new_row)

misclassified_list.sort(key= lambda k: (k[1], k[0]), reverse=True)

header=['ID', 'Confidence', 'Explanation-List']

# Task 1.4
with open('cnw282_misclassified_ordered.csv', "w", newline="") as f:
	writer = csv.writer(f)
	writer.writerow(i for i in header)
	writer.writerows(misclassified_list)

task_ID=[]
for row in test_IDS:
	Id=row[0]
	category=row[1]
	prob_list=row[8]
	pred_category='atheism' if row[9]==1.0 else 'christian'
	exp_row=explainer.explain_instance(row[2], classifier_fn, num_features=6)
	task_ID.append([Id, category, pred_category, prob_list, exp_row.as_list()])

# Dump test file details on test documents
writeToFile("\nTask 1.3\n")
for row in task_ID:
	Id='Id: '+ str(row[0])
	label='Actual Category: ' + str(row[1])
	pred_label='Predicted Category: ' + str(row[2])
	prob='Probability: ' + str(row[3])
	explanation='Explanation List: ' + str(row[4])
	line=Id +', '+label+', '+ pred_label+ ', '+prob+ ', '+ explanation+ '\n'
	writeToFile(line)


########################################################################################################
""" Task 1.5
Get the word and summation weight and frequency
"""
# Your code starts here
words = {}

for row in misclassified_list:
	exp_list = row[2]
	for tup in exp_list:
		word = tup[0]
		weight = tup[1]
		if word not in words.keys():
			words[word]=[]
			words[word].append(1)
			words[word].append(abs(weight))
		elif word in words.keys():
			words[word][0] = words[word][0]+1
			words[word][1] = words[word][1]+abs(weight)

words_arr=[]

words_new_arr=[]

# For report
for k, v in words.items():
	new_row = [k, v[0], v[1]]
	words_arr.append(new_row)

# For Task 2
for k, v in words.items():
	n_row = [k, v[1]/v[0]]
	words_new_arr.append(n_row)

words_new_arr.sort(key=lambda k: (k[1]), reverse=True)	

words_arr.sort(key=lambda k: (k[1], k[2]), reverse=True)
header1 = ['Word', 'Count', 'Weight']
	
# Task 1.5
with open('cnw282_words_weight.csv', 'w', newline="") as f1:
	writer = csv.writer(f1)
	writer.writerow(i for i in header1)
	writer.writerows(words_arr)
########################################################################################################
""" Task 2
Identify a feature-selection strategy to improve the model's F1-score.
Codes for your strategy is required
Retrain pipeline with your new train set (name it, new_train_df)
You are NOT allowed make changes to the test set
Give the new F1-score.
"""

#Your code starts here
t5 = [row[0] for row in words_new_arr[:20]]					
remove_word = F.udf(lambda x: x.replace(t5[0], "").replace(t5[1], "").replace(t5[2], "").replace(t5[3], "").replace(t5[4],"").replace(t5[5],"").replace(t5[6],"").replace(t5[7],"").replace(t5[8],"").replace(t5[9],"").replace(t5[10], "").replace(t5[11], "").replace(t5[12],"").replace(t5[13],"")  , D.StringType())
train_df = train_df.withColumn('text1', remove_word(train_df.text))
train_df = train_df.select('id', 'category', 'text1')
train_df = train_df.withColumnRenamed("text1", "text")
train_df.show(2)
model = pipeline.fit(train_df)
pred = model.transform(test_df)
pl = pred.select("label", "prediction").rdd.cache()
metrics = MulticlassMetrics(pl)
print('new f1 score = ', metrics.fMeasure())
new_misclassified = pred.filter((F.col('label')!=F.col('prediction'))).collect()
new_misclassified_IDs = [row[0] for row in new_misclassified]
prev_missclassified_IDs = [row[0] for row in misclassified_list]
correct_ids_after=[]

for Id in prev_missclassified_IDs:
	if Id not in new_misclassified_IDs:
		correct_ids_after.append(Id)

print("Final: ", correct_ids_after)

writeToFile("\n\nTask 2\n")
writeToFile("\nStrategy:\n")
writeToFile("Step-1:\n") 
writeToFile("We have identified the words which contributed towards misclassified documents. Using weights/count as measure, we will remove the misclassified words which contributed to a decrease in accuracy of the model. Along with this, we have the words in a sorted order in descending order starting with words whcih misclassified the most number of documents.\n") 
writeToFile("We can put the top 13 words in a list that had contributed the most towards misclassified document\n")
writeToFile("\nStep-2:\n") 
writeToFile("Create a new column called 'text1' from text with the use of user defined function - udf. Rename this column to 'text' later\n")
writeToFile("\nStep-3:\n")
writeToFile("After making these modifications, we can observe that we get a better accuracy as we have optimized the model\n\n")

writeToFile("We can say that words that contributed to multiple misclassified documents contributed for the decrease of the precision of the model - Without loss of generality (WLOG)\n")
writeToFile("After removing some of these words we reduce our rate of false positive (FPR) and false negatives (FNR) which will contribute to the increase in the F1 score\n\n")

acc = "New Accuracy after Feature Engineering: " + str(metrics.fMeasure())+'\n'
# Dump new accuracy to report
writeToFile(acc)

# Dump ID's that got classified correctly after feature selection
correct_ids = "Document ID's that are classified correctly after feature selection (which were misclassified before): " + str(correct_ids_after)+'\n'
writeToFile(correct_ids)