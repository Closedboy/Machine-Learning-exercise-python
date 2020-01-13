# <div align="center"> Machine Learning </div>

## Programming Exercise 2: Logistic Regression

### Requirement
 - numpy
 - matplotlib
 - scipy

### Logistic Regression

```
python ex2.py
```

#### Problem description
 - In this part of the exercise, you will build a logistic regression model to predict whether a student gets admitted into a university. 
 - Suppose that you are the administrator of a university department and you want to determine each applicant’s chance of admission based on their results on two exams. You have historical data from previous applicants that you can use as a training set for logistic regression. For each training example, you have the applicant’s scores on two exams and the admissions decision.

#### [Data](ex2data1.txt)

 - First column of x refers to the score of exam 1
 - Second column of x refers to the score of exam 2
 - y refers to the admission decision [0, 1]

### Regularized logistic regression

```
python ex2_reg.py
```

#### Problem description

 - In this part of the exercise, you will implement regularized logistic regression to predict whether microchips from a fabrication plant passes quality assurance (QA). During QA, each microchip goes through various tests to ensure it is functioning correctly.
 - Suppose you are the product manager of the factory and you have the test results for some microchips on two different tests. From these two tests, you would like to determine whether the microchips should be accepted or rejected. To help you make the decision, you have a dataset of test results on past microchips, from which you can build a logistic regression model.

#### [Data](ex2data2.txt)

 - First column of x refers to Microchip Test 1
 - Second column of x refers to Microchip Test 2
 - y refers to whether pass the quality assurance or not [0, 1]