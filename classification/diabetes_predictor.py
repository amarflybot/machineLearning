import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import Imputer


def main():
    # Load and review data
    df = pd.read_csv("pima-indians-diabetes.csv", header=None,
                     names=['num_preg', 'glucose_conc', 'diastolic_bp', 'thicknes', 'insulin', 'bmi', 'diab_pred',
                            'age', 'diabetes'])
    print("\n", df.head(5))

    # Check for null values
    print("Are there any null values? ", df.isnull().values.any())

    # Visualization of correlation
    # plot_correlation(df)

    # Define correlation
    print("\nCorrelation\n", df.corr())

    # Check for true/false ratio
    num_true = len(df.loc[df['diabetes'] == True])
    num_false = len(df.loc[df['diabetes'] == False])
    print("\nNumber of True cases: {0} ({1:2.2f} %)".format(num_true, (num_true / (num_true + num_false)) * 100))
    print("Number of False cases: {0} ({1:2.2f} %)".format(num_false, (num_false / (num_true + num_false)) * 100))

    # Splitting the data 70% traning 30% test
    feature_col_names = ['num_preg', 'glucose_conc', 'diastolic_bp', 'thicknes', 'insulin', 'bmi', 'diab_pred',
                         'age']
    predicted_class_names = ['diabetes']
    x = df[feature_col_names].values  # predictor feature columns (8 X m)
    y = df[predicted_class_names].values  # predictor feature columns (1 X m)
    split_test_size = 0.30  # test_Size = 30%

    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=split_test_size, random_state=42)
    #   random_state -> Sets the seed that is used as a part of the splitting process.

    # Check the split
    print("\n{0:0.2f} % in training set".format((len(X_train) / len(df.index)) * 100))
    print("{0:0.2f} % in test set".format((len(X_test) / len(df.index)) * 100))

    # Verifying predicted value was split correctly
    print("\nNumber of Original True cases: {0} ({1:0.2f} %)".format(len(df.loc[df['diabetes'] == True]), (
    len(df.loc[df['diabetes'] == True]) / (len(df.index))) * 100))
    print("Number of Original False cases: {0} ({1:0.2f} %)".format(len(df.loc[df['diabetes'] == False]), (
    len(df.loc[df['diabetes'] == False]) / (len(df.index))) * 100))

    print("\nNumber of Training True cases: {0} ({1:0.2f} %)".format(len(Y_train[Y_train[:] == True]), (
    len(Y_train[Y_train[:] == True]) / (len(Y_train))) * 100))
    print("Number of Training False cases: {0} ({1:0.2f} %)".format(len(Y_train[Y_train[:] == False]), (
    len(Y_train[Y_train[:] == False]) / (len(Y_train))) * 100))

    print("\nNumber of Test True cases: {0} ({1:0.2f} %)".format(len(Y_test[Y_test[:] == True]), (
    len(Y_test[Y_test[:] == True]) / (len(Y_test))) * 100))
    print("Number of Test False cases: {0} ({1:0.2f} %)".format(len(Y_test[Y_test[:] == False]), (
    len(Y_test[Y_test[:] == False]) / (len(Y_test))) * 100))

    # Post-split data Preparation
    print("\n# rows in dataFrame {0}".format(len(df)))
    print("# rows missing glucose_conc: {0}".format(len(df.loc[df['glucose_conc'] == 0])))
    print("# rows missing diastolic_bp: {0}".format(len(df.loc[df['diastolic_bp'] == 0])))
    print("# rows missing thicknes: {0}".format(len(df.loc[df['thicknes'] == 0])))
    print("# rows missing insulin: {0}".format(len(df.loc[df['insulin'] == 0])))
    print("# rows missing bmi: {0}".format(len(df.loc[df['bmi'] == 0])))
    print("# rows missing diab_pred: {0}".format(len(df.loc[df['diab_pred'] == 0])))
    print("# rows missing age: {0}".format(len(df.loc[df['age'] == 0])))

    # impute with the mean
    fill_0 = Imputer(missing_values=0, strategy="mean", axis=0)
    X_train = fill_0.fit_transform(X_train)
    X_test = fill_0.fit_transform(X_test)

    # Training Initial Algorithm Naive Bayes
    nb_model = GaussianNB()
    nb_model.fit(X_train, Y_train.ravel())

    # Performance on Training data
    # Predict values using the training data
    nb_predict_train = nb_model.predict(X_train)

    # Accuracy on training data
    print("\nAccuracy: {0:0.4f}".format(metrics.accuracy_score(Y_train, nb_predict_train)))

    # Performance on Testing data
    # Predict values using the testing data
    nb_predict_test = nb_model.predict(X_test)

    # Accuracy on Testing data
    print("Accuracy: {0:0.4f}".format(metrics.accuracy_score(Y_test, nb_predict_test)))

    # Metrics
    print("\nConfusion Matrix with Naive Bayes algorithm")
    # Note the use of labels for set 1=True to upper left and 0=False to lower right
    print("{0}".format(metrics.confusion_matrix(Y_test, nb_predict_test, labels=[1, 0])))
    # Confusion Matrix
    #     [[ 52  28]
    #      [ 33 118]]

    print("\nClassification Matrix with Naive Bayes algorithm")
    # Note the use of labels for set 1=True to upper left and 0=False to lower right
    # [[ 52  28]  [[TP FP]   Perfect     [[80  0]
    # [ 33 118]]  [FN TN]]  Classifier   [0 151]]
    #
    # Classification Matrix
    # precision -> How often the patients actually had diabetics when the model said they would.(TP/(TP+FP))
    #           Precession shall be more.
    # recall  -> How well the model is predicting diabetes when the result is actually diabetes (TP/(TP+FN))
    #           Recall shall be greater than or equal to 70%
    print("{0}".format(metrics.classification_report(Y_test, nb_predict_test, labels=[1, 0])))
    #   Classification Matrix
    #                precision    recall  f1-score   support
    #
    #           1       0.61      0.65      0.63        80
    #           0       0.81      0.78      0.79       151
    #
    # avg / total       0.74      0.74      0.74       231

    # Random Forest
    lr_model= RandomForestClassifier(random_state=42) # Create random forest object
    lr_model.fit(X_train, Y_train.ravel())

    # Performance on Training data
    # Predict values using the training data
    lr_predict_train = lr_model.predict(X_train)

    # Accuracy on training data
    print("\nAccuracy: {0:0.4f}".format(metrics.accuracy_score(Y_train, lr_predict_train)))

    # Performance on Testing data
    # Predict values using the testing data
    lr_predict_test = lr_model.predict(X_test)

    # Accuracy on Testing data
    print("Accuracy: {0:0.4f}".format(metrics.accuracy_score(Y_test, lr_predict_test)))

    # Metrics
    print("\nConfusion Matrix with Random Forest algorithm")
    # Note the use of lebels for set 1=True to upper left and 0=False to lower right
    print("{0}".format(metrics.confusion_matrix(Y_test, lr_predict_test, labels=[1, 0])))

    print("\nClassification Matrix with Random Forest algorithm")
    print("{0}".format(metrics.classification_report(Y_test, lr_predict_test, labels=[1, 0])))
    # This is a classic example for Over fitting

    # Logistic Regression
    lr_model= LogisticRegression(C=0.7 , random_state=42) # Create random forest object
    lr_model.fit(X_train, Y_train.ravel())

    # Performance on Training data
    # Predict values using the training data
    lr_predict_train = lr_model.predict(X_train)

    # Accuracy on training data
    print("\nAccuracy: {0:0.4f}".format(metrics.accuracy_score(Y_train, lr_predict_train)))

    # Performance on Testing data
    # Predict values using the testing data
    lr_predict_test = lr_model.predict(X_test)

    # Accuracy on Testing data
    print("Accuracy: {0:0.4f}".format(metrics.accuracy_score(Y_test, lr_predict_test)))

    # Metrics
    print("\nConfusion Matrix with Logistic Regression algorithm")
    # Note the use of labels for set 1=True to upper left and 0=False to lower right
    print("{0}".format(metrics.confusion_matrix(Y_test, lr_predict_test, labels=[1, 0])))

    print("\nClassification Matrix with Logistic Regression algorithm")
    print("{0}".format(metrics.classification_report(Y_test, lr_predict_test, labels=[1, 0])))

    # Automating Recall value testing
    C_start = 0.1
    C_end = 5
    C_inc = 0.1
    C_values, recall_scores = [],[]
    C_val = C_start
    best_recall_score = 0
    while C_val < C_end:
        C_values.append(C_val)
        lr_model_loop = LogisticRegression(C=C_val, random_state=42)
        lr_model_loop.fit(X_train, Y_train.ravel())
        lr_predict_loop_test = lr_model_loop.predict(X_test)
        recall_score = metrics.recall_score(Y_test, lr_predict_loop_test)
        recall_scores.append(recall_score)
        if recall_score > best_recall_score:
            best_recall_score = recall_score
            best_lr_predict_test = lr_predict_loop_test

        C_val += C_inc

    best_recall_C_val = C_values[recall_scores.index(best_recall_score)]
    print("1st max value of {0:0.3f} occured at C={1:0.3f}".format(best_recall_score, best_recall_C_val))

    plt.plot(C_values, recall_scores, "-")
    plt.xlabel("C Values")
    plt.ylabel("Recall values")
    # plt.show()

    # Logistic regression with class_weight='balanced'
    # Automating Recall value testing
    C_start = 0.1
    C_end = 5
    C_inc = 0.1
    C_values, recall_scores = [],[]
    C_val = C_start
    best_recall_score = 0
    while C_val < C_end:
        C_values.append(C_val)
        lr_model_loop = LogisticRegression(C=C_val, class_weight="balanced", random_state=42)
        lr_model_loop.fit(X_train, Y_train.ravel())
        lr_predict_loop_test = lr_model_loop.predict(X_test)
        recall_score = metrics.recall_score(Y_test, lr_predict_loop_test)
        recall_scores.append(recall_score)
        if recall_score > best_recall_score:
            best_recall_score = recall_score
            best_lr_predict_test = lr_predict_loop_test

        C_val += C_inc

    best_recall_C_val = C_values[recall_scores.index(best_recall_score)]
    print("1st max value of {0:0.3f} occured at C={1:0.3f}".format(best_recall_score, best_recall_C_val))

    plt.plot(C_values, recall_scores, "-")
    plt.xlabel("C Values")
    plt.ylabel("Recall values")
    plt.show()

    # Again try the test on Test Data Set
    lr_model= LogisticRegression(C=0.3,class_weight='balanced' , random_state=42) # Create random forest object
    lr_model.fit(X_train, Y_train.ravel())

    # Performance on Training data
    # Predict values using the training data
    lr_predict_train = lr_model.predict(X_train)

    # Accuracy on training data
    print("\nAccuracy with class_weight = balanced: {0:0.4f}".format(metrics.accuracy_score(Y_train, lr_predict_train)))

    # Performance on Testing data
    # Predict values using the testing data
    lr_predict_test = lr_model.predict(X_test)

    # Accuracy on Testing data
    print("Accuracy with class_weight = balanced: {0:0.4f}".format(metrics.accuracy_score(Y_test, lr_predict_test)))

    # Metrics
    print("\nConfusion Matrix with Logistic Regression algorithm with class_weight = balanced:")
    # Note the use of labels for set 1=True to upper left and 0=False to lower right
    print("{0}".format(metrics.confusion_matrix(Y_test, lr_predict_test, labels=[1, 0])))

    print("\nClassification Matrix with Logistic Regression algorithm with class_weight = balanced:")
    print("{0}".format(metrics.classification_report(Y_test, lr_predict_test, labels=[1, 0])))

# Function plots a graphical correlation matrix for each pair of columns in th dataframe.
# Input :
#       df: pandas DataFrame
#       size: Vertical and horizontal size of the plot
# Displays :
#       matrix of correlation between cloumns. Blue-cyan-yellow-red-darkred ==> less to more correlated
#                                               0 -------------------------> 1
#                                               Expect a darkred line running from top left to bottom right
def plot_correlation(df, size=11):
    corr = df.corr()  # data frame correlation functions
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)  # color code the rectangles by correlation value
    plt.xticks(range(len(corr.columns)), corr.columns)  # draw x tick marks
    plt.yticks(range(len(corr.columns)), corr.columns)  # draw y tick marks
    plt.show()


if __name__ == '__main__':
    main()
