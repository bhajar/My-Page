Incorrect Cross Val
===================

0. ARRAY_OF_SCORES = []

1. FIT & TRANSFORM (all) DOCUMENTS to Document-Term-Matrix (DTM) using VECTORIZER.
   # Because the DOCUMENTS contain words, and our models 
   # don't understand words.
   # Hence, we'll need to convert the words/documents into
   # numerical vectors (i.e. DTM)

2. For i in range(1,6):  # range(1,6) gives you [1,2,3,4,5]

3.     Train = DTM[ int((i-1)/5*len(DTM)) : int(i/5*len(DTM)) ]
       # For 1st (of 5 folds), take the first 20% of the DTM data
       # For 2nd (of 5 folds), take the second 20% of the DTM data
       # ...
       # For 5th (of 5 folds), take the last 20% of the DTM data

4.     Test = remaining DTM data after we exclude the data from "Train"

5a.    Fit the training (DTM) data into model
5b.    Predict with the testing data (DTM)
5c.    Evaluate with the ground-truth of the testing data.
5d.    Save the score into "ARRAY_OF_SCORES"

    # End of For Loop

6.  score = calculate the mean of ARRAY_OF_SCORES


Correct Cross Val
=================

0. ARRAY_OF_SCORES = []

1. For i in range(1,6):  # range(1,6) gives you [1,2,3,4,5]

2.     Train = DOCUMENT[ int((i-1)/5*len(DOCUMENT)) : int(i/5*len(DOCUMENT)) ]
       # For 1st (of 5 folds), take the first 20% of the DTM data
       # For 2nd (of 5 folds), take the second 20% of the DTM data
       # ...
       # For 5th (of 5 folds), take the last 20% of the DTM data

3.     Test = remaining DOCUMENT data after we exclude the data from "Train"

4.       *Fit* the VECTORIZER on train.
       - *Transform* Train to TrainDTM
       - *Transform* Test to TestDTM

5a.    Fit the training (DTM) data into model
5b.    Predict with the testing data (DTM)
5c.    Evaluate with the ground-truth of the testing data.
5d.    Save the score into "ARRAY_OF_SCORES"

    # End of For Loop

6.  score = calculate the mean of ARRAY_OF_SCORES