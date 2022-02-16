# SVM RF GREOBLE TREES

### Objectives

* Discover SVM and Random Forest algorithms.  
* Predict tree planting year that are missing in the [Grenoble Aglomération Trees dataset](https://data.metropolegrenoble.fr/ckan/dataset/les-arbres-de-grenoble).  
* This project was made in three days at the campus numérique in the Apls.


### Framework

* Python and scikit learn library  
* Pickle file system

## What can you find in this repository ?

* The `Data/` folder is missing for saving space purpose

After pull, create a folder in the repo name `Data`, and put the .csv file into.  
The .csv file can be found. [here](https://data.metropolegrenoble.fr/ckan/dataset/les-arbres-de-grenoble) and need to be rename `ARBRE.csv`.
  
* preprocessing.py

This file contains the *processing* of the csv file (delete useless columns, separate data with / without planting year).

* pipelines.py

This file contains the *processing* of the data (modifying categories into sparse matrices, ordinal features into numerical, ...)

* SVM.py

Selection and training of Support Vector Machine algorithms.

* randomforest.py

Selection and training of Random Forest algorithms.

* linear_ElasticNet.py

Training of a naive linear model and selection and training of an Elastic Net model.

* comparison_between_models.py

This last file is for comparison between the four models, retrain them on all available data, and finally predict the missing values.


## Points of Improvement 

* Feature Selection!!!
* Parallelize all model (n_jobs=-1)
* I need to comment my code more!
* Pickle system file is nice for saving python object, but binary files and Git ...

