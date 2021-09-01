Summary of contents in this repo:
	Folders:
		1) dataScienceTask/: contains original data
		2) exploration/: contains jupyter notebooks for exploratory analysis
		3) Datasets/: contains train-val-test processed datasets
	Python Files:
		1) data_processing_demo.py: pre-proceesing steps for data on demographics
		2) data_processing_indicators.py: pre-proceesing steps for data on health indicators (Eg. creatinine)
		3) data_processing_meds.py : pre-proceesing steps for data on medications
		4) split_datasets.py: splits processed data after pre-processing to train-val-test sets
		5) train.py: trains the deep learning model with training dataset and validate with validation dataset
		6) test.py: evaluate with test dataset
		7) model.py: contains the model constructors
		8) utils.py: contains functions needed for python files above
	Others:
		Include csv files that are products and inputs of intermediate steps.