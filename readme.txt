Command to run our program:

2.Regression

	python simple_linear_regression.py    

Then input the attribute you want to use eg. X1

	python multi_linear_regression.py

3.Classification
	python multi-layer-perceptron-single.py

	python multi-layer-perceptron-multiple.py

Then input the file path:

data/jumping jacks sets.xlsx
data/lunges sets.xlsx
data/squat sets.xlsx

For single axis you also need to choose which axis you want to use. x/y/z



Set separation:

First we find all peaks(or bottoms) in dataset. 
For jumping jacks, we set the threshold to mean*0.5 and the minimum height of all peaks to 10. Then, we delete all data in the period of no changes. After that, we select each set from peak-1 to peak + 3, and add another 3 zeros to get total size 8.
For lunges, we set the lower bond of bottom to 0 and the minimal distance between bottoms to 5. Then, we select each set from bottom-2 to bottom+2 and add another 3 zeros to get total size 8.
For squat, we set the minimal distance to 3, then we delete all bottoms in the range between 9.5 to 10.5. If there are still more than 18 bottoms, we delete all bottoms larger than 9.3. Finally we select the sets between two bottoms with size 8.


for all questions, we randomly choose a step as our testing data. Also we shuffled the training data to avoid our model learning sequence of training data, so every run of the program could get different result.