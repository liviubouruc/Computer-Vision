1. the libraries required to run the project including the full version of each library;
python version==3.9.6

numpy==1.21.4
opencv_python==4.5.4
sklearn==1.0.2
skimage==0.19.1
matplotlib==3.5.0

2. how to run each task and where to look for the output file.

Example:

For Task 1 & Task 2:
script: RunProject.py
function: Run Project will first try and upload the positive and negative descriptors. If they don't exist, it creates them. Then it will train the classifier with the descriptors. Finally, it will run the sliding window and safe the detections, scores and filenames
output: the output files are in 'evaluare\\fisiere_solutie\\Liviu_Bouruc_334\\task1\\' and 'evaluare\\fisiere_solutie\\Liviu_Bouruc_334\\task2\\'