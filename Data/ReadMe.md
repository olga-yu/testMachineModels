~~1. 2017_sem2_attendance_data.csv - original file collected by UNSW Sutjarittham, T., Gharakheili, H.H., Kanhere, S.S. and Sivaraman, V. (2019). Experiences with IoT and AI in a Smart Campus for Optimizing Classroom Usage. IEEE Internet of Things Journal, pp.1–1. doi: https://doi.org/10.1109/jiot.2019.2902410~~
is not used in the current code
2. Neural Networks - adapted from Natalia Yerashenia https://github.com/Yerashenia/Predictive-Computational-Model-PCM/tree/master/Market_Index_Prediction
https://github.com/Yerashenia/Predictive-Computational-Model-PCM/tree/master/Bankruptcy_Prediction
3. Data transform takes place in data_transform.py to transform original textual data into numbered data
- the data is saved to result_output.csv
- data preprocessing  with data2017.csv
-- attendance is sorted against possible bigger and larger rooms
-- attendance trend is  reviewed closer to the end of semester
- comparison is made to original lecture and tutorial sorting and possible bigger and smaller lecture
# bigger and smaller size tutorial size rooms instead
4. CAP_NN.py - contains neural network that should produce predicted attendance, but produces. [[1.]]
5. CAP_NN_testing - test against  new data result_output.csv, compare results MAE, MSE.
- Compare predictions against actual results, but it predictions do not exist
- produce graphs
6. the time is divided into following groups in the realdata.csv
11:00-12:59 - 1
13:00-14:59 - 2
15:00-16:59 - 3
17:00-18:59 - 4 
-----
Results for realdata.csv 
Root Mean Square Error: 0.28070681980741335
Mean Absolute Error: 0.12440689997148525
R squared: 89.54
Mean Square Error: 0.07879631868639164

Process finished with exit code 0