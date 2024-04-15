# Taller de Aprendizaje Automatico (Machine Learning Workshop)
#### Facultad de Ingenieria, UdelaR. 2024

The course is based on Geron's "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems".

This repository contains the individual work done in the course (although each workshop starts in an in-person class so we get to discuss and share details with other students and profs). There's one workshop ("taller" in spanish) per week, contained in each `tN` folder (the `t` stands for "taller").

## Workshop summaries and conclusions

1. Titanic survivor binary classifier, based on features of every passenger. Concluded that the most significant features are the passenger's class and sex (as women were prioritized in the lifeboats).
2. IMDB movie reviews sentiment binary classifier. Final result of 87% accuracy in test set (we weren't taking into account precision, recall, F1 and other metrics yet), using stop words, bigrams and tf-idf metric for weighting.
3. Bike rental regressor. Cross validation RMSLE of ~0.37 (the error would probable be a lot higher on the test set) using random forest with gradient boosting (XGBoost), Kaggle top 5 leaderboard error is in the 0.35 ball park.
