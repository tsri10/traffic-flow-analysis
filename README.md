## Traffic Flow Prediction using SparkMLlib and TensorFlow

This project focuses on predicting traffic flow using two prominent machine learning tools: Apache Spark’s MLlib and TensorFlow. The goal is to leverage big data processing capabilities and machine learning models to forecast traffic patterns and compare the efficiency and accuracy of these frameworks.

### Project Overview

With increasing traffic congestion in urban areas, accurate traffic prediction is crucial for efficient transportation management. This project uses traffic data collected from multiple junctions and applies machine learning models to predict the number of vehicles passing through each junction at different time intervals. By utilizing SparkMLlib for scalable machine learning and TensorFlow for deep learning, the project compares their effectiveness in predicting traffic flow.

### Key Features

  •	Data Processing with SparkMLlib: Applied MLlib to handle large-scale traffic data efficiently, leveraging Spark’s parallel processing capabilities.
  
  •	Deep Learning with TensorFlow: Built a GRU (Gated Recurrent Unit) neural network using TensorFlow to predict traffic flow, achieving a high level of accuracy.
  
  •	Feature Engineering: Extracted relevant features from datetime fields and processed the data for time series analysis.
  
  •	Comparative Analysis: Evaluated both approaches using RMSE (Root Mean Squared Error) to compare their performance in traffic prediction.

### Dataset

  •	Source: Traffic Data consisting of 48,120 observations from four different junctions.
  
  •	Features: DateTime, Junction ID, Number of Vehicles.  
  
### Tools and Technologies

  •	Apache Spark: For large-scale data processing and machine learning using SparkMLlib.
  
  •	TensorFlow: For building and training GRU-based neural networks.
  
  •	Pandas & NumPy: For data manipulation and feature engineering.
  
  •	Matplotlib & Seaborn: For data visualization and exploratory analysis.

### Model Performance

•	GRU Neural Network with TensorFlow:
  •	Achieved RMSE of 0.24 for Junction 1, indicating good predictive accuracy.
    
  •	Trained using stochastic gradient descent (SGD) and optimized using Dropout layers to prevent overfitting.
    
•	SparkMLlib:
  
  •	Leveraged Spark's parallel processing to handle large datasets, providing a scalable approach to traffic prediction.

### Results and Insights

  •	Junction 1 exhibited a significantly higher traffic volume compared to other junctions, with a clear pattern of weekly and daily traffic variations.
  
  •	The TensorFlow-based model demonstrated superior accuracy for Junction 1, while SparkMLlib performed better in terms of processing speed and handling larger datasets.
  
  •	The GRU model proved effective for time-series data prediction, especially when traffic patterns followed distinct seasonal and hourly trends.

### Conclusion

This project successfully demonstrates the use of big data tools like Spark and deep learning frameworks like TensorFlow for traffic flow prediction. By comparing both approaches, we highlight their strengths in different contexts—TensorFlow for its predictive accuracy in time-series forecasting and SparkMLlib for its scalability and performance with large datasets.



