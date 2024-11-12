# Stock-market-Analysis
The stock market is known for its dynamic and volatile nature, making accurate price prediction a challenging task. In recent years, machine learning models have gained traction for stockmarket forecasting due to their ability to uncover hidden patterns and trends from historical data.

This is a project on analysing and predicting stock market. The prediction models are built on python using Tensorflow and Sci-kit Learn. The prediction is based on comparing three different machine learning model and presents the most approximated results. The models used in prediction are Long Short Term Memory(LSTM), Linear Regression and Random Forest. This study explores three different machine learning techniques—Linear Regression, Random Forest, and Long Short-Term Memory (LSTM) neural networks—comparing their effectiveness in predicting stock prices. The goal is to evaluate the performance of these models using historical stock price data, to better understand their prediction capabilities and limitations.

The methodology involves preprocessing historical stock price data, including normalization and data splitting into training and test sets. Linear Regression, a statistical approach, is first  implemented to model the relationship between stock prices and their historical data. Random Forest, an ensemble learning method, is then applied to reduce overfitting by creating multiple decision trees and aggregating their predictions. Finally, LSTM, a type of recurrent neural network (RNN) specialized for timeseries forecasting, is used to model the sequential dependencies in the stock price data. Each model is trained and tested, and their predictive performance is evaluated based on key metrics such as Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE). Visual comparisons between actual and predicted stock prices further illustrate the models' performance.

The results reveal that the LSTM model, designed to handle sequential data, outperforms both Linear Regression and Random Forest in predicting stock price movements. LSTM's ability to capture longterm dependencies in stock price patterns provides a significant edge over the other models. However, Random Forest demonstrates robustness and accuracy in nonsequential data prediction, while Linear Regression, although the simplest of the three, still offers valuable insights in capturing general trends in stock prices.

In conclusion, while no model is flawless, the LSTM neural network shows the most promise for accurate stock market prediction due to its capacity to learn from historical sequences. Random Forest proves to be a solid contender for general pattern recognition, while Linear Regression serves as a baseline model for trend identification. This comparative analysis highlights the importance of selecting appropriate machine learning models based on thenature of the data and the specific forecasting goals. Future work can explore hybrid models that integrate the strengths of multiple techniques for improved stock price prediction accuracy.
