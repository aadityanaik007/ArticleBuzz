# ArticleBuzz

Description:

Predict the future popularity of your articles with the Article Share Count Predictor app. This GitHub repository hosts a user-friendly application that leverages advanced analytics to forecast an article's reach based on social media shares. Receive real-time insights, optimization suggestions, and stay ahead in the dynamic world of digital content.

Clone the repository.
Run the app on your local machine.
Input your article details to receive predictions and insights.
Elevate your content strategy - welcome to the future of article prediction!

Installation

1. Clone the repository:
git clone https://github.com/your_username/your_repository.git

2. Install the required Python packages:
pip install -r requirements.txt

Usage
1.Ensure you have the necessary data file (OnlineNews.csv) in the specified location.

2.Run the upload_and_train_model function to train the machine learning model:

3. Once the model is trained, you can use the predict_news_shares function to make predictions:

4.To run the web application locally
i)Run the backend server
 -- python3 Backend/main.py

Data
The dataset used for training and testing the model is stored in the file OnlineNews.csv. It contains various features related to online news articles, including the number of tokens, keywords, and other relevant information.

Training
The model is trained using the GradientBoostingRegressor from the scikit-learn library. Data cleaning and preprocessing steps are performed to handle outliers and scale the features appropriately.

Prediction
The trained model is used to make predictions on new data. The predicted share counts are compared with the actual share counts to evaluate the model's performance.

Evaluation
Model performance is evaluated using metrics such as Mean Absolute Error (MAE) and Root Mean Square Error (RMSE). These metrics provide insights into how well the model is performing in predicting the number of shares.
## API Reference

#### Get all items

```http
  POST /get_prediction_for_csv
```

| Body | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `file` | `file` | **Required**. Data for prediction |

#### Check Status

```http
  GET /check_status
```
| Body | Type     | Description                |
| :-------- | :------- | :------------------------- |
|  |  | Check the model status |

```http
  POST /prediction_api
```
| Body | Type     | Description                |
| :-------- | :------- | :------------------------- |
|  data | `Dict` | **Required**. Lambda prediction endpoint |

```http
 GET /
```
| Body | Type     | Description                |
| :-------- | :------- | :------------------------- |
|  |  | **Required**. HTML display |
