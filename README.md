Stock Price Predictor (LSTM + Streamlit)
This project is a Stock Price Prediction Application built with Python, TensorFlow, and Streamlit. It uses historical stock data from the Alpha Vantage API to train an LSTM model, which predicts the next day's closing price.

🚀 Features

Fetches real-time stock data (e.g., TSLA, AAPL, MSFT) using the Alpha Vantage API.
Preprocesses data with MinMaxScaler and creates time-series sequences for training.
Trains an LSTM neural network to capture stock price patterns.
Provides a Streamlit web interface for real-time predictions.
Visualizes the last 30 days of stock prices along with the predicted price.


⚙️ Setup Instructions
1️⃣ Clone Repository
git clone https://github.com/yourusername/stock-price-predictor.git
cd stock-price-predictor

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Set API Key
Obtain a free API key from Alpha Vantage and set it as an environment variable:
export ALPHA_VANTAGE_KEY=your_api_key_here

4️⃣ Fetch Stock Data
Run the data fetching script:
python data_fetcher.py

5️⃣ Train Model
Train the LSTM model using the fetched data:
python train_model.py

6️⃣ Run Streamlit App
Launch the Streamlit web application:
streamlit run app.py
