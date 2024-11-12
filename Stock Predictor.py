import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from keras.models import load_model # type: ignore
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import joblib as jb
import yfinance as yf

st.set_page_config(page_title='Stock Analyzer!', page_icon = "📈")
st.title("Stock trend Prediction")

if 'ticker' not in st.session_state:
    st.session_state.ticker = ""

# Sidebar input
with st.sidebar:
    ticker = st.text_input('Enter Stock Symbol', value=st.session_state.ticker)
    start = st.date_input('Start Date', pd.to_datetime('2000-01-01'))
    end = st.date_input('End Date')
    if ticker:
        st.session_state.ticker = ticker

# Main content
if st.session_state.ticker:
    df = yf.download(st.session_state.ticker, start=start, end=end)
    df = df.reset_index()

    st.subheader(f'Data from {start} - {end}')
    st.write(df.describe())

    # Create tabs for different charts
    tab1, tab2, tab3, tab4 = st.tabs(["Closing Price Chart", "MA Chart", "MA Comparison", "Predictions"])

    with tab1:
        st.subheader('Closing Price vs Time "Chart"')
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Closing Price'))
        fig1.update_layout(title='Closing Price vs Time', xaxis_title='Date', yaxis_title='Price', template='plotly_white')
        st.plotly_chart(fig1)

    with tab2:
        st.subheader('Closing Price vs Time chart with 100MA')
        ma100 = df['Close'].rolling(window=100).mean()
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Closing Price', line=dict(color='blue')))
        fig2.add_trace(go.Scatter(x=df['Date'], y=ma100, mode='lines', name='100 Day MA', line=dict(color='red')))
        fig2.update_layout(title='Closing Price vs 100 Day Moving Average', xaxis_title='Date', yaxis_title='Price', template='plotly_white')
        st.plotly_chart(fig2)

    with tab3:
        st.subheader('Closing Price vs Time chart with 100MA and 200MA')
        ma200 = df['Close'].rolling(window=200).mean()
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Closing Price', line=dict(color='blue')))
        fig3.add_trace(go.Scatter(x=df['Date'], y=ma100, mode='lines', name='100 Day MA', line=dict(color='red')))
        fig3.add_trace(go.Scatter(x=df['Date'], y=ma200, mode='lines', name='200 Day MA', line=dict(color='green')))
        fig3.update_layout(title='Closing Price vs 100 Day and 200 Day Moving Average', xaxis_title='Date', yaxis_title='Price', template='plotly_white')
        st.plotly_chart(fig3)

    # Prepare data for predictions
    train = pd.DataFrame(df[['Date', 'Close']][0:int(len(df)*0.75)])
    test = pd.DataFrame(df[['Date', 'Close']][int(len(df)*0.75):])

    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train[['Close']])
    model = load_model('/home/yuvan/Programs/Projects/stock-predictor/predict_stock.h5')

    # Prepare test data
    past_100_days = train.tail(100)
    final_test = pd.concat([past_100_days, test], ignore_index=True)
    inputs = scaler.transform(final_test[['Close']])  # Use the same scaler to transform the input
    x_test = []
    y_test = []

    for i in range(100, inputs.shape[0]):
        x_test.append(inputs[i-100:i])
        y_test.append(inputs[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)

    # Make predictions
    y_pred = model.predict(x_test)

    # Scale back the predictions
    y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    def lr(df):
        scaler1 = jb.load('/home/yuvan/Programs/Projects/stock-predictor/scaler.pkl')
        model1 = jb.load('/home/yuvan/Programs/Projects/stock-predictor/stock_predictor_model.pkl')

        data = df
        if not df.empty:
            data['Close'] = df['Close'].fillna(method='ffill')
            
            data_for_prediction = np.array(df['Close']).reshape(-1, 1)
            scaled_data = scaler1.transform(data_for_prediction)

            predictions = model1.predict(scaled_data)

        return predictions

    def rf(df):
        scaler2 = jb.load('/home/yuvan/Programs/Projects/stock-predictor/scaler2.pkl')
        model2 = jb.load('/home/yuvan/Programs/Projects/stock-predictor/random_forest_stock_model.pkl')

        data1 = df
        if not df.empty:
            data1['Close'] = df['Close'].fillna(method='ffill')
            data_for_prediction = df[['Close']].values

            scaled_data = scaler2.transform(data_for_prediction)
            window_size = 10
            x_te = []
            for i in range(window_size, len(scaled_data)):
                x_te.append(scaled_data[i-window_size:i,0])
            x_te = np.array(x_te)

            predictions_scaled = model2.predict(x_te)

            predictions = scaler.inverse_transform(predictions_scaled.reshape(-1, 1))
        return predictions
    def calculate_accuracy(y, y_):
        min_length = min(len(y), len(y_))
        y = y[-min_length:]
        y_ = y_[-min_length:]
        return r2_score(y, y_)
    
    lstm_acc = calculate_accuracy(y_test, y_pred)
    lr_acc = calculate_accuracy(df['Close'], lr(df))
    rf_acc = calculate_accuracy(df['Close'], rf(df))


    with tab4:
        st.header('Predictions')
        tab5, tab6, tab7 = st.tabs(['LSTM Model', 'Linear Regressing', 'Random Forest'])
        with tab5:
            st.subheader('Predictions vs Original')
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(x=list(range(len(y_test))), y=y_test.flatten(), mode='lines', name='Original Price', line=dict(color='blue')))
            fig4.add_trace(go.Scatter(x=list(range(len(y_pred))), y=y_pred.flatten(), mode='lines', name='Predicted Price', line=dict(color='red')))
            fig4.update_layout(title='Predictions vs Original Prices', xaxis_title='Time', yaxis_title='Price', template='plotly_white')
            st.plotly_chart(fig4)
        with tab6:
            st.subheader('Predictions vs Original')
            fig5 = go.Figure()
            fig5.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Original Price', line=dict(color='blue')))
            fig5.add_trace(go.Scatter(x=df.index, y=lr(df), mode='lines', name='Predicted Price', line=dict(color='red')))
            fig5.update_layout(title='Predictions vs Original Prices', xaxis_title='Time', yaxis_title='Price', template='plotly_white')
            st.plotly_chart(fig5)
        with tab7:
            st.subheader('Predictions vs Original')
            fig6 = go.Figure()
            fig6.add_trace(go.Scatter(x=df.index[10:], y=df['Close'][10:], mode='lines', name='Actual Prices', line=dict(color='blue')))
            fig6.add_trace(go.Scatter(x=df.index[10:], y=rf(df).flatten(), mode='lines', name='Predicted Prices', line=dict(color='red')))

            fig6.update_layout(title='Predictions vs Original Prices', xaxis_title="Time", yaxis_title='Price', template='plotly_white')
            st.plotly_chart(fig6)
        
        if lstm_acc >= lr_acc and lstm_acc >= rf_acc:
            st.write(f"The next day Stock Price of {ticker} is ${y_pred[0][0]:.2f} predicted by LSTM Model with accuracy {lstm_acc * 100:.2f}%")
        elif lr_acc >= lstm_acc and lr_acc >= rf_acc:
            st.write(f"The next day Stock Price of {ticker} is ${lr(df)[-1]:.2f} predicted by Linear Regression Model with accuracy {lr_acc * 100:.2f}%")
        elif rf_acc >= lstm_acc and rf_acc >= lr_acc:
            st.write(f"The next day Stock Price of {ticker} is ${rf(df)[0]:.2f} predicted by Random Forest Model with accuracy {rf_acc * 100:.2f}%")
        else:
            st.write(f"Cannot fetch the next day Stock Price for {ticker}")



else:
    st.info("Please enter a stock symbol in the sidebar to see the results.")