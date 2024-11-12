import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

def get_stock_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, group_by='ticker')
    if data.empty:
        st.error(f"No data found for {ticker} between {start} and {end}.")
    return data

def plot_stock_data(df, title):
    fig = go.Figure()
    if 'Close' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name=title))
    else:
        st.error(f"Missing 'Close' column in the data for {title}.")
    fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Price')
    return fig

def get_top_gainers_losers(df):
    if 'Close' not in df.columns or 'Open' not in df.columns:
        st.error("Data is missing required 'Close' or 'Open' columns.")
        return pd.DataFrame(), pd.DataFrame()
    
    df['Daily Change %'] = (df['Close'] - df['Open']) / df['Open'] * 100
    df['Stock Symbol'] = df.index
    top_gainers = df.nlargest(5, 'Daily Change %')[['Stock Symbol', 'Open', 'Close', 'Daily Change %']]
    top_losers = df.nsmallest(5, 'Daily Change %')[['Stock Symbol', 'Open', 'Close', 'Daily Change %']]
    return top_gainers, top_losers

st.set_page_config(page_title='Stock Analyzer!', page_icon="📈")

st.title("Stock Analyzer 📈")
st.sidebar.success("")

start_date = st.date_input('Start Date', pd.to_datetime('2023-01-01'))
end_date = st.date_input('End Date', pd.to_datetime('today'))

sensex_tickers = ["RELIANCE.BO", "TCS.BO", "HDFCBANK.BO", "INFY.BO", "HINDUNILVR.BO"]
nifty_tickers = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS"]

sensex_data = get_stock_data('^BSESN', start=start_date, end=end_date)
nifty_data = get_stock_data('^NSEI', start=start_date, end=end_date)

top_gainers_sensex, top_losers_sensex = get_top_gainers_losers(sensex_data)
top_gainers_nifty, top_losers_nifty = get_top_gainers_losers(nifty_data)

sensex_tab, nifty_tab = st.tabs(["Sensex", "Nifty"])

# Sensex Tab
with sensex_tab:
    st.subheader("Sensex Data")
    sensex_chart_tab, sensex_gainers_tab, sensex_losers_tab = st.tabs(["Sensex Chart", "Top Gainers", "Top Losers"])
    
    with sensex_chart_tab:
        st.plotly_chart(plot_stock_data(sensex_data, "Sensex"))

    with sensex_gainers_tab:
        st.subheader("Top 5 Gainers in Sensex")
        st.dataframe(top_gainers_sensex)

    with sensex_losers_tab:
        st.subheader("Top 5 Losers in Sensex")
        st.dataframe(top_losers_sensex)

with nifty_tab:
    st.subheader("Nifty Data")
    
    nifty_chart_tab, nifty_gainers_tab, nifty_losers_tab = st.tabs(["Nifty Chart", "Top Gainers", "Top Losers"])
    
    with nifty_chart_tab:
        st.plotly_chart(plot_stock_data(nifty_data, "Nifty"))

    with nifty_gainers_tab:
        st.subheader("Top 5 Gainers in Nifty")
        st.dataframe(top_gainers_nifty)

    with nifty_losers_tab:
        st.subheader("Top 5 Losers in Nifty")
        st.dataframe(top_losers_nifty)
