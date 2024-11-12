import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from tensorflow import keras
#from keras.models import load_model
import streamlit as st
import plotly.express as px
import yfinance as yf


st.set_page_config(page_title='Stock Analyzer!', page_icon = "📈")
st.title("Stock trend Analysis")

if 'ticker' not in st.session_state:
    st.session_state.ticker = ""

with st.sidebar:
    ticker = st.text_input('Enter Stock Symbol', value=st.session_state.ticker)
    start = st.date_input('Start Date', pd.to_datetime('2000-01-01'))
    end = st.date_input('End Date')
    if ticker:
        st.session_state.ticker = ticker


if st.session_state.ticker:
    df = yf.download(ticker, start, end)

    fig = px.line(df, x=df.index, y = df['Close'], title = ticker)
    st.plotly_chart(fig)

    pricings, fundamentals, news = st.tabs(['Pricings', 'Fundamentals', 'News'])

    with pricings:
        st.header("Pricing Movements")
        data = df
        data['% Change'] = df['Close'] / df['Close'].shift(1)
        data.dropna(inplace = True)
        st.write(data)
        annual_return = data['% Change'].mean()*252*100
        st.write('Annual return is', annual_return, '%')
        sd = np.std(data['% Change']) * np.sqrt(252)
        st.write('Standard Deviation is', sd*100, '%')
        st.write('Risk Adj. Return is', annual_return/sd*100)

    from alpha_vantage.fundamentaldata import FundamentalData as f
    with fundamentals:
        key = "7NVK89JAQ4DP75JA"
        fd = f(key, output_format = 'pandas')
        st.subheader('Balance Sheet')
        balance_sheet = fd.get_balance_sheet_annual(ticker)[0]
        bs = balance_sheet.T[2:]
        bs.columns = list(balance_sheet.T.iloc[0])
        st.write(bs)
        st.subheader('Income Statement')
        income_statement = fd.get_income_statement_annual(ticker)[0]
        i = income_statement.T[2:]
        i.columns = list(income_statement.T.iloc[0])
        st.write(i)
        st.subheader('Cash Flow Statement')
        cash_flow = fd.get_cash_flow_annual(ticker)[0]
        cf = cash_flow.T[2:]
        cf.columns = list(cash_flow.T.iloc[0])
        st.write(cf)
        
    from stocknews import StockNews
    with news:
        st.header(f'News about {ticker}')
        sn = StockNews(ticker, save_news=False)
        df_news = sn.read_rss()
        for i in range(15):
            st.subheader(f'News{i + 1}')
            st.write(df_news['published'][i])
            st.write(df_news['title'][i])
            st.write(df_news['summary'][i])
            title_sentiment = df_news['sentiment_title'][i]
            st.write(f'Title Sentiment {title_sentiment}')
            news_sentiment = df_news['sentiment_summary'][i]
            st.write(f'News Sentiment {news_sentiment}')
else:
    st.info("Please enter a stock symbol in the sidebar to see the results.")
