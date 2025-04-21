import os
import json
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import argparse
import pytz
import numpy as np

def get_speech_dates(format_folders):
    """
    Extract dates and filenames from speech JSON files in the specified format folders.
    
    Args:
        format_folders (list): List of folder names to process
        
    Returns:
        pd.DataFrame: DataFrame with speech dates and filenames
    """
    dates = []
    filenames = []
    
    for folder in format_folders:
        if not os.path.exists(folder):
            print(f"Warning: Folder {folder} not found")
            continue
            
        json_files = [f for f in os.listdir(folder) if f.endswith('.json')]
        for json_file in json_files:
            try:
                with open(os.path.join(folder, json_file), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Parse date and localize to NY timezone
                    speech_date = datetime.strptime(data["date"], "%m/%d/%Y")
                    speech_date = pytz.timezone('America/New_York').localize(speech_date)
                    dates.append(speech_date)
                    filenames.append(json_file)
            except Exception as e:
                print(f"Error processing {json_file}: {str(e)}")
    
    return pd.DataFrame({
        'date': dates,
        'filename': filenames
    }).sort_values('date')

def calculate_momentum(returns):
    """Calculate momentum as cumulative return over a period."""
    print(f"Returns used for momentum calculation: {returns.values}")
    return (1 + returns).prod() - 1

def calculate_volatility(returns):
    """Calculate unannualized volatility."""
    print(f"Returns used for volatility calculation: {returns.values}")
    return returns.std(ddof=1) 

def get_asset_data(dates, ticker, days_before=5, days_after=10, 
                   momentum_before=5, momentum_after=10,
                   volatility_before=5, volatility_after=10):
    """
    Fetch asset price data for specified dates.
    
    Args:
        dates (pd.Series): Series of speech dates
        ticker (str): Asset ticker symbol
        days_before (int): Number of trading days before speech to fetch
        days_after (int): Number of trading days after speech to fetch
        momentum_before (int): Number of days for pre-speech momentum calculation
        momentum_after (int): Number of days for post-speech momentum calculation
        volatility_before (int): Number of days for pre-speech volatility calculation
        volatility_after (int): Number of days for post-speech volatility calculation
        
    Returns:
        pd.DataFrame: DataFrame with price data for each speech date
    """
    all_data = []
    
    # Get asset data
    asset = yf.Ticker(ticker)
    
    # Process each speech date
    for speech_date in dates:
        max_days = max(days_before, momentum_before, volatility_before)
        start_date = speech_date - timedelta(days=max_days * 2)  # Extra buffer for market holidays
        
        max_days_after = max(days_after, momentum_after, volatility_after)
        end_date = speech_date + timedelta(days=max_days_after * 2)
        
        hist_data = asset.history(start=start_date, end=end_date)
        
        if hist_data.empty:
            print(f"No data available for {speech_date}")
            continue
        
        # Calculate daily returns
        hist_data['Returns'] = hist_data['Close'].pct_change()
        
        # Split data into before and after speech
        before_data = hist_data[hist_data.index < speech_date].tail(days_before)
        after_data = hist_data[hist_data.index > speech_date].head(days_after)
        
        speech_day_data = hist_data[hist_data.index.date == speech_date.date()]
        if not speech_day_data.empty:
            speech_close = speech_day_data['Close'].iloc[-1]
            if not after_data.empty:
                next_day_open = after_data['Open'].iloc[0]
                next_day_close = after_data['Close'].iloc[0]
                # Calculate percentage returns
                pre_market_movement = (next_day_open - speech_close) / speech_close  # Overnight return
                after_1_day_market = (next_day_close - speech_close) / speech_close  # One-day return
            else:
                pre_market_movement = None
                after_1_day_market = None
        else:
            speech_close = None
            pre_market_movement = None
            after_1_day_market = None
        
        # Get data for momentum and volatility calculations
        before_momentum_data = hist_data[hist_data.index < speech_date].tail(momentum_before)
        after_momentum_data = hist_data[hist_data.index > speech_date].head(momentum_after)
        before_volatility_data = hist_data[hist_data.index < speech_date].tail(volatility_before)
        after_volatility_data = hist_data[hist_data.index > speech_date].head(volatility_after)
        
        row_data = {
            'ticker': ticker,
            'speech_date': speech_date,
            'speech_close': speech_close,
            'pre_market_movement': pre_market_movement,
            'after_1_day_market': after_1_day_market
        }
        
        # Add before prices
        for i, (idx, row) in enumerate(before_data.iterrows(), 1):
            row_data[f'before_{days_before-i+1}_open'] = row['Open']
            row_data[f'before_{days_before-i+1}_close'] = row['Close']
            
        # Add after prices
        for i, (idx, row) in enumerate(after_data.iterrows(), 1):
            row_data[f'after_{i}_open'] = row['Open']
            row_data[f'after_{i}_close'] = row['Close']
        
        # Add momentum
        if not before_momentum_data.empty:
            row_data[f'momentum_{momentum_before}d_before'] = calculate_momentum(before_momentum_data['Returns'].dropna())
        else:
            row_data[f'momentum_{momentum_before}d_before'] = None
            
        if not after_momentum_data.empty:
            row_data[f'momentum_{momentum_after}d_after'] = calculate_momentum(after_momentum_data['Returns'].dropna())
        else:
            row_data[f'momentum_{momentum_after}d_after'] = None
        
        # Add volatility
        if not before_volatility_data.empty:
            row_data[f'volatility_{volatility_before}d_before'] = calculate_volatility(before_volatility_data['Returns'].dropna())
        else:
            row_data[f'volatility_{volatility_before}d_before'] = None
            
        if not after_volatility_data.empty:
            row_data[f'volatility_{volatility_after}d_after'] = calculate_volatility(after_volatility_data['Returns'].dropna())
        else:
            row_data[f'volatility_{volatility_after}d_after'] = None
            
        all_data.append(row_data)
    
    return pd.DataFrame(all_data)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate asset datasets based on speech dates')
    parser.add_argument('group', type=str, choices=['1', '2', 'both'], 
                       help='Group to process (1, 2, or both)')
    parser.add_argument('--days-before', type=int, default=5,
                       help='Number of trading days before speech to fetch prices')
    parser.add_argument('--days-after', type=int, default=10,
                       help='Number of trading days after speech to fetch prices')
    parser.add_argument('--momentum-before', type=int, default=3,
                       help='Number of days for pre-speech momentum calculation')
    parser.add_argument('--momentum-after', type=int, default=5,
                       help='Number of days for post-speech momentum calculation')
    parser.add_argument('--volatility-before', type=int, default=3,
                       help='Number of days for pre-speech volatility calculation')
    parser.add_argument('--volatility-after', type=int, default=5,
                       help='Number of days for post-speech volatility calculation')
    
    args = parser.parse_args()
    
    # Define format folders for each group with correct paths
    group_folders = {
        '1': ['output/Format 1 JSON', 'output/Format 2 JSON'],
        '2': ['output/Format 3 JSON', 'output/Format 4 JSON']
    }
    
    # Determine which groups to process
    groups_to_process = ['1', '2'] if args.group == 'both' else [args.group]
    
    # Process each group
    for group in groups_to_process:
        print(f"\nProcessing Group {group}...")
        
        # Get speech dates
        speech_df = get_speech_dates(group_folders[group])
        print(f"Found {len(speech_df)} speeches in Group {group}")
        
        # Get asset data for each ticker
        tickers = {
            'SPY': 'sp500', 
            'TLT': 'bonds'
        }
        
        for ticker, asset_name in tickers.items():
            print(f"\nFetching {asset_name.upper()} data...")
            
            asset_df = get_asset_data(
                speech_df['date'], 
                ticker,
                days_before=args.days_before,
                days_after=args.days_after,
                momentum_before=args.momentum_before,
                momentum_after=args.momentum_after,
                volatility_before=args.volatility_before,
                volatility_after=args.volatility_after
            )
            
            merged_df = pd.merge(
                speech_df,
                asset_df,
                left_on='date',
                right_on='speech_date',
                how='left'
            )
            
            merged_df = merged_df.drop('speech_date', axis=1)
            
            output_file = f'group{group}_{asset_name}_data.csv'
            merged_df.to_csv(output_file, index=False)
            print(f"Saved {asset_name} data to {output_file}")

if __name__ == "__main__":
    main() 
