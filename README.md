# Stock Bot

An automated tool for monitoring stock market price drops and sending email alerts when significant changes are detected.

## Overview

Stock Bot fetches real-time stock data from the StockAnalysis API, analyzes price movements to identify significant drops, and sends customized email alerts to notify users of potential investment opportunities or risks. The system is designed to be resilient with robust error handling, retry mechanisms, and fallback options.

## Features

- **Real-time Stock Monitoring**: Fetches comprehensive data for thousands of stocks
- **Drop Detection Algorithm**: Identifies stocks that have dropped by a configurable threshold
- **Email Alerts**: Sends professionally formatted HTML emails with detailed stock information
- **Data Persistence**: Stores fetched stock data in JSON format for historical analysis
- **Robust Error Handling**: Implements retry mechanisms and graceful degradation
- **Comprehensive Logging**: Maintains detailed logs of all operations and errors

## Requirements

```
yfinance
pandas
requests
tenacity
logging
```

## Project Structure

```
stock_bot/
│
├── main.py              # Main application script
├── requirements.txt     # Python dependencies
├── README.md           # Project documentation
│
├── data/               # Stock data storage
│   └── stocks_data_*.json  # Historical stock data files
│
└── logs/               # Application logs
    └── stock_alerts_*.log  # Daily log files
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/MuhammadKashif124/stock_bot.git
   cd stock_bot
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure email settings:
   - Edit the `main.py` file to update the email configuration:
     - `from_email`: Your sender email address
     - `to_email`: Recipient email address
     - `smtp_user`: Your SMTP username (typically your email address)
     - `smtp_password`: Your SMTP password or app password
   
   Note: For Gmail, you'll need to use an App Password (https://myaccount.google.com/apppasswords)

## Usage

Run the script to fetch stock data and check for significant drops:

```bash
python main.py
```

The script will:
1. Fetch current stock data from the StockAnalysis API
2. Analyze stocks for significant price drops (default threshold: 50%)
3. Send email alerts if any stocks meet the threshold criteria
4. Save stock data and logs for future reference


## Configuration

The main threshold for stock drop alerts can be configured by modifying the `threshold` parameter in the `main()` function call:

```python
success = main(threshold=25)  # Change to your desired percentage (e.g., 25%)
```

## Logging

Logs are stored in the `logs/` directory with the naming pattern `stock_alerts_YYYYMMDD.log`. These logs include:
- API request details
- Stock processing information
- Alert generation events
- Email sending status
- Error messages and exceptions

## Data Saving

All successfull run will save a latest stock data in the form of json for future use or analysis.