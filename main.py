import requests
import json
import time
import os
import logging
import sys
import yfinance as yf
import pandas as pd
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Configure console for proper encoding (especially for Windows)
def configure_console_encoding():
    """
    Configure console encoding to handle special characters correctly
    """
    try:
        # For Windows: attempt to set console encoding to UTF-8
        if sys.platform == 'win32':
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleOutputCP(65001)  # Set to UTF-8
            
            # Force stdout/stderr to use UTF-8
            sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
            sys.stderr.reconfigure(encoding='utf-8', errors='backslashreplace')
    except Exception as e:
        # If this fails, we'll fall back to ASCII-safe logging
        print(f"Warning: Could not configure console encoding: {e}")

# Configure console encoding
configure_console_encoding()

# Ensure the logs directory exists
log_dir = "logs"
try:
    os.makedirs(log_dir, exist_ok=True)
except Exception as e:
    print(f"Warning: Could not create logs directory: {e}")
    log_dir = "."  # Fallback to current directory

log_file = os.path.join(log_dir, f"stock_alerts_{time.strftime('%Y%m%d')}.log")

# Configure logging
# Use utf-8 encoding for log file to support emojis and special characters
try:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
except Exception as e:
    # Fallback to stdout only if file logging fails
    print(f"Warning: Could not set up file logging: {e}")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

logger = logging.getLogger("stock_alerts")

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((smtplib.SMTPException, ConnectionError, TimeoutError)),
    reraise=True
)
def send_email_via_smtp(subject, html_body, plain_body, to_email, from_email, smtp_user, smtp_password):
    """
    Send an email via SMTP with retry mechanism.
    Will retry up to 3 times with exponential backoff if network errors occur.
    """
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email

    # Attach both plain text and HTML (Gmail prefers HTML but fallback matters)
    msg.attach(MIMEText(plain_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as server:
            server.login(smtp_user, smtp_password)
            server.sendmail(from_email, to_email, msg.as_string())
        logger.info("[SUCCESS] Email sent successfully.")
        return True
    except smtplib.SMTPAuthenticationError as e:
        logger.error(f"[ERROR] Authentication failed: Check your username and password: {e}")
        raise  # Don't retry auth failures
    except (smtplib.SMTPException, ConnectionError, TimeoutError) as e:
        logger.error(f"[ERROR] Failed to send email (will retry): {e}")
        raise  # Will be caught by retry decorator
    except Exception as e:
        logger.error(f"[ERROR] Unexpected error when sending email: {e}")
        raise  # Don't retry unexpected errors

def send_stock_drop_email(result, email_lines, toemail):
    """
    Composes and sends a styled email for stock drop alerts.
    
    :param result: List of tickers that triggered the alert
    :param email_lines: List of text blocks describing each stock alert
    :return: Boolean indicating if email was sent successfully
    """
    if not email_lines:
        logger.info("No alerts to send in email.")
        return True

    # Build plain text email (optional for fallback or logging)
    plain_body = ("\n\n" + "-" * 60 + "\n\n").join(email_lines)
    plain_body += "\n\nThis alert was generated based on your configured drop threshold."

    # Subject
    subject = f"Stock Alerts: {len(result)} stocks triggered drop alerts"

    # HTML Email Body
    html_template_start = """
    <html>
    <head>
      <style>
        body { font-family: Arial, sans-serif; color: #333; }
        .stock-block { margin-bottom: 30px; padding-bottom: 20px; border-bottom: 1px solid #ccc; }
        .stock-title { font-size: 18px; font-weight: bold; color: #1a73e8; }
        .label { font-weight: bold; }
        .drop { color: #d93025; }
        .rise { color: #188038; }
        .footer { font-size: 13px; color: #666; margin-top: 40px; }
      </style>
    </head>
    <body>
    """

    html_template_end = """
    <div class="footer">
      This alert was generated based on your configured drop threshold.
    </div>
    </body>
    </html>
    """

    html_blocks = []
    for line in email_lines:
        # naive conversion from plain text to HTML blocks
        html_line = line.replace("\n", "<br>").replace("  ", "&nbsp;&nbsp;")
        html_blocks.append(f"<div class='stock-block'>{html_line}</div>")

    html_body = html_template_start + "\n".join(html_blocks) + html_template_end

    from_email = "dummy@example.com"
    to_email = toemail
    smtp_user = "dummy@example.com"
    smtp_password = "123456789"  # Not your Gmail password, CREATE an App Password in your Google Account settings https://myaccount.google.com/apppasswords

    try:
        success = send_email_via_smtp(
            subject=subject,
            html_body=html_body,
            plain_body=plain_body,
            to_email=to_email,
            from_email=from_email,
            smtp_user=smtp_user,
            smtp_password=smtp_password,
        )
        return success
    except Exception as e:
        logger.error(f"Email sending ultimately failed after retries: {e}")
        return False

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((requests.RequestException, ConnectionError, TimeoutError)),
    reraise=True
)
def fetch_page(url, params):
    """
    Fetch a single page with retry mechanism.
    Will retry up to 5 times with exponential backoff if network errors occur.
    
    :param url: The URL to fetch
    :param params: Query parameters
    :return: JSON response
    """
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()  # Raise exception for 4XX/5XX status codes
        return response.json()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            logger.warning(f"Rate limited (429). Retrying after longer wait...")
            time.sleep(10)  # Extra wait for rate limits
            raise
        logger.error(f"HTTP error fetching page {params.get('p', 'unknown')}: {e}")
        raise
    except (requests.RequestException, ConnectionError, TimeoutError) as e:
        logger.error(f"Network error fetching page {params.get('p', 'unknown')}: {e}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON response for page {params.get('p', 'unknown')}: {e}")
        raise

def fetch_all_stocks():
    """
    Fetches all stock data from the StockAnalysis API with robust error handling.
    Handles retries for individual page fetches and gracefully continues on failures.
    
    :return: List of stock dictionaries.
    """
    base_url = "https://stockanalysis.com/api/screener/s/f"
    params = {
        "m": "s",
        "s": "asc",
        "c": "s,n,industry,price,low,close,change",
        "cn": 500,
        "i": "stocks",
        "p": 1  # Starting page
    }

    all_data = []
    max_failures = 3
    consecutive_failures = 0
    
    # Create data directory if it doesn't exist
    data_dir = "data"
    try:
        os.makedirs(data_dir, exist_ok=True)
    except Exception as e:
        logger.warning(f"Could not create data directory: {e}")
        data_dir = "."  # Fallback to current directory
    
    while True:
        try:
            logger.info(f"Fetching page {params['p']}...")
            json_data = fetch_page(base_url, params)
            
            page_data = json_data.get("data", {}).get("data", [])
            
            if not page_data:
                logger.info("No more data. Exiting.")
                break
                
            all_data.extend(page_data)
            params["p"] += 1
            consecutive_failures = 0  # Reset failure counter on success
            time.sleep(2)  # Gentle rate limiting
            
        except Exception as e:
            consecutive_failures += 1
            logger.error(f"Failed to fetch page {params['p']} after multiple retries: {e}")
            
            if consecutive_failures >= max_failures:
                logger.critical(f"Too many consecutive failures ({consecutive_failures}). Stopping data collection.")
                break
                
            params["p"] += 1  # Skip this problematic page and try the next one
    
    if all_data:
        try:
            # Save the data to a JSON file
            time_stamp = time.strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(data_dir, f"stocks_data_{time_stamp}.json")
            
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(all_data, f, indent=4)
            logger.info(f"Successfully saved {len(all_data)} stocks to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save data to file: {e}")
    else:
        logger.warning("No stock data was collected!")

    return all_data


def check_loss_tickers(stock_data, threshold=1):
    """
    Returns tickers of stocks that dropped more than `threshold` percent from previous close
    either intraday (low) or current price.

    :param stock_data: List of stock dictionaries with keys 's', 'price', 'low', 'close'
    :param threshold: Drop percent to filter, default 1
    :return: Tuple of (list of ticker symbols, list of email message lines)
    """
    email_lines = []
    result = []
    
    if not stock_data:
        logger.warning("No stock data provided to check_loss_tickers!")
        return result, email_lines

    processed_count = 0
    invalid_count = 0

    for stock in stock_data:
        price = stock.get("price")
        low = stock.get("low")
        close = stock.get("close")
        stock_name = stock.get("n")
        ticker_symbol = stock.get("s")

        if None in (price, low, close, ticker_symbol) or close == 0:
            invalid_count += 1
            continue  # skip invalid data

        try:
            drop_from_low = ((close - low) / close) * 100
            price_status = ((price - close) / close) * 100  # Signed: positive = gain, negative = loss

            if drop_from_low >= threshold or price_status <= -threshold:
                line = (
                    f"Stock Drop Alert: {stock_name} ({ticker_symbol})\n"
                    f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"Ticker: {ticker_symbol}\n\n"
                    f"Previous Day's Close: ${close:.2f}\n"
                    f"Current Price:        ${price:.2f}\n"
                    f"Intraday Low Drop:    {drop_from_low:.2f}% (from yesterday's close to today's lowest price)\n"
                    f"Current Stock Status: {price_status:+.2f}% (from yesterday's close to current price)"
                )

                email_lines.append(line)
                result.append(ticker_symbol)
            
            processed_count += 1
            
        except (ZeroDivisionError, TypeError) as e:
            logger.error(f"Error calculating drop for {ticker_symbol}: {e}")
            invalid_count += 1

    logger.info(f"Processed {processed_count} stocks, found {len(result)} alerts, skipped {invalid_count} invalid entries")
    return result, email_lines


def load_from_file(file_path):
    """
    Attempt to load stock data from a previously saved JSON file.
    Useful as a fallback if API fetching fails.
    
    :param file_path: Path to the JSON file
    :return: List of stock dictionaries or None if loading fails
    """
    try:
        with open(file_path, 'r', encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Successfully loaded {len(data)} stocks from {file_path}")
        return data
    except (FileNotFoundError, json.JSONDecodeError, IOError) as e:
        logger.error(f"Failed to load data from {file_path}: {e}")
        return None

def find_latest_data_file():
    """
    Find the most recently created stocks data JSON file
    
    :return: Path to the latest file or None if no files found
    """
    # Try in data directory first, then fallback to current directory
    search_dirs = ["data", "."]
    
    for directory in search_dirs:
        try:
            if not os.path.exists(directory):
                continue
                
            files = [os.path.join(directory, f) for f in os.listdir(directory) 
                     if f.startswith('stocks_data_') and f.endswith('.json')]
            
            if files:
                # Sort by creation time, newest first
                files.sort(key=lambda x: os.path.getctime(x), reverse=True)
                logger.info(f"Found latest data file: {files[0]}")
                return files[0]
                
        except Exception as e:
            logger.error(f"Error finding latest data file in {directory}: {e}")
    
    logger.warning("No data files found in any directory")
    return None

def main(threshold=50):
    """
    Main execution function with error handling and fallbacks
    
    :param threshold: The threshold percentage for stock drops
    :return: True if execution completed successfully, False otherwise
    """
    # Create a lock file to prevent multiple instances from running
    lock_file = "stock_alerts.lock"
    
    if os.path.exists(lock_file):
        # Check if the lock file is stale (older than 2 hours)
        lock_age = time.time() - os.path.getctime(lock_file)
        if lock_age < 7200:  # 2 hours in seconds
            logger.warning(f"Lock file exists and is recent ({lock_age/60:.1f} minutes old). Another instance may be running. Exiting.")
            return False
        else:
            logger.warning(f"Found stale lock file (age: {lock_age/60:.1f} minutes). Removing it.")
            try:
                os.remove(lock_file)
            except Exception as e:
                logger.error(f"Could not remove stale lock file: {e}")
                return False
    
    # Create the lock file
    try:
        with open(lock_file, 'w') as f:
            f.write(f"Process started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception as e:
        logger.error(f"Could not create lock file: {e}")
        # Continue anyway since this is not critical
    
    try:
        logger.info("Starting stock alerts job")
        
        # Try to fetch fresh data
        all_stocks_data = fetch_all_stocks()
        
        # If no data was fetched, try to load from the most recent file
        if not all_stocks_data:
            logger.warning("No data fetched from API, trying to load from latest file")
            latest_file = find_latest_data_file()
            
            if latest_file:
                all_stocks_data = load_from_file(latest_file)
            
        # If we still don't have data, we can't proceed
        if not all_stocks_data:
            logger.error("Failed to obtain stock data from API or file. Exiting.")
            return False
            
        # Process the data and prepare alerts
        result, email_lines = check_loss_tickers(all_stocks_data, threshold=threshold)
        
        # Only send email if we have alerts
        if result:
            logger.info(f"Found {len(result)} stocks with significant drops, sending email alert.")
            email_success = send_stock_drop_email(result, email_lines, "kashiftariq7654@gmail.com") #mushafmughal12@gmail.com

            if not email_success:
                logger.error("Failed to send email alert after multiple attempts.")
                return False
        else:
            logger.info(f"No stocks found with drops exceeding {threshold}% threshold.")
            
        logger.info("Stock alerts job completed successfully")
        return True
        
    except Exception as e:
        logger.critical(f"Unexpected error in main execution: {e}", exc_info=True)
        return False
    finally:
        # Always remove the lock file at the end, regardless of success or failure
        try:
            if os.path.exists(lock_file):
                os.remove(lock_file)
        except Exception as e:
            logger.error(f"Failed to remove lock file: {e}")
            # Continue anyway as the process is ending

if __name__ == "__main__":
    try:
        success = main(threshold=50)
        exit_code = 0 if success else 1
        exit(exit_code)  # Return proper exit code for scheduler
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        exit(130)  # Standard exit code for Ctrl+C
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        exit(1)  # Non-zero exit code indicates failure