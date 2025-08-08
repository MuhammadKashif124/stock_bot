import requests
import json
import time
import os
import logging
import sys
import pandas as pd
import smtplib
import re
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
openai_api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)


def generate_alert_openai(stock_list):
    prompt = """
    You are an expert formatting assistant that generates professional email content for stock drop alerts.

    You will be given a list of stocks that have either dropped more than 50% intraday or are currently down more than 50% from the previous close.

    Each stock alert includes the following fields:
    - company_name
    - ticker
    - previous_close
    - current_price
    - drop_from_low (percentage from yesterday's close to today's lowest price)
    - price_status (percentage from yesterday's close to current price)
    - alert_time (timestamp)

    Format a plain text summary for **each stock alert** in a clear, readable structure like this:

    Stock Drop Alert: {company_name} ({ticker})
    Date: {alert_time}
    Ticker: {ticker}

    PDC: $ {previous_close:.2f}
    CP:  $ {current_price:.2f}
    ILD: {drop_from_low:.2f}%
    CSS: {price_status:+.2f}%

    Only return the formatted stock alert strings for each stock. Do not add explanation or extra text.
    """
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": prompt
            },
            {"role": "user", "content": f"Here are the stocks: {json.dumps(stock_list)}"}

        ]
    )

    output = completion.choices[0].message.content
    return output

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
    bcc_emails = ["mushafmughal99@gmail.com"]
    all_recipients = [to_email] + bcc_emails

    # Attach both plain text and HTML (Gmail prefers HTML but fallback matters)
    msg.attach(MIMEText(plain_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as server:
            server.login(smtp_user, smtp_password)
            server.sendmail(from_email, all_recipients, msg.as_string())
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

def htmlify_stock_block(text):
    def colorize(line):
        line = re.sub(r"(-\d+\.\d+%)", r"<span class='drop'>\1</span>", line)
        line = re.sub(r"(\+\d+\.\d+%)", r"<span class='rise'>\1</span>", line)
        return line

    return "<pre>" + "\n".join(colorize(line) for line in text.splitlines()) + "</pre>"

def send_no_alert_email(toemail):
    """
    Sends an email when no stock alerts were triggered.
    
    :param toemail: Email address to send the alert to
    :return: Boolean indicating if email was sent successfully
    """
    logger.info("Sending 'No alert today' email.")
    
    # Subject
    subject = "Stock Alerts: No alerts today"
    
    # Plain text body
    plain_body = "No stocks triggered drop alerts today based on your configured threshold."
    plain_body += "\n\nYour stock monitoring system is running normally."
    
    # HTML Email Body
    html_body = """
    <html>
    <head>
      <style>
        body { font-family: Arial, sans-serif; color: #333; padding: 20px; }
        .message { margin: 30px 0; font-size: 16px; line-height: 1.6; }
        .footer { font-size: 13px; color: #666; margin-top: 40px; }
      </style>
    </head>
    <body>
      <div class="message">
        <p>No stocks triggered drop alerts today based on your configured threshold.</p>
        <p>Your stock monitoring system is running normally.</p>
      </div>
      <div class="footer">
        <p>This is an automated message from your stock monitoring system.</p>
      </div>
    </body>
    </html>
    """
    
    from_email = "loudhome12@gmail.com"
    to_email = toemail
    smtp_user = "loudhome12@gmail.com"
    smtp_password = "qgmv rstg tewl zpjk"  # App Password
    
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
        logger.error(f"'No alert' email sending failed: {e}")
        return False

def send_stock_drop_email(result, stock_list, toemail):
    """
    Composes and sends a styled email for stock drop alerts.
    
    :param result: List of tickers that triggered the alert
    :param stock_list: List of text blocks describing each stock alert
    :return: Boolean indicating if email was sent successfully
    """
    if not stock_list:
        logger.info("No alerts to send in email.")
        return True
    
    # Build plain text email (optional for fallback or logging)
    plain_body = ("\n\n" + "-" * 60 + "\n\n").join(stock_list)
    plain_body += "\n\nThis alert was generated based on your configured drop threshold."
    plain_body += "\n\nAcronym Glossary:"
    plain_body += "\nPDC = Previous Day's Close"
    plain_body += "\nCP = Current Price"
    plain_body += "\nILD = Intraday Low Drop (from yesterday's close to today's lowest price)"
    plain_body += "\nCSS = Current Stock Status (from yesterday's close to current price)"

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
        .glossary { margin-top: 20px; background-color: #f8f9fa; padding: 15px; border-radius: 5px; }
        .glossary h3 { margin-top: 0; color: #1a73e8; }
        .glossary p { margin: 5px 0; }
      </style>
    </head>
    <body>
    """

    html_template_end = """
    <div class="footer">
      <p>This alert was generated based on your configured drop threshold.</p>
      <div class="glossary">
        <h3>Acronym Glossary:</h3>
        <p>PDC = Previous Day's Close</p>
        <p>CP = Current Price</p>
        <p>ILD = Intraday Low Drop (from yesterday's close to today's lowest price)</p>
        <p>CSS = Current Stock Status (from yesterday's close to current price)</p>
      </div>
    </div>
    </body>
    </html>
    """

    html_blocks = []
    for stock in stock_list:
        # OpenAI now directly generates the acronym format, no conversion needed
        # Convert to HTML with colorization
        html_line = htmlify_stock_block(stock)
        html_blocks.append(f"<div class='stock-block'>{html_line}</div>")

    html_body = html_template_start + "\n".join(html_blocks) + html_template_end

    from_email = "loudhome12@gmail.com"
    to_email = toemail
    smtp_user = "loudhome12@gmail.com"
    smtp_password = "qgmv rstg tewl zpjk"  # Not your Gmail password, CREATE an App Password in your Google Account settings https://myaccount.google.com/apppasswords

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


def check_loss_tickers(stock_data, threshold=50):
    """
    Returns tickers of stocks that dropped more than `threshold` percent from previous close
    either intraday (low) or current price.

    :param stock_data: List of stock dictionaries with keys 's', 'price', 'low', 'close'
    :param threshold: Drop percent to filter, default 1
    :return: Tuple of (list of ticker symbols, list of email message lines)
    """
    stock_list = []
    result = []
    
    if not stock_data:
        logger.warning("No stock data provided to check_loss_tickers!")
        return result, stock_list

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

            if price_status <= -threshold or (drop_from_low >= threshold and price_status <= -threshold):

                alert = {
                    "stock_name": stock_name,
                    "ticker_symbol": ticker_symbol,
                    "alert_time": time.strftime('%Y-%m-%d %H:%M:%S'),
                    "previous_close": close,
                    "current_price": price,
                    "drop_from_low": drop_from_low,
                    "price_status": price_status,
                }

                stock_list.append(alert)
                result.append(ticker_symbol)
            
            processed_count += 1
            
        except (ZeroDivisionError, TypeError) as e:
            logger.error(f"Error calculating drop for {ticker_symbol}: {e}")
            invalid_count += 1

    logger.info(f"Processed {processed_count} stocks, found {len(result)} alerts, skipped {invalid_count} invalid entries")
    return result, stock_list


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
    # Check if we should use locking (default is True, but can be disabled via env var)
    use_locking = os.environ.get("USE_LOCK_FILE", "True").lower() in ["true", "1", "yes"]
    lock_file = "stock_alerts.lock"
    
    if use_locking and os.path.exists(lock_file):
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
    
    # Create the lock file if locking is enabled
    if use_locking:
        try:
            with open(lock_file, 'w') as f:
                f.write(f"Process started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        except Exception as e:
            logger.error(f"Could not create lock file: {e}")
            # Continue anyway since this is not critical
    
    try:
        logger.info("Starting stock alerts job")
        
        # Set recipient email address
        recipient_email = "loudhome12@gmail.com"
        
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
        result, stocks_list = check_loss_tickers(all_stocks_data, threshold=threshold)
        
        # Send appropriate email based on results
        if result:
            logger.info(f"Found {len(result)} stocks with significant drops, sending email alert.")
            email_lines = generate_alert_openai(stocks_list)
            email_lines = re.split(r'\n(?=Stock Drop Alert:)', email_lines.strip())
            email_success = send_stock_drop_email(result, email_lines, recipient_email)

            if not email_success:
                logger.error("Failed to send email alert after multiple attempts.")
                return False
        else:
            logger.info(f"No stocks found with drops exceeding {threshold}% threshold. Sending 'no alert' email.")
            email_success = send_no_alert_email(recipient_email)
            
            if not email_success:
                logger.error("Failed to send 'no alert' email after multiple attempts.")
                return False
            
        logger.info("Stock alerts job completed successfully")
        return True
        
    except Exception as e:
        logger.critical(f"Unexpected error in main execution: {e}", exc_info=True)
        return False
    finally:
        # Always remove the lock file at the end, regardless of success or failure
        if use_locking:
            try:
                if os.path.exists(lock_file):
                    os.remove(lock_file)
            except Exception as e:
                logger.error(f"Failed to remove lock file: {e}")
                # Continue anyway as the process is ending

if __name__ == "__main__":
    try:
        # Main scheduling loop
        while True:
            logger.info("Starting scheduled stock alert check...")
            start_time = time.time()
            
            success = main(threshold=100)
            if not success:
                logger.warning("Stock alert check completed with errors")
            else:
                logger.info("Stock alert check completed successfully")
            
            # Calculate time to sleep (24 hours from start of current run)
            execution_time = time.time() - start_time
            sleep_time = max(0, 24*60*60 - execution_time)  # 24 hours in seconds
            
            logger.info(f"Next check scheduled in {sleep_time/3600:.2f} hours")
            time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        exit(130)  # Standard exit code for Ctrl+C
    except Exception as e:
        logger.critical(f"Fatal error in scheduling loop: {e}", exc_info=True)
        exit(1)  # Non-zero exit code indicates failure