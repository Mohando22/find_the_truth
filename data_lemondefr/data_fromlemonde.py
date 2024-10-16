import time
import csv
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

# Set up Selenium WebDriver
options = webdriver.ChromeOptions()
options.add_argument("--headless")  # Run in headless mode (without opening a browser window)
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

# Initialize the browser (automatically downloads the appropriate ChromeDriver if not already installed)
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

url = "https://www.lemonde.fr/politique/"
driver.get(url)

scroll_pause_time = 2  
last_height = driver.execute_script("return document.body.scrollHeight")

articles = []

try:
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(scroll_pause_time)


        soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        for article in soup.find_all("section", class_="teaser"):
            title_element = article.find("h3", class_="teaser__title")
            date_element = article.find("span", class_="meta__date")
            
            if title_element and date_element:
                title = title_element.get_text(strip=True)
                publication_date = date_element.get_text(strip=True)
                
                current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                articles.append({
                    "title": title,
                    "publication_date": publication_date,
                    "scrape_date": current_date
                })
        
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break 
        last_height = new_height

finally:
    driver.quit()

# Save the collected articles in a CSV file
with open("lemonde_articles.csv", "w", encoding="utf-8", newline="") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=["title", "publication_date", "scrape_date"])
    writer.writeheader()
    writer.writerows(articles)

print("Articles with dates and scrape dates have been saved to 'lemonde_articles.csv'")