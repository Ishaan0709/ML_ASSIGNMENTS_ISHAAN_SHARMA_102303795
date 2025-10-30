import requests
from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time

# ==========================================================
# Task 1: Books to Scrape
# ==========================================================
print("Starting data extraction: Books to Scrape")

base_url = "https://books.toscrape.com/catalogue/page-{}.html"
book_records = []

for page_num in range(1, 51):
    response = requests.get(base_url.format(page_num))
    if response.status_code != 200:
        break
    soup = BeautifulSoup(response.text, "html.parser")
    books = soup.select("article.product_pod")
    if not books:
        break
    for b in books:
        title = b.h3.a["title"]
        price = b.select_one("p.price_color").text.strip()
        stock = b.select_one("p.instock.availability").text.strip()
        rating = b.p.get("class")[1]
        book_records.append([title, price, stock, rating])

books_df = pd.DataFrame(book_records, columns=["Title", "Price", "Availability", "Star Rating"])
books_df.to_csv("books.csv", index=False)
print(f"Books dataset exported ({len(books_df)} records) → books.csv\n")


# ==========================================================
# Task 2: IMDb Top 250
# ==========================================================
print("Starting data extraction: IMDb Top 250 Movies")

chrome_opts = Options()
chrome_opts.add_argument("--headless")
chrome_opts.add_argument("--disable-gpu")
chrome_opts.add_argument("--no-sandbox")

from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_opts)

driver.get("https://www.imdb.com/chart/top/")
time.sleep(3)

movie_entries = driver.find_elements("css selector", "tbody.lister-list tr")
movie_records = []

for entry in movie_entries:
    rank = entry.find_element("css selector", ".titleColumn").text.split('.')[0]
    title = entry.find_element("css selector", ".titleColumn a").text
    year = entry.find_element("css selector", ".secondaryInfo").text.strip("()")
    rating = entry.find_element("css selector", "td.ratingColumn.imdbRating strong").text
    movie_records.append([rank, title, year, rating])

driver.quit()

movies_df = pd.DataFrame(movie_records, columns=["Rank", "Movie Title", "Year", "IMDB Rating"])
movies_df.to_csv("imdb_top250.csv", index=False)
print(f"IMDb dataset exported ({len(movies_df)} records) → imdb_top250.csv\n")


# ==========================================================
# Task 3: Global Weather Data
# ==========================================================
print("Starting data extraction: Time and Date Weather")

weather_url = "https://www.timeanddate.com/weather/"
response = requests.get(weather_url)
soup = BeautifulSoup(response.text, "html.parser")

rows = soup.select("table#wt-48 tbody tr")
weather_records = []

for row in rows:
    city = row.find("a").text.strip()
    temperature = row.select_one("td.rbi").text.strip()
    condition = row.select("td")[2].text.strip()
    weather_records.append([city, temperature, condition])

weather_df = pd.DataFrame(weather_records, columns=["City", "Temperature", "Condition"])
weather_df.to_csv("weather.csv", index=False)
print(f"Weather dataset exported ({len(weather_df)} records) → weather.csv\n")

print("Data extraction completed for all tasks.")
