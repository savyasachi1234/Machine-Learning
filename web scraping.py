# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 19:37:16 2025

@author: ADMIN
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd

# Fetch the webpage
response = requests.get('https://books.toscrape.com/')
html_content = response.text

if html_content:
    soup = BeautifulSoup(html_content, 'html.parser')
    books = soup.find_all('article', class_='product_pod')

    # Extract book data
    data = []
    for book in books:
        title = book.find('h3').find('a')['title']
        price = book.find('p', class_='price_color').text
        data.append([title, price])  # Append to list

    # Create a DataFrame
    df = pd.DataFrame(data, columns=['Title', 'Price'])
    print(df)

    # Save to CSV
    df.to_csv('books_data.csv', index=False)
    print("Data saved to books_data.csv")
else:
    print("Failed to fetch the webpage.")
