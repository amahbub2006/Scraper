Movie Finder
This project is a movie finder application that utilizes data from an Excel sheet to help users search and filter movies based on various attributes. The application is designed to provide an easy-to-use interface for finding movies by title, director, genre, and other relevant details.

Features
Search Movies: Find movies by title or other attributes.
Filter Results: Narrow down results based on genres, release years, and more.
Detailed Movie Information: View detailed information about each movie, including director, genre, and release date.
Prerequisites
Python 3.x
pandas library
openpyxl library (for Excel file handling)
Installation
Clone the Repository:

sh
Copy code
git clone <your-repository-url>
cd movie_finder
Create and Activate a Virtual Environment:

sh
Copy code
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
Install Dependencies:

sh
Copy code
pip install pandas openpyxl
Prepare Your Excel File:

Ensure your Excel file (movies.xlsx) is formatted with columns such as id, title, director, genre, and release_year.

Usage
Run the Application:

sh
Copy code
python movie_finder.py
Follow the Prompts:

Enter search criteria to find movies.
Filter results based on available attributes.
Example
Here’s an example of how to use the application:

python
Copy code
import pandas as pd

# Load movie data from Excel
df = pd.read_excel('movies.xlsx')

# Example: Search for movies with the title 'Inception'
search_title = 'Inception'
result = df[df['title'].str.contains(search_title, case=False, na=False)]
print(result)
