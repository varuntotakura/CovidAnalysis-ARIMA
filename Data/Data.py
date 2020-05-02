import requests
import pandas as pd
import numpy as np
url = "https://api.covid19india.org/csv/latest/state_wise_daily.csv"
#      https://api.covid19india.org/csv/latest/case_time_series.csv

req = requests.get(url)
url_content = req.content
csv_file = open('downloaded.csv', 'wb')

csv_file.write(url_content)
csv_file.close()
