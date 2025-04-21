import requests

url = 'http://localhost:8000/upload-file'
file_path = 'generated_eeg_data.csv'

with open(file_path, 'rb') as f:
    files = {'file': ('generated_eeg_data.csv', f, 'text/csv')}

    response = requests.post(url, files=files)

print(response.status_code)
print(response.json())
