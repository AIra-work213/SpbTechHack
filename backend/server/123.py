import requests

response = requests.post("http://localhost:8000/query", json={"query": "Как стать крутым человеком?"})
print(response.json())