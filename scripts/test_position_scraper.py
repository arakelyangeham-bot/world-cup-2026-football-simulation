from sofascore_utils import BASE_URL, get_json

player_id = 839956  # Haaland example

url = f"{BASE_URL}/player/{player_id}"
data = get_json(url)

print(data.keys())
print(data)