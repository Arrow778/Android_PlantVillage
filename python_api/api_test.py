

import requests




# 配置
API_KEY = "a049ac887afb4624aad5e65e8832c295"
LOCATION = ""  # 北京的 location ID（可通过城市搜索接口查）
API_HOSTS = "kw38kwd3q2.re.qweatherapi.com"

city:str = input("请输入城市名称：")
FIND_LOCATION_ID_URL = f'https://{API_HOSTS}/geo/v2/city/lookup?location={city}'
# 获取实时天气
params_location = {
    "key": API_KEY,
    "location": LOCATION
}
params_weather = {
    "key": API_KEY
}
response1 = requests.get(FIND_LOCATION_ID_URL, params=params_location)
location_data:dict = response1.json()
citys =[]
for item in location_data['location']:
    citys.append(f"{item['name']}-{item['id']}")

for x in citys:
    print(x)
print(citys[1])
BASE_URL = f'https://{API_HOSTS}/v7/weather/3d?location={citys[1][3:]}'
response = requests.get(BASE_URL, params=params_weather)
data = response.json()
print(data)
