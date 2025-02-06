import requests

dataset_url = "https://www.data.gouv.fr/api/1/datasets/accidents-de-la-route/"

response = requests.get(dataset_url)

if response.status_code == 200:
    data = response.json()
    resources = data.get("resources", [])
    
    if not resources:
        print("❌ No resources found in this dataset.")
    else:
        print("✅ Found resources:")
        for resource in resources:
            print(f"📂 File: {resource.get('title', 'No title')}\n🔗 URL: {resource.get('url', 'No URL')}\n")
else:
    print(f"❌ API request failed! Status code: {response.status_code}")


