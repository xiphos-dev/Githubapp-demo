from pysmee import Client
from dotenv import load_dotenv


load_dotenv()

# Replace with your actual webhook proxy URL and local server URL
webhook_proxy_url = "WEBHOOK_PROXY_URL"
local_server_url = "http://localhost:3000/api/webhook"

# Create a pysmee client
client = Client(webhook_proxy_url)

# Start the proxy
client.start_proxy(local_server_url)