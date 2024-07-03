import os
import jwt
import logging
import hmac
from flask import Flask, request, jsonify
from github import Github, GithubIntegration
from github import Auth
from dotenv import load_dotenv
import pysmee
import requests
import time

load_dotenv()

PRIVATE_KEY_PATH = os.environ.get("PRIVATE_KEY_PATH")
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET")
APP_ID = os.environ.get("APP_ID")
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')

with open(PRIVATE_KEY_PATH, 'r', encoding='utf-8') as key_file:
    private_key = key_file.read()

# Print the private key to verify
#print(private_key)

def create_jwt(app_id, private_key):
    payload = {
        # issued at time
        'iat': int(time.time()),
        # JWT expiration time (10 minute maximum)
        'exp': int(time.time()) + (10 * 60),
        # GitHub App's identifier
        'iss': app_id
    }
    jwt_token = jwt.encode(payload, private_key, algorithm='RS256')
    return jwt_token

jwt_token = create_jwt(APP_ID, private_key)
github_integration = GithubIntegration(APP_ID, private_key)

message_for_new_prs  = "Thanks for opening a new PR! Please follow our contributing guidelines to make your PR easier to review."

g = Github(GITHUB_TOKEN)

app = Flask(__name__)

@app.route("/webhook", methods=["POST"])
def webhook():
    event = request.headers.get('X-GitHub-Event')
    signature = request.headers.get('X-Hub-Signature-256')
    payload = request.get_data()

    if not is_valid_signature(payload, signature):
        return jsonify({"message": "Invalid signature"}), 400

    if event == "pull_request" and request.json['action'] == "opened":
        handle_pull_request_opened(request.json)

    return jsonify({"message": "Webhook received"}), 200

def is_valid_signature(payload, signature):
    secret = bytes(webhook_secret, 'utf-8')
    hash_value = hmac.new(secret, payload, hashlib.sha256).hexdigest()
    expected_signature = f"sha256={hash_value}"
    return hmac.compare_digest(expected_signature, signature)

def handle_pull_request_opened(payload):
    pr_number = payload['pull_request']['number']
    print(f"Received a pull request event for #{pr_number}")

    try:
        repo = g.get_repo(f"{payload['repository']['owner']['login']}/{payload['repository']['name']}")
        issue = repo.get_issue(number=pr_number)
        issue.create_comment("Thank you for your pull request! We will review it soon.")
    except Exception as error:
        print(f"Error! {error}")

if __name__ == "__main__":
    app.run(port=3000)