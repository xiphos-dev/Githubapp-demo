import os
import requests
from flask import request, jsonify
from github import Github
from dotenv import load_dotenv
import base64
import json
import pprint

class Menu:
    def __init__(self, options):
        """
        Initializes the Menu with a list of options.
        
        :param options: List of strings representing the menu options.
        """
        self.options = options
        
    def get_commits(self, owner, repo_name):
        """
        Fetches the commits from the specified GitHub repository.
        
        :param owner: The owner of the repository.
        :param repo_name: The name of the repository.
        :return: A list of commit messages.
        """
        url = f"https://api.github.com/repos/{owner}/{repo_name}/commits"
        url = "https://api.github.com/repos/xiphos-dev/Githubapp-demo/commits"
        token = os.environ.get("GITHUB_TOKEN")
        try:
            headers = {"Accept": "application/vnd.github+json", "Authorization": f"Bearer {token}", "X-GitHub-Api-Version": "2022-11-28"}
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            commits = response.json()
            if commits:
                
                #print(commits[0].keys())
                most_recent_commit_sha = commits[0]["sha"]
                most_recent_commit_url = commits[1]["url"]
                
                commit_response = requests.get(most_recent_commit_url, headers=headers)
                commit_response.raise_for_status()
                print(commit_response)
                commit_details = commit_response.json()
                
                files = commit_details.get('files', [])
                file_details = [{"filename": file['filename'], "sha": file['sha']} for file in files]
                '''
                
                return {
                    "commit_message": commits[0]['commit']['message'],
                    "commit_sha": most_recent_commit_sha,
                    "files": file_details
                }
            else:
                return [commit['commit']['message'] for commit in commits]
                
                '''
                content = []
                for file in file_details:
                    content.append(self.get_file_content_by_sha("xiphos-dev","Githubapp-demo",file["sha"],token))
                return content
        except requests.RequestException as e:
            print(f"An error occurred: {e}")
            return []
    
    def get_issues(self, owner, repo_name):
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {os.environ.get('GITHUB_TOKEN')}",
            "X-GitHub-Api-Version": "2022-11-28"
        }
        
        url = f" https://api.github.com/repos/{owner}/{repo_name}/issues"
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            content = response.json()
            if len(content) == 0:
                return "No issues found"
            try:
                #print(content[0])
                print(content[0].keys())
                formatted_content = content[0]["comments_url"]
                #json_content = json.loads(content[0])
                #formatted_content = json.dumps(json_content, indent=4)
            except json.JSONDecodeError:
                formatted_content = content[0]
                #formatted_content = repr(content)
                #print("Pretty content:"+formatted_content)
            #print("Content by sha:"+content)
            return formatted_content
        except requests.RequestException as e:
            return f"An error occurred: {e}"
        
    def get_branches(self, owner, repo_name):
        
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {os.environ.get('GITHUB_TOKEN')}",
            "X-GitHub-Api-Version": "2022-11-28"
        }
        
        url = f" https://api.github.com/repos/{owner}/{repo_name}/branches"
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            branches = response.json()

            return [{"name":branch["name"], "sha": branch["commit"]["sha"]} for branch in branches]
        except requests.RequestException as e:
            return f"An error occurred: {e}"
      
    def get_file_content_by_sha(self, owner, repo_name, sha, token):
        """
        Fetches the content of a file by its SHA from the specified GitHub repository.
        
        :param owner: The owner of the repository.
        :param repo_name: The name of the repository.
        :param sha: The SHA of the file blob.
        :param token: GitHub personal access token for authentication.
        :return: The content of the file.
        """
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {os.environ.get('GITHUB_TOKEN')}",
            "X-GitHub-Api-Version": "2022-11-28"
        }
        
        url = f"https://api.github.com/repos/{owner}/{repo_name}/git/blobs/{sha}"
        #url = f"https://api.github.com/repos/xiphos-dev/Githubapp-demo/git/blobs/{sha}"
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            blob = response.json()
            content = base64.b64decode(blob['content']).decode('utf-8')
            # Pretty print JSON files
            
            try:
                json_content = json.loads(content)
                formatted_content = json.dumps(json_content, indent=4)
            except json.JSONDecodeError:
                formatted_content = content
                #formatted_content = repr(content)
                #print("Pretty content:"+formatted_content)
            #print("Content by sha:"+content)
            return formatted_content
        except requests.RequestException as e:
            return f"An error occurred: {e}"
    
    def get_filepath_content(self,owner,repo_name,filepath):
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {os.environ.get('GITHUB_TOKEN')}",
            "X-GitHub-Api-Version": "2022-11-28"
        }
        
        url = f"https://api.github.com/repos/{owner}/{repo_name}/contents/{filepath}"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        file_data = response.json()
        
        return self.get_file_content_by_sha(owner,repo_name,file_data["sha"],"")
    
    def display_menu(self):
        """
        Displays the menu options to the user.
        """
        print("Please select an option:")
        for i, option in enumerate(self.options, 1):
            print(f"{i}. {option}")
    
    def get_user_choice(self):
        """
        Prompts the user to select an option from the menu.
        
        :return: The user's choice as an integer.
        """
        while True:
            try:
                choice = int(input("Enter the number of your choice: "))
                if 1 <= choice <= len(self.options):
                    return choice
                else:
                    print(f"Please enter a number between 1 and {len(self.options)}.")
            except ValueError:
                print("Invalid input. Please enter a valid integer.")
    
    def run(self):
        """
        Runs the menu selection process.
        
        :return: The user's choice (1-indexed).
        """
        self.display_menu()
        return self.get_user_choice()

if __name__ == "__main__":
    load_dotenv()

    # Define the list of options
    options = ["Fetch Commits", "Branches", "File content", "Issues", "Exit"]
    
    # Initialize the Menu class with the list of options
    menu = Menu(options)
    
    while True:
        # Run the menu and get the user's choice
        choice = menu.run()
        
        if choice == 1:
            # Get the repository owner and name from the user
            #owner = input("Enter the repository owner: ")
            #repo_name = input("Enter the repository name: ")
            owner = "xiphos-dev"
            repo_name = "Githubapp-demo"
            
            # Fetch and display the commits
            commits = menu.get_commits(owner, repo_name)
            if isinstance(commits, dict):
                print(f"Most recent commit message: {commits['commit_message']}")
                print(f"Commit SHA: {commits['commit_sha']}")
                print("Modified files:")
                for file in commits['files']:
                    print(f"- {file['filename']}: {file['sha']}")
            else:
                for content in commits:
                    print(repr(content))

        elif choice == 2:
            owner = "xiphos-dev"
            repo_name = "Githubapp-demo"
            
            # Fetch and display the commits
            branches = menu.get_branches(owner, repo_name)
            if isinstance(branches, list):
                for branch in branches:
                    print(f"Name: {branch['name']}")
                    print(f"Commit SHA: {branch['sha']}")
        
        elif choice == 3:
            owner = "xiphos-dev"
            repo_name = "Githubapp-demo"
            path="handler.py"
            
            # Fetch and display the commits
            file_blob = menu.get_filepath_content(owner, repo_name, path)
            print(file_blob)
                    

        elif choice == 4:
            owner = "qgis"
            repo_name = "QGIS"
            
            # Fetch and display the commits
            most_recent_issue = menu.get_issues(owner, repo_name)
            print(most_recent_issue)
            
        elif choice == len(options):
            print("Exiting...")
            break