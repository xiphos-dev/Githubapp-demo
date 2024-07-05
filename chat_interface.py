import os
import requests
from flask import request, jsonify
from github import Github
from dotenv import load_dotenv
import base64
import json
import pprint

class Menu:
    def __init__(self, options, github_access_token, github_repo_owner, github_repo_name):
        """
        Initializes the Menu with a list of options and GitHub repository details.
        
        :param options: List of strings representing the menu options.
        :param github_access_token: GitHub personal access token.
        :param github_repo_owner: GitHub repository owner's username.
        :param github_repo_name: GitHub repository name.
        """
        self.options = options
        self.github_access_token = github_access_token
        self.github_repo_owner = github_repo_owner
        self.github_repo_name = github_repo_name
        self.headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.github_access_token}",
            "X-GitHub-Api-Version": "2022-11-28"
        }
        
    def get_commits(self, owner="", repo_name="", branch=""):
        """
        Fetches the commits from the specified GitHub repository.
        
        :param owner: The owner of the repository.
        :param repo_name: The name of the repository.
        :return: A list of commit messages.
        """
        if owner == "" or repo_name == "":
            url = f"https://api.github.com/repos/{self.github_repo_owner}/{self.github_repo_name}/commits"
        else:
            url = f"https://api.github.com/repos/{owner}/{repo_name}/commits"
            
        if branch != "":
            params = {"sha": branch}
        #url = "https://api.github.com/repos/xiphos-dev/Githubapp-demo/commits"
        print(f"URL:{url}")
        try:
            if branch != "":
                response = requests.get(url, headers=self.headers, params=params)
            else:
                response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            commits = response.json()
            if commits:
                if isinstance(commits,list): 
                   # most_recent_commit_sha = commits[0]["sha"]
                  #  most_recent_commit_url = commits[0]["url"]
                    for num, commit in enumerate(commits[:5]):
                        print(f"Commit #{num+1}" + "*"*100)
                        print(f"Author: {commit['commit']['author']['name']}")
                        print(f"Date: {commit['commit']['author']['date']}")
                        print(f"Commit message: {commit['commit']['message']}")
                        print(f"Commit SHA: {commit['sha']}")

                    
                    while True:
                        try:
                            choice = int(input("Please enter a commit number or 0 to return"))
                            if choice == 0:
                                break
                            elif isinstance(choice, int) and choice <= len(commits):
                                modified_files = self.get_modified_files_for_commit(commits[choice-1]["sha"], owner, repo_name)
                                print(f"Modified files for commit '{commits[choice-1]['sha']}':")
                                for num, file in enumerate(modified_files):
                                    print(f"File #{num+1}")
                                    print(f"{file['filename']}")
                                    
                                choice = int(input("Please enter a file number or 0 to return"))
                                if choice == 0:
                                    break
                                
                                elif isinstance(choice, int) and choice <= len(modified_files):
                                    print("Fetching commit")
                                    print(self.get_file_content_by_sha(modified_files[choice-1]["sha"], owner, repo_name))
                                
                                else:
                                    print("Number does not correspond to a file")
                                    break
                                break
                            else:
                                print(f"Please enter a branch number.")
                        except ValueError:
                            print("Invalid input. Please enter a valid integer.")
                    
                    '''
                    commit_response = requests.get(most_recent_commit_url, headers=self.headers)
                    commit_response.raise_for_status()
                    print(commit_response)
                    commit_details = commit_response.json()
                    
                    files = commit_details.get('files', [])
                    file_details = [{"filename": file['filename'], "sha": file['sha']} for file in files]

                    content = []
                    for file in file_details:
                        content.append(self.get_file_content_by_sha(file["sha"], owner, repo_name))
                    '''
                    return 
        except requests.RequestException as e:
            print(f"An error occurred: {e}")
            return []
    
    
    def get_modified_files_for_commit(self, commit_sha, owner="", repo_name=""):
        """
        Lists all the modified files for a given commit.

        :param commit_sha: The SHA of the commit.
        :return: A list of modified files.
        """
        
        if owner == "" or repo_name == "":
            url = f"https://api.github.com/repos/{self.github_repo_owner}/{self.github_repo_name}/commits/{commit_sha}"
        else:
            url = f"https://api.github.com/repos/{owner}/{repo_name}/commits/{commit_sha}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        commit = response.json()
        
        modified_files = [{"filename": file['filename'], "sha": file["sha"] } for file in commit['files']]

       # print(f"Modified files for commit '{commit_sha}':")
        #for file in modified_files:
         #   print(f"- {file}")

        return modified_files
    
    
    def get_issues(self, owner, repo_name, n):

        
        #url = f" https://api.github.com/repos/{self.github_repo_owner}/{self.github_repo_name}/issues"
        url = f" https://api.github.com/repos/{owner}/{repo_name}/issues"
        response = requests.get(url, headers=self.headers)
        issues = response.json()
        
        print(f"Last {n} issues for {owner}/{repo_name}:")
        for issue in issues[:n]:
            print(f"Issue #{issue['number']}: {issue['title']} - {issue['state']}")
            print(f"  Created at: {issue['created_at']}")
            print(f"  Updated at: {issue['updated_at']}")
            print(f"  Body: {issue['body']}\n")
            
        while True:
            choice = input
        
    def get_branches(self, owner="", repo_name=""):
        
        if owner == "" or repo_name=="":    
            url = f" https://api.github.com/repos/{self.github_repo_owner}/{self.github_repo_name}/branches"
        else:
            url = f" https://api.github.com/repos/{owner}/{repo_name}/branches"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            branches = response.json()


            list_branches= [{"name":branch["name"], "sha": branch["commit"]["sha"]} for branch in branches]
            if isinstance(list_branches, list):
                for num, branch in enumerate(list_branches):
                    print(f"Branch #{num+1}")
                    print(f"Name: {branch['name']}")
                    print(f"Commit SHA: {branch['sha']}")
            while True:
                try:
                    choice = int(input("Enter the branch number you wish to examine or 0 to return: "))
                    if choice == 0:
                        break
                    elif isinstance(choice, int) and choice <= len(list_branches):
                        print(self.get_commits(owner, repo_name, list_branches[choice-1]["name"]))
                        break
                    else:
                        print(f"Please enter a branch number.")
                except ValueError:
                    print("Invalid input. Please enter a valid integer.")
        except requests.RequestException as e:
            return f"An error occurred: {e}"
      
    def get_file_content_by_sha(self, sha, owner="", repo_name=""):
        """
        Fetches the content of a file by its SHA from the specified GitHub repository.
        
        :param owner: The owner of the repository.
        :param repo_name: The name of the repository.
        :param sha: The SHA of the file blob.
        :param token: GitHub personal access token for authentication.
        :return: The content of the file.
        """

        if owner == "" or repo_name == "":
            url = f"https://api.github.com/repos/{self.github_repo_owner}/{self.github_repo_name}/git/blobs/{sha}"
        else:
            url = f"https://api.github.com/repos/{owner}/{repo_name}/git/blobs/{sha}"
        #url = f"https://api.github.com/repos/xiphos-dev/Githubapp-demo/git/blobs/{sha}"
        print(f"URL:{url}")
        try:
            response = requests.get(url, headers=self.headers)
            print("Retrieved URL")
            response.raise_for_status()
            blob = response.json()
            
            content = base64.b64decode(blob['content']).decode('utf-8')
            print(f"Decoded:{content}")
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
            print(f"An error occurred: {e}")
            return f"An error occurred: {e}"
        except Exception as e:
            print(f"Error decoding content: {e}")
            return f"Error decoding content: {e}"
    def get_filepath_content(self, filepath, owner="", repo_name=""):
        
        if owner == "" or repo_name=="":
            url = f"https://api.github.com/repos/{self.github_repo_owner}/{self.github_repo_name}/contents/{filepath}"
        else:
            url = f"https://api.github.com/repos/{owner}/{repo_name}/contents/{filepath}"
        print(f"URL:{url}")
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        file_data = response.json()
        
        return self.get_file_content_by_sha(file_data["sha"])
    
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
                
                
    def get_most_popular_comment(self, issue_number):
        """
        Fetches the most popular comment for a given issue.
        
        :param issue_number: The number of the issue.
        :return: The most popular comment based on the number of reactions.
        """
        url = f"https://api.github.com/repos/{self.github_repo_owner}/{self.github_repo_name}/issues/{issue_number}/comments"
        response = requests.get(url, headers=self.headers)
        comments = response.json()

        if not comments:
            print(f"No comments found for issue #{issue_number}")
            return

        most_popular_comment = max(comments, key=lambda c: sum(reaction['count'] for reaction in c['reactions'].values() if isinstance(reaction, int)))

        print(f"Most popular comment for issue #{issue_number}:")
        print(f"  User: {most_popular_comment['user']['login']}")
        print(f"  Created at: {most_popular_comment['created_at']}")
        print(f"  Body: {most_popular_comment['body']}")
        print(f"  Reactions: {most_popular_comment['reactions']}")
        
        
    def list_pull_requests(self, owner="", repo_name="", state="open", sort="created", direction="desc"):
        """
        Lists pull requests for the specified GitHub repository.
        """
        if owner == "" and repo_name == "":
            url = f"https://api.github.com/repos/{self.github_repo_owner}/{self.github_repo_name}/pulls"
            print(f"Pull Requests for {self.github_repo_owner}/{self.github_repo_name}:")
        else:
            url = f"https://api.github.com/repos/{owner}/{repo_name}/pulls"
            print(f"Pull Requests for {owner}/{repo_name}:")
 
        params = {
            'state': state,
            'sort': sort,
            'direction': direction
        }
        response = requests.get(url, headers=self.headers, params=params)

        pull_requests = response.json()
        
        
        pr_numbers = []
        for pr in pull_requests:
            pr_numbers.append(pr['number'])
            print(f"PR #{pr['number']}: {pr['title']} - {pr['state']}")
            print(f"  Created at: {pr['created_at']}")
            print(f"  Updated at: {pr['updated_at']}")
            
        while True:
            try:
                choice = int(input("Enter the PR number you wish to examine or 0 to return: "))
                if choice in pr_numbers:
                    self.list_modified_files_in_pr(choice, owner, repo_name)
                    break
                elif choice == 0:
                    break
                else:
                    print(f"Please enter a PR number.")
            except ValueError:
                print("Invalid input. Please enter a valid integer.")
           # print(f"  Body: {pr['body']}\n")
        
    def list_modified_files_in_pr(self, pr_number, owner="", repo_name=""):
        """
        Lists the modified files in a given pull request.
        
        :param pr_number: The number of the pull request.
        """
        if owner == "" or repo_name == "":
            url = f"https://api.github.com/repos/{self.github_repo_owner}/{self.github_repo_name}/pulls/{pr_number}/files"
        else:
            url = f"https://api.github.com/repos/{owner}/{repo_name}/pulls/{pr_number}/files"
        print(url)
        response = requests.get(url, headers=self.headers)
        files = response.json()
        
        print(f"Modified files in PR #{pr_number}:")
        for num, file in enumerate(files):
            print(f"File no.{num+1}")
            print(f"File: {file['filename']}")
            print(f"  Status: {file['status']}")
            print(f"  Additions: {file['additions']}")
            print(f"  Deletions: {file['deletions']}")
            print(f"  Changes: {file['changes']}\n")
            
        while True:
            try:
                choice = int(input("Enter the file number you wish to examine or 0 to return: "))
                if choice == 0:
                    break
                elif isinstance(choice, int) and choice <= len(files):
                    print(self.get_file_content_by_sha(files[choice-1]["sha"], owner, repo_name))
                    break
                else:
                    print(f"Please enter a valid number.")
            except ValueError:
                print("Invalid input. Please enter a valid integer.")
        
    def list_collaborators(self, owner="", repo_name="", affiliation="all", permission=None, org=None):
        """
        Lists collaborators for the repository with optional filters for affiliation, permission, and organization.

        :param affiliation: Filter by the affiliation of the collaborators. Can be 'outside', 'direct', or 'all'.
        :param permission: Filter by the permission level of the collaborators. Can be 'pull', 'push', or 'admin'.
        :param org: Filter by the organization of the collaborators.
        :return: A list of collaborators.
        """
        
        if owner == "" or repo_name == "":
            url = f"https://api.github.com/repos/{self.github_repo_owner}/{self.github_repo_name}/collaborators"
            print(f"Collaborators for {self.github_repo_owner}/{self.github_repo_name}:")
        else:
            url = f"https://api.github.com/repos/{owner}/{repo_name}/collaborators"
            print(f"Collaborators for {owner}/{repo_name}:")
        print(f"URL:{url}")
        params = {'affiliation': affiliation}
        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        collaborators = response.json()

        filtered_collaborators = []
        for collaborator in collaborators:
            user_data = requests.get(collaborator['url'], headers=self.headers).json()
            #print(collaborator['permissions'])
            if permission and collaborator['permissions'][permission] != True:
                continue
            if org and 'company' in user_data and user_data['company'] != org:
                continue
            filtered_collaborators.append(collaborator)

        print(f"Filtered collaborators (affiliation='{affiliation}', permission='{permission}', org='{org}'): ")
        for collab in filtered_collaborators:
            print(f"- {collab['login']} (Permissions: {json.dumps(collab['permissions'])})")

    def search_file_in_repo(self,filename, owner="", repo_name=""):
        
        if owner == "" or repo_name == "":
            url = f"https://api.github.com/search/code?q={filename}+repo:{self.github_repo_owner}/{self.github_repo_name}"
        else:
            url = f"https://api.github.com/search/code?q={filename}+repo:{owner}/{repo_name}"
        

        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            results = response.json()
            
            # Extract relevant information from the response
            items = results.get('items', [])
            file_results = []
            
            for item in items:
                file_results.append({
                    'name': item['name'],
                    'path': item['path'],
                    'repository': item['repository']['full_name'],
                    'url': item['html_url'],
                    'sha': item['sha']
                })
            if items:    
                for num, item in enumerate(file_results):
                    print(f"File #{num+1}")
                    print(f"File Name: {item['name']}")
                    print(f"File Path: {item['path']}")
                    print(f"Repository: {item['repository']}")
                    print(f"URL: {item['url']}")
                    print()
            else:
                print("No results found or error occurred.")
                
            while True:
                try:
                    choice = int(input("Enter the file number you wish to examine or 0 to return: "))
                    if choice == 0:
                        break
                    elif isinstance(choice, int) and choice <= len(file_results):
                        if owner == "":
                            owner = self.github_repo_owner
                            repo_name = self.github_repo_name
                        print(file_results[choice-1].keys() )
                        print(self.get_file_content_by_sha(file_results[choice-1]["sha"], owner, repo_name))
                        break
                    else:
                        print(f"Please enter a valid number.")
                except ValueError:
                    print("Invalid input. Please enter a valid integer.")
            
            return 
        
        except requests.RequestException as e:
            print(f"An error occurred: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None
    
    def run(self):
        """
        Runs the menu selection process.
        
        :return: The user's choice (1-indexed).
        """
        self.display_menu()
        return self.get_user_choice()

if __name__ == "__main__":
    load_dotenv()

    options = [
        "Fetch Most Recent Commit Details",
        "List Branches",
        "Fetch File Content by path",
        "List Last N Issues",
        "List Pull Requests",
        "List collaborators",
        "Search file name",
        "Exit"
    ]
    
    # Initialize the Menu class with the list of options and GitHub details
    github_access_token = os.getenv("GITHUB_TOKEN")
    #github_repo_owner = 'xiphos-dev'
    #github_repo_name = 'Githubapp-demo'
    github_repo_owner = 'myfuture-ai-team'
    github_repo_name = 'lisa_ia'
    menu = Menu(options, github_access_token, github_repo_owner, github_repo_name)
    
    while True:
        # Run the menu and get the user's choice
        choice = menu.run()
        
        if choice == 1:
            # Get the repository owner and name from the user
            #owner = input("Enter the repository owner: ")
            #repo_name = input("Enter the repository name: ")
            
            # Fetch and display the commits
            
            owner = "qgis"
            repo_name = "QGIS"
            commits = menu.get_commits(owner, repo_name)
            #commits = menu.get_commits()
            '''
            if isinstance(commits, dict):
                print(f"Most recent commit message: {commits['commit_message']}")
                print(f"Commit SHA: {commits['commit_sha']}")
                print("Modified files:")
                for file in commits['files']:
                    print(f"- {file['filename']}: {file['sha']}")
            else:
                for content in commits:
                    print(repr(content))
            '''
        elif choice == 2:
            owner = "qgis"
            repo_name = "QGIS"
            #branches = menu.get_branches(owner, repo_name)
            branches = menu.get_branches()

        
        elif choice == 3:
            path="vision.Dockerfile"

            file_blob = menu.get_filepath_content(path)
            print(file_blob)
                    

        elif choice == 4:
            owner = "qgis"
            repo_name = "QGIS"
            
            
            most_recent_issue = menu.get_issues(owner, repo_name, 5)
            print(most_recent_issue)
            
        elif choice == 5:
            owner = "qgis"
            repo_name = "QGIS"
            state="open" 
            sort="popularity" 
            direction="asc"
            menu.list_pull_requests(owner, repo_name, state, sort, direction)
            #menu.list_pull_requests(state="closed")
            
        elif choice == 6:
            owner = "qgis"
            repo_name = "QGIS"
            permission="admin"
            menu.list_collaborators(owner, repo_name)
            #menu.list_collaborators(permission=permission)
            
        elif choice == 7:
                filename = "handler"
                
                results = menu.search_file_in_repo(filename)

            
        elif choice == len(options):
            print("Exiting...")
            break