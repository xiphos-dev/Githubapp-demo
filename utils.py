import requests
from typing import Dict
import base64
import json

class Issues:
    def search_issues(self, token: str, url: str):
        
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28"
        }   
        
        response = requests.get(url, headers=headers)
        response = response.json()
        return response

class Files:
        
    def search_file(self, token: str, url: str):
        
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28"
        }   
        
        response = requests.get(url, headers=headers)
        response = response.json()
        return response
    
    def get_file_content_by_sha(self, sha: str, token: str, repo_owner: str ="", repo_name: str =""):
        """
        Fetches the content of a file by its SHA from the specified GitHub repository.
        
        :param sha: The SHA of the file blob.
        :param token: GitHub personal access token for authentication.        
        :param repo_owner: The owner of the repository.
        :param repo_name: The name of the repository.
        :return: The content of the file.
        """

        url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/git/blobs/{sha}"
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

class Branches:
    
    def list_all(self, token: str, repo_owner: str, repo_name: str, protected: bool = None, page: int = None, per_page: int = None):
        """
        Lists pull requests from repository
        
        Parameters:
            token (string): Github access token used for authentication.
            repo_owner (string): Github username for the repository owner.  
            repo_name (string): Github name for the repository.

        Returns:
            Document: The  list of all branch names
        """

        url = f" https://api.github.com/repos/{repo_owner}/{repo_name}/branches"
        
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28"
        }
        
        try:
            params = {"protected": protected, "per_page": per_page, "page": page}
            params = {k: v for k, v in params.items() if v is not None}
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            branches = response.json()

            '''
            list_branches= [{"name":branch["name"], "sha": branch["commit"]["sha"]} for branch in branches]
            if isinstance(list_branches, list):
                for num, branch in enumerate(list_branches):
                    print(f"Branch #{num+1}")
                    print(f"Name: {branch['name']}")
                    print(f"Commit SHA: {branch['sha']}")

            return list_branches
            '''
            return branches
        except requests.RequestException as e:
            return f"An error occurred: {e}"
        
    def get_files_from_branch(self, token, owner, repo, branch):
        url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
        headers = {
            'Authorization': f'Bearer {token}',
            'Accept': 'application/vnd.github.v3+json'
        }

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Raise an exception for bad status codes
            data = response.json()
            
            # Extract file information
            files = []
            if 'tree' in data:
                for item in data['tree']:
                    if item['type'] == 'blob':  # Check if it's a file (blob)
                        print(item.keys())
                        files.append({
                            'path': item['path'],
                            'url': item['url'],
                            'type': item['type'],
                            'size': item['size']
                        })
            
            return files

        except requests.exceptions.RequestException as e:
            print(f"Error fetching files from GitHub: {e}")
            return None

class Commits:


    def list_all(self, token: str, repo_owner: str, repo_name: str, sha: str = "", path: str = "", author: str = "", committer: str = "", since: str = "", until: str = "", per_page: int = 30, page: int = 1):
        """
        Lists commits from repository
        
        Parameters:
            token (string): Github access token used for authentication.
            repo_owner (string): Github username for the repository owner.
            repo_name (string): Github name for the repository.
            
        Returns:
            Document: The filtered list of pull requests
        """ 
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28"
        }
        url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/commits"
        
        params = {k: v for k, v in {
            "sha": sha,
            "path": path,
            "author": author,
            "committer": committer,
            "since": since,
            "until": until,
            "per_page": per_page,
            "page": page
        }.items() if v != ""}
    
        #url = "https://api.github.com/repos/xiphos-dev/Githubapp-demo/commits"
        print(f"URL:{url}")
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            commits = response.json()
            if commits:
                if isinstance(commits,list): 
                    # most_recent_commit_sha = commits[0]["sha"]
                    #  most_recent_commit_url = commits[0]["url"]
                    '''
                    for num, commit in enumerate(commits[:5]):
                        print(f"Commit #{num+1}" + "*"*100)
                        print(f"Author: {commit['commit']['author']['name']}")
                        print(f"Date: {commit['commit']['author']['date']}")
                        print(f"Commit message: {commit['commit']['message']}")
                        print(f"Commit SHA: {commit['sha']}")              
                    '''
                    return commits
        except requests.RequestException as e:
            print(f"An error occurred: {e}")
            return []
        
    def get_modified_files_for_commit(self, commit_sha: str, token: str, repo_owner: str, repo_name: str):
        """
        Lists all the modified files for a given commit.
        token (string): Github access token used for authentication.
        repo_owner (string): Github username for the repository owner.
        repo_name (string): Github name for the repository.
        commit_sha (string): The SHA of the commit.
        
        :return: A list of modified files.
        """
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28"
        }
        url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/commits/{commit_sha}"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        commit = response.json()
        
        modified_files = [{"filename": file['filename'], "sha": file["sha"] } for file in commit['files']]

       # print(f"Modified files for commit '{commit_sha}':")
        #for file in modified_files:
         #   print(f"- {file}")

        return modified_files
    
    def get_diff(self, commit_sha: str, token: str, repo_owner: str, repo_name: str):
        """
        Get the diff for a given commit.

        Parameters:
            token (string): Github access token used for authentication.
            repo_owner (string): Github username for the repository owner.
            repo_name (string): Github name for the repository.
            commit_sha (str): The SHA of the commit to retrieve the diff for.

        Returns:
            dict: A dictionary containing the commit information and diff.
        """
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28"
        }
        url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/commits/{commit_sha}"
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            commit_data = response.json()
            
            # Extracting diff information
            files = commit_data.get('files', [])
            diffs = []
            for file in files:
                diffs.append({
                    'filename': file['filename'],
                    'status': file['status'],
                    'additions': file['additions'],
                    'deletions': file['deletions'],
                    'changes': file['changes'],
                    'patch': file.get('patch')  # The actual diff patch (unified diff format)
                })
            
            return {
                'commit_sha': commit_sha,
                'author': commit_data['commit']['author']['name'],
                'date': commit_data['commit']['author']['date'],
                'message': commit_data['commit']['message'],
                'files_changed': diffs
            }
        
        except requests.RequestException as e:
            print(f"An error occurred: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None

class PullRequest:
    
    def pull_requests_to_markdown(self, pull_requests):
        markdown = "# Pull Requests\n\n"
        for pr in pull_requests:
            pr_number = pr['number']
            pr_title = pr['title']
            pr_state = pr['state']
            markdown += f"- **Number:** {pr_number}, **Title:** {pr_title}, **State:** {pr_state}\n"
        return markdown
    
    
    def list_all(self, token: str, repo_owner: str, repo_name: str, state: str = "open", head: str = "", base: str = "", sort: str = "open", direction: str ="desc", page: int = 1, per_page: int = 30):
        """
        Lists pull requests from repository
        
        Parameters:
            token (string): Github access token used for authentication.
            repo_owner (string): Github username for the repository owner.
            repo_name (string): Github name for the repository.
            state (string): Choose to filter "open", "closed" or "all" pull requests.
            head (string): Filter pulls by head user or head organization and branch name in the format of user:ref-name or organization:ref-name. For example: github:new-script-format or octocat:test-branch.
            base (string): Filter pulls by base branch name.
            sort (string): What to sort results by: created, updated, popularity, long-running.
            direction (string): Direction to sort by: asc, desc.
            pages (int): The page number of the results to fetch. 
            per_page (int): The number of results per page.
        Returns:
            Document: The filtered list of pull requests
        """
        
        # Prepare API request
        
        ## Header definition
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28"
        }
        

        url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/pulls"
        #print(f"Pull Requests for {owner}/{repo_name}:")
 
        ## Parameter definition
        params = {k: v for k, v in {
            "state": state,
            "head": head,
            "base": base,
            "sort": sort,
            "direction": direction,
            "per_page": per_page,
            "page": page
        }.items() if v != ""}
        response = requests.get(url, headers=headers, params=params)

        pull_requests = response.json()
        
        '''
        for pr in pull_requests:
            print(f"PR #{pr['number']}: {pr['title']} - {pr['state']}")
            print(f"  Created at: {pr['created_at']}")
            print(f"  Updated at: {pr['updated_at']}")
        '''
        
        return pull_requests
    
    def create_pull_request_review(self, token: str, repo_owner: str, repo_name: str, pull_number: int, event: str, body: str = "", commit_id: str = "", comments: dict = None):
        """
        Create a review for a pull request on GitHub.

        Parameters:
        - owner (str): The owner of the repository.
        - repo (str): The name of the repository.
        - pull_number (int): The number of the pull request.
        - body (str): The body text of the review.
        - event (str): The review action ('APPROVE', 'REQUEST_CHANGES', 'COMMENT').
        - token (str): The personal access token for GitHub authentication.

        Returns:
        - dict: The response from the GitHub API.
        """
        url = f'https://api.github.com/repos/{repo_owner}/{repo_name}/pulls/{pull_number}/reviews'
        headers = {
            'Authorization': f'Bearer {token}',
            'X-GitHub-Api-Version': '2022-11-28',
            'Accept': 'application/vnd.github+json'
        }
        payload = {k: v for k, v in {
            'body': body,
            'event': event,
            'commit_id': commit_id   
            }.items() if v != ""
        }
        
        if comments != None:
            payload["comments"] = comments

        try:
            response = requests.post(url, json=payload, headers=headers)
            #response.raise_for_status()  # Raise an exception for HTTP errors
            return response.json()
        
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            if e.response.status_code == 422:
                print(f"Response content: {e.response.json()}")
            return response.json()
        
    def update_pull_request(self, token: str, repo_owner: str, repo_name: str, pull_number: int, state: str, body: str = "", title: str = "", base: str = ""):
        """

        Parameters:
        - owner (str): The owner of the repository.
        - repo (str): The name of the repository.
        - pull_number (int): The number of the pull request.
        - body (str): The body text of the review.
        - token (str): The personal access token for GitHub authentication.

        Returns:
        - dict: The response from the GitHub API.
        """
        url = f'https://api.github.com/repos/{repo_owner}/{repo_name}/pulls/{pull_number}'
        headers = {
            'Authorization': f'Bearer {token}',
            'X-GitHub-Api-Version': '2022-11-28',
            'Accept': 'application/vnd.github+json'
        }
        payload = {k: v for k, v in {
            'body': body,
            'state': state,
            'base': base,
            'title': title   
            }.items() if v != ""
        }

        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()  # Raise an exception for HTTP errors
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            if e.response.status_code == 422:
                print(f"Response content: {e.response.json()}")
                return response.json()

    def merge(self, token: str, repo_owner: str, repo_name: str, pull_number: int, commit_title: str = "", commit_message: str = "", sha: str = "", merge_method: str = ""):
        """
        Merge a Pull Request.
        
        Parameters:
            token (string): Github access token used for authentication.
            repo_owner (string): Github username for the repository owner.
            repo_name (string): Github name for the repository.
            pull_number (int): Pull request identifier.
            commit_title (string): The review action you want to perform. The review actions include: APPROVE, REQUEST_CHANGES, or COMMENT. 
            commit_message (string): Commit SHA. Defaults to the most recent commit in the pull request when you do not specify a value.
            sha (string): Required when using REQUEST_CHANGES or COMMENT for the event parameter. The body text of the pull request review.
            merge_method (string): Comments specific for each file, for specific lines inside.
        Returns:
            Dict: The result of the merge operation
        """
        
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28"
        }
        
        url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/pulls/{pull_number}/merge"
        
        params = {k: v for k, v in {
            'commit_title': commit_title,
            'commit_message': commit_message,
            'sha': sha,
            'merge_method': merge_method
            }.items() if v != ""
        }
        
        try:
            if len(params.keys()) > 0:
                response = requests.put(url, headers=headers, json=params)
            else:    
                response = requests.put(url, headers=headers)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            result = response.json()
            
            if result.get('merged'):
                print(f"Pull request #{pull_number} merged successfully.")
            else:
                print(f"Failed to merge pull request #{pull_number}: {result.get('message')}")
                
            return result
        except requests.RequestException as e:
            print(f"An error occurred: {e}")
            return response.json()
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None
            
        
    def close(self, token: str, repo_owner: str, repo_name: str, pull_number: int):
        """
        Merge a Pull Request.
        
        Parameters:
            token (string): Github access token used for authentication.
            repo_owner (string): Github username for the repository owner.
            repo_name (string): Github name for the repository.
            pull_number (int): Pull request identifier.
            title (string): The title of the pull request.
            body (string): The contents of the pull request.
            base (string): The name of the branch you want your changes pulled into.
            maintainer_can_modify (bool): Indicates whether maintainers can modify the pull request.
        Returns:
            Dict: Result of the close operation
        """
        url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/pulls/{pull_number}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        data = {
            "state": "closed"
        }
        
        try:
            response = requests.patch(url, headers=headers, json=data)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            result = response.json()
            
            if result.get('state') == 'closed':
                print(f"Pull request #{pull_number} closed successfully.")
            else:
                print(f"Failed to close pull request #{pull_number}: {result.get('message')}")
            return result
            
        except requests.RequestException as e:
            print(f"An error occurred: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None
        
    def diff(self, token: str, repo_owner: str, repo_name: str, pull_number: int, page: int = -1, per_page: int = -1):
        """
        Merge a Pull Request.
        
        Parameters:
            token (string): Github access token used for authentication.
            repo_owner (string): Github username for the repository owner.
            repo_name (string): Github name for the repository.
            pull_number (int): Pull request identifier.
        Returns:
            List: List of diff elements from the pull request.
        """
        url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/pulls/{pull_number}/files"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        params = {k: v for k, v in {
            "page": page,
            "per_page": per_page,
        }.items() if v != -1}
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            files = response.json()
            '''
            diff_items = []
            for file in files:
                diff_items.append({
                    'filename': file['filename'],
                    'status': file['status'],
                    'additions': file['additions'],
                    'deletions': file['deletions'],
                    'changes': file['changes'],
                    'patch': file['patch'] if 'patch' in file else 'No patch information'
                })
            '''
            
            return files
        
        except requests.RequestException as e:
            print(f"An error occurred: {e}")
            return response.json()
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None



import os

pr = PullRequest()

parameters={}
token = os.getenv("GITHUB_TOKEN").strip()
if not token:
    raise ValueError("Token is missing or empty")

parameters["token"] = token
parameters["repo_owner"] = "xiphos-dev"
parameters["repo_name"] = "Githubapp-demo"
#parameters["repo_owner"] = "myfuture-ai-team"
#parameters["repo_name"] = "lisa_ia-dev"
#parameters["repo_owner"] = "qgis"
#parameters["repo_name"] = "QGIS"
#parameters["base"] = "non-branch"
'''
listado = pr.list_all(**parameters)

if isinstance(listado, list):
    if len(listado) == 0:
        print("No pulls found")
    else:
        print(f"Found {len(listado)} pulls")
        print(pr.pull_requests_to_markdown(listado))
elif isinstance(listado, dict):
    print(listado)
    print(f"Could not perform read pull operation:\nMessage: {listado['message']}\nError code: {listado['status']}")
'''

parameters["pull_number"] = 5
parameters["event"] = "APPROVE"
parameters["body"] = "Lorem ipsum change"

parameters_update={}
parameters_update["token"] = token
parameters_update["pull_number"] = 5
parameters_update["repo_owner"] = "xiphos-dev"
parameters_update["repo_name"] = "Githubapp-demo"
parameters_update["state"] = "open"
parameters_update["body"] = "Lorem ipsum change"

parameters_merge = {}
parameters_merge["token"] = token
parameters_merge["repo_owner"] = "xiphos-dev"
parameters_merge["repo_name"] = "Githubapp-demo"
parameters_merge["pull_number"] = 5
parameters_merge["commit_title"] = "API test"
parameters_merge["commit_message"] = "Lorem ipsum"
parameters_merge["sha"] = ""
parameters_merge["merge_method"] = ""


parameters_diff = {}
parameters_diff["token"] = token
parameters_diff["repo_owner"] = "xiphos-dev"
parameters_diff["repo_name"] = "Githubapp-demo"
parameters_diff["pull_number"] = 6

#respuesta = pr.create_pull_request_review(**parameters)
#respuesta = pr.update_pull_request(**parameters_update)
#respuesta = pr.merge(**parameters_merge)
respuesta = pr.diff(**parameters_diff)
#print(respuesta)

def check_diff(diff):
    '''
    for item in diff:
        print(item.keys())
        print(f"Patch:{item.get('patch','')}")
        print(f"Filename:{item['filename']}")
        print(f"Additions:{item['additions']}")
        print(f"Changes:{item['changes']}")
    '''
    print(diff[3])

check_diff(respuesta)
'''
import os

commit = Commits()
parameters={}
token = os.getenv("GITHUB_TOKEN").strip()
if not token:
    raise ValueError("Token is missing or empty")
parameters["token"] = token
parameters["repo_owner"] = "qgis"
parameters["repo_name"] = "QGIS"
#parameters["base"] = "non-branch"
listado = commit.list_all(**parameters)
print(type(listado))
'''
'''
import os
files = Files()
issues = Issues()
parameters={}
token = os.getenv("GITHUB_TOKEN").strip()
if not token:
    raise ValueError("Token is missing or empty")
#url = "https://api.github.com/search/code?q=lorem+in:path,file+created:>=2023-03-01+repo:qgis/QGIS&sort=created&order=asc"
url="https://api.github.com/search/code?q=extension:txt+repo:qgis/QGIS"
#parameters["base"] = "non-branch"
#listado = files.search_file(token, url)
#print(type(listado))
#print("status" in listado.keys())
url_issue = "https://api.github.com/search/issues?q=is:issue+is:open+repo:qgis/QGIS"
comentarios = issues.search_issues(token, url_issue)
print(type(comentarios))
'''

import os

branch = Branches()
parameters = {}

token = os.getenv("GITHUB_TOKEN").strip()
if not token:
    raise ValueError("Token is missing or empty")
parameters["token"] = token
parameters["owner"] = "qgis"
parameters["repo"] = "QGIS"
parameters["branch"] = "release-2_18"

files = branch.get_files_from_branch(**parameters)

if files:
    for file in files:
        print(f"Path: {file['path']}, URL: {file['url']}, Type: {file['type']}, Size: {file['size']} bytes")
else:
    print("Failed to fetch files.")