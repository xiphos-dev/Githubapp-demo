import requests
import concurrent.futures
from git import Repo
import base64
import json
import os
import aiohttp
import asyncio
import logging

class GithubRepo():
    
    
    async def list_all(self, token: str, repo_owner: str, repo_name: str, protected: bool = None, page: int = 1, per_page: int = None):
        """
        
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
            all_branches=[]
            async with aiohttp.ClientSession() as session:
                while True:
                    params = {k: v for k, v in {"protected":protected, "per_page":per_page, "page": page}.items() if v is not None}
                    async with session.get(url, headers=headers, params=params) as response:
                        response.raise_for_status()
                        branches = await response.json()
                        #print(f"Page {page}: {branches}") 
                        if not branches:
                            break
                        all_branches.extend(branches)
                        page += 1

            return all_branches
        
        except aiohttp.ClientResponseError as e:
            print(f"Client response error: {e.status}, message: {e.message}")  # Logging error
            return []
        except aiohttp.ClientConnectionError as e:
            print(f"Client connection error: {e}")  # Logging error
            return []
        except aiohttp.ClientError as e:
            print(f"Client error: {e}")  # Logging error
            return []
        except asyncio.TimeoutError:
            print("Request timed out")  # Logging error
            return []
        except Exception as e:
            print(f"An error occurred: {e}")  # Logging error
            return []
        
        
    def get_files_from_branch(self,token, owner, repo, branch):
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
            folders = []
            #print(f"Largo {len(data['tree'])}")
            if 'tree' in data:
                #print(f"Tree:{data['tree'][0]}")
                for item in data['tree']:
                    #print(item.keys())
                    if item['type'] == 'blob':  # Check if it's a file (blob)
                        #print(item.keys())
                        files.append({
                            'path': item['path'],
                            'url': item['url'],
                            'type': item['type'],
                            'size': item['size'],
                            'mode': item['mode'],
                            'sha': item['sha']
                        })
                    elif item['type'] == 'tree':
                        folders.append(item['path'])
            return files, folders

        except requests.exceptions.RequestException as e:
            print(f"Error fetching files from GitHub: {e}")
            return None


    def run_query(token, query, variables):
        
        url = 'https://api.github.com/graphql'
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"
        }
        response = requests.post(url, json={'query': query, 'variables': variables}, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Query failed to run by returning code of {response.status_code}. {query}")
        

    
    async def fetch_files_in_parallel(self, token, owner, repo, branch, folder_paths, max_concurrent_requests=10):
        # Function to fetch file contents in parallel using graphql
        semaphore = asyncio.Semaphore(max_concurrent_requests)
        async with aiohttp.ClientSession() as session:
            tasks = [
                self.fetch_files_from_folder(semaphore, session, token, owner, repo, branch, folder_path)
                for folder_path in folder_paths
            ]
            results = await asyncio.gather(*tasks)
        
        return results
    
    async def fetch_files_from_folder(self, semaphore, session, token, owner, repo, branch, folder_path):
        max_retries = 5
        delay = 1  # Start with 1 second delay
        url = "https://api.github.com/graphql"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        query = """
        query($owner: String!, $repo: String!, $expression: String!) {
            repository(owner: $owner, name: $repo) {
                object(expression: $expression) {
                    ... on Tree {
                        entries {
                            name
                            type
                            object {
                                ... on Blob {
                                    text
                                }
                            }
                        }
                    }
                }
            }
        }
        """
        
        expression = f"{branch}:{folder_path}"
        
        variables = {
            "owner": owner,
            "repo": repo,
            "expression": expression
        }

        for attempt in range(max_retries):
            try:
                async with semaphore:
                    async with session.post(url, headers=headers, json={'query': query, 'variables': variables}) as response:
                        # Check for rate limit error
                        if response.status == 403:
                            error_data = await response.json()
                            if any(error['type'] == 'RATE_LIMITED' for error in error_data.get('errors', [])):
                                raise aiohttp.ClientResponseError(
                                    request_info=response.request_info,
                                    history=response.history,
                                    status=403,
                                    message='RATE_LIMITED'
                                )
                        
                        response.raise_for_status()
                        data = await response.json()
                        
                        files = data['data']['repository']['object']['entries']
                        
                        # Filter out directories
                        files = [file for file in files if file['type'] == 'blob']
                        
                        
                        
                        for blob_dictionary in files:
                            blob_dictionary['name'] = f"{folder_path}/{blob_dictionary['name']}"
                            #print(f"Name:{blob_dictionary['name']}")
                        return files
                    
            except aiohttp.ClientResponseError as e:
                # Handle rate limiting and other HTTP errors
                if e.status == 403:
                    logging.error(f"Rate limit exceeded: {e.message}")
                    retry_after = e.headers.get('Retry-After')
                    if retry_after:
                        await asyncio.sleep(int(retry_after))
                    else:
                        delay = 2 ** attempt  # Exponential backoff
                        await asyncio.sleep(delay)
                else:
                    logging.error(f"Client response error: {e.status}, message={e.message}")
                    break
            except aiohttp.ClientConnectionError as e:
                logging.error(f"Client connection error: {e}")
                break
            except aiohttp.ClientError as e:
                logging.error(f"Client error: {e}")
                break
            except asyncio.TimeoutError:
                logging.error("Request timed out")
                break
            except Exception as e:
                logging.error(f"An unexpected error occurred: {e}")
                break
            
    def clone_repository_from_env(self, token, repo_url, destination_path):

        authenticated_url = repo_url.replace("https://", f"https://{token}@")
        try:
            print(f"Cloning repository from {authenticated_url} to {destination_path}...")
            Repo.clone_from(authenticated_url, destination_path)
            print("Repository cloned successfully.")
        except Exception as e:
            print(f"Error: {e}")
            
            
    def list_repositories(self, token):
        url = "https://api.github.com/user/repos"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json"
        }
        params = {
            "visibility": "all",  # You can filter by 'all', 'public', or 'private'
            "affiliation": "owner,collaborator,organization_member"  # Adjust as needed
        }
        repositories = []
        page = 1
        
        while True:
            response = requests.get(url, headers=headers, params={**params, "page": page})
            response.raise_for_status()
            repos = response.json()
            if not repos:
                break
            repositories.extend(repos)
            page += 1
        
        return repositories
    
    
    async def download_files_decision(self, token, owner, repo, branch):
        # This function fetches all file data from a branch, and then executes the adequate function to download its files: GraphQL API for <5000 files, and repository cloning for greater file volumes
        
        repo_file_metadata, folder_paths = self.get_files_from_branch(token, owner, repo, branch)
        if  len(folder_paths) > 5000: #Github API request limit per hour
            #TODO check requests remaining for the hour in this token instead of assuming all 5000 requests are available
            repo_url="https://github.com/{owner}/{repo}.git"
            destination_path = "./destination"
            self.clone_repository_from_env(token, repo_url, destination_path)
            
            return destination_path
        
        else:
            
            files, folders = self.get_files_from_branch(token, owner, repo, branch)
            downloaded_files = await self.fetch_files_in_parallel(token, owner, repo, branch, folders)
            '''
            for lista in downloaded_files:
                for diccionario in lista:
                    print(diccionario['name'])
            print(f"Folders:{folders[:1000]}")
            '''
            return downloaded_files    

        

async def main():
    br = GithubRepo()
    token = os.getenv("GITHUB_TOKEN").strip()
    repo="QGIS"
    owner="qgis"
    branch = "master"
    #files, folders = br.get_files_from_branch(token, owner, repo, branch)
    #downloaded_files = await br.fetch_files_in_parallel(token, owner, repo, branch, folders[:2])
    downloaded_files = await br.download_files_decision(token, owner, repo, branch)
    
    for file_list in downloaded_files:
        print("º"*600)
        print(f"Lista:{file_list}")
        if isinstance(file_list, list):
            for blob_metadata in file_list:
                print(blob_metadata['name'])
    #print(f"Folders:{folders[:1000]}")
    
    '''
    branches = await br.list_all(token, owner, repo)
    print(branches)
    for branch in branches:
        print(branch['name'])
    ''' 
asyncio.run(main())
    #owner = "xiphos-dev"
    #repo = "Githubapp-demo"
    #branch = "main"

    #print(f"Total carpetas:{len(folders)}")
    #print(f"Folders:{folders[:2]}")
    #print(folders[threshold])
    #
    #print(downloaded_files)
    #print(len(downloaded_files))



# Run the main function

#repositories = br.list_repositories(token)
#for repo in repositories:
#    print(repo['name'])
        