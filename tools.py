from typing import  Optional, Dict, Any, List
import together
import asyncio
import os
import sys

#if '__file__' not in globals():
 #   __file__ = os.path.abspath('tools.py')  

prompts_path = os.path.abspath('./prompts')

sys.path.append(os.path.abspath(prompts_path))
print("sys.path:", sys.path)
import ast
from pydantic import BaseModel
from utils import PullRequests, Commits, Branches, Files, Issues
from prompts import UNIVERSAL_ANALYSIS,  UNIVERSAL_UNION
import logging, requests
import tiktoken
from transformers import AutoTokenizer
from huggingface_hub.hf_api import HfFolder

#from langchain.tools import BaseTool

class GithubDiff():
    
    HfFolder.save_token(os.getenv('HF_TOKEN').strip())
    def __init__(self, diff, user_prompt, model, export=False, max_tokens=3000):
        self.diff = diff
        self.user_prompt = user_prompt
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.MAX_TOKENS = max_tokens
        self.model = "Hermes" if "Hermes" in model else "Llama"
        self.model_repository = model
        self.export=False
    
    api_key = os.getenv("TOGETHERAI_MIXTRAL_API_KEY")
    if not api_key:
        raise ValueError("API key is missing or empty")
    together.api_key = api_key.strip()
    
    def token_count(self, prompt):
        return len(self.tokenizer.tokenize(prompt))

    def neo_format_prompt(self, prompt, prompt_type, model, final=False):
    
        if prompt_type == "SYSTEM":
            if model == "Llama":
                return "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"+prompt+"<|eot_id|>"
            
        elif prompt_type == "USER":
            if model == "Llama":
                end = ""
                if final:
                    end = "<|start_header_id|>assistant<|end_header_id|>\n\n"
                return "<|start_header_id|>user<|end_header_id|>\n\n"+prompt+"<|eot_id|>"+end
    '''
    def format_prompt(self, prompt, prompt_type, model, divided_user_prompt="", final=False, type=""): 
    
        if prompt_type == "SYSTEM_SPLIT":
            if model == "Llama":
                return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{prompt}The previous instruction is as follows: {divided_user_prompt}.<|eot_id|>"
            elif model == "Hermes":
                return "<|im_start|>system\n"+prompt+divided_user_prompt+"<|im_end|>"
            
        if prompt_type == "SYSTEM":
            
            if model == "Llama":
                if "code" in type:
                    return "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"+prompt+"<|eot_id|>"+CODE_TEMPLATE_2
                elif "documentation" in type:
                    return "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"+prompt+"<|eot_id|>"+DOCUMENTATION_TEMPLATE
                
                return "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"+prompt+"<|eot_id|>"
            elif model == "Hermes":
                return "<|im_start|>system\n"+prompt+"<|im_end|>"
            
        elif prompt_type == "ASSISTANT":
            end = ""
            if model == "Llama":
                if final:
                    end = "<|eot_id|>"
                return "<|start_header_id|>assistant<|end_header_id|>\n\n"+prompt+end
            elif model == "Hermes":
                if final:
                    end = "<|im_end|>"
                return "<|im_start|>assistant\n"+prompt+end
            
        elif prompt_type == "USER":
            end = ""
            if model == "Llama":
                if final:
                    end = "<|start_header_id|>assistant<|end_header_id|>\n\n"
                return "<|start_header_id|>user<|end_header_id|>\n\n"+prompt+"<|eot_id|>"+end
            elif model == "Hermes":
                if final:
                    end = "<|im_start|>assistant\n"
                return "<|im_start|>user\n"+prompt+"<|im_end|>"+end
    '''           
    def format_diff(self, diff, one_diff=False):
        
        if isinstance(diff, list):
            if not one_diff: # one diff is a flag used for debugging single diff elements from the entire list
                all_diffs = ""
                
                for patch in diff:
                    all_diffs = f"{all_diffs}\nFilename - {patch.get('filename')}:```{patch.get('patch','No patch')}```"
                return all_diffs
            else:
                formatted_diffs = []
                for patch in diff:# give format to diff patch information: Filename\n Status\n code
                    file_data = {'Filename': patch.get('filename'), 'Status': patch.get('Status')}
                    formatted_diffs.append((file_data,f"<code> {patch.get('patch','No patch')} </code>"))
                return formatted_diffs
        elif isinstance(diff, dict):
            return "diccionario"
                
            
    def truncate_to_tokens(self, text, max_tokens, model_name="meta-llama/Meta-Llama-3-70B", token_overlap=0, file_details=None):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokens = tokenizer.tokenize(text)
        begin_code_tag = "<code>"
        end_code_tag = "</code>"
        #print(f"TOKENS:{len(tokens)}")
        #print(f"MAX_TOKENS:{max_tokens}")
        if len(tokens) <= max_tokens:
            return f"Filename:{file_details['Filename']}\nStatus:{file_details['Status']}\n{text}", ""
        
        truncated_tokens = tokens[:max_tokens-10] # make room for '<code>' and '</code>' tags
        remaining_tokens = tokens[max_tokens-token_overlap-10:]
        
        truncated_text = f"{tokenizer.convert_tokens_to_string(truncated_tokens)} {end_code_tag}" # terminate the code block with the proper tag after cutting the text in two
        split_string = truncated_text.strip().split("\n")
        #print(f"Splitting patch diff over breakline character [Line 126] - {split_string}")
        if len(split_string) > 2:
            if isinstance(file_details, dict):
                filename = f"Filename:{file_details['Filename']}"
                status = f"Status:{file_details['Status']}"
            else:
                filename = split_string[0] if "Filename" in split_string[0] else ""
                status = split_string[1] if "Status" in split_string[1] else ""
                print(f"Splitting String [Line 172] - Filename:{filename}, Status:{status}")
            remaining_text = f"{filename}\{status}\n{begin_code_tag} {tokenizer.convert_tokens_to_string(remaining_tokens)}"
        else:
            remaining_text = tokenizer.convert_tokens_to_string(remaining_tokens)
    
        return truncated_text, remaining_text
    
    async def inference(self, prompt, model, max_tokens=512):
        try:
            if model == "meta-llama/Meta-Llama-3.1-70B-Instruct":
                    model +="-Turbo"
            print(f"Model:{model}")
            print(prompt)
            output =  together.Complete.create(
                #model = "mistralai/Mixtral-8x7B-Instruct-v0.1", 
                #model="cognitivecomputations/dolphin-2.5-mixtral-8x7b",
                #model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",   
                #model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT",   
                #model="Qwen/Qwen1.5-14B-Chat",
                model=model,
                prompt=prompt,
                #model="Qwen/Qwen2-72B-Instruct",
                #model="meta-llama/Llama-3-70b-chat-hf",
                #model="meta-llama/Llama-3-8b-chat-hf",
                #model="deepseek-ai/deepseek-coder-33b-instruct",
                #prompt=prompt_3,
                #prompt=PROMPT_SATORI_FINAL_L.format(query=query, hist=t2),
                temperature=0.1,
                top_p=0.85,
                top_k=40,
                max_tokens=max_tokens,  
                repetition_penalty=1.1,
                #stop = ["</s>", "[INST]", "<|im_end|>"]
                stop = ["</s>", "[INST]", "User:", "<|eot_id|>"]#, "`", "` "] #, '"', "\"\n\n", "</json>",'"\n']#, "01", "_v01"]
            )
            #print("\n")
            #print(f"Prompt run clash (answer inst):{answer_inst}")
            #out_long = output['output']['choices'][0]['text']
            # print(f"Answer prompt output:\n{out_long}")
            print(f"Answer:{output['output']['choices'][0]['text']}")
            print(output["output"]["usage"])
            print("*"*25)
            return output
    
        except requests.exceptions.HTTPError as e:
            # Log the HTTP error details
            logging.error(f"HTTPError: {e}")
            logging.error(f"Response status code: {e.response.status_code}")
            logging.error(f"Response content: {e.response.content}")
            raise

        except Exception as e:
            # Log any other exceptions
            logging.error(f"Exception: {e}")
            raise
        
    async def _aprep_clash(self,
        chat_history: list, 
        #agent_type: str, 
        #user_p: str, 
        #q_len: int
    ):
        
        self.hist = chat_history
        
        string_list = self.format_diff(self.diff, True)
        #split_diff(listado_strings, tokenizer_hermes)
        #model_name = "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO"
        
        token_and_string_list = []
        
        for file_data, string in string_list:
            token_and_string_list.append((file_data, string, self.token_count(string)))
        
        token_and_string_list = sorted(token_and_string_list, key=lambda x: x[2])
        
        #diff_patch = "```{patch}```"
        tokens_actual = 0
        result = "" # result accumulates strings up to MAX_TOKENS length. Each string accumulated  corresponds to some file's diff patch. This means that there's 1 string per modified/added/removed file in the pull request 
        answers_list = []
        
        for file_data, string, tokens in token_and_string_list:
            
            tokens_actual += tokens
            if tokens_actual <= self.MAX_TOKENS: # accumulate shorter than MAX_TOKENS data inside the prompt
                truncated, _ = self.truncate_to_tokens(string, self.MAX_TOKENS, self.model_repository, file_details=file_data)
                result += truncated + "\n\n"
                #result += diff_patch.format(patch=truncated) + "\n"
                
            else:
                outputs = []
                tokens_actual = 0
                if result != "": # applying inference over previously accumulated data
                    prompt_system = self.neo_format_prompt( UNIVERSAL_ANALYSIS.format(patch=result), "SYSTEM", self.model )
                    prompt_user = f"{self.neo_format_prompt(self.user_prompt.strip(), 'USER', self.model, final=True)}"
                    prompt_truncated = prompt_system + prompt_user
                    #print(f"Prompt_1:{prompt_truncated}")
                    answer = await self.inference(prompt_truncated, self.model_repository)
                    #print(f"Answer:{answer['output']['choices'][0]['text']}")
                    answers_list.append(answer['output']['choices'][0]['text'])
                    
                # splitting long string patch into MAX_TOKEN sized chunks
                truncated, remainder = self.truncate_to_tokens(string, self.MAX_TOKENS, self.model_repository, token_overlap=100, file_details=file_data) 
                #result = diff_patch.format(patch=truncated) + "\n"
                result = truncated + "\n\n"
                remainder_tokens = self.token_count(remainder)
                
                # execute inference over first chunk of string
                prompt_system = self.neo_format_prompt(UNIVERSAL_ANALYSIS.format(patch=result), "SYSTEM", self.model )
                prompt_user = f"{self.neo_format_prompt(self.user_prompt, 'USER', self.model, final=True)}"
                prompt_truncated = prompt_system + prompt_user
                #print(f"Prompt_2:{prompt_truncated}")
                answer = await self.inference(prompt_truncated, self.model_repository)
                #print(f"Answer:{answer['output']['choices'][0]['text']}")
                outputs.append(answer['output']['choices'][0]['text'])
                
                while remainder_tokens > 0: # make sure to finish processing the second part of the split
                    
                    truncated, remainder = self.truncate_to_tokens(remainder, self.MAX_TOKENS, self.model_repository, token_overlap=100, file_details=file_data)
                    result = truncated + "\n\n"
                    #result = diff_patch.format(patch=truncated) + "\n"
                    
                    #previous_instruction = prompt_user.split("```")[0]
                    prompt_system = self.neo_format_prompt(UNIVERSAL_ANALYSIS.format(patch=result), "SYSTEM", self.model )
                    prompt_user = f"{self.neo_format_prompt(self.user_prompt, 'USER', self.model, final=True)}"
                    prompt_truncated = prompt_system + prompt_user
                    #print(f"Prompt_3:{prompt_truncated}")
                    answer = await self.inference(prompt_truncated, self.model_repository)
                    #print(f"Answer:{answer['output']['choices'][0]['text']}")
                    outputs.append(answer['output']['choices'][0]['text'])
                    
                    remainder_tokens = self.token_count(remainder)
                    truncated = remainder
    
                
                #for num, out in enumerate(outputs):
                    #print(f"output #{num}:{out}")
                answers_list.append(outputs)
                result = "" # reset the cumulative content variable
        if result != "":
            prompt_system = self.neo_format_prompt(UNIVERSAL_ANALYSIS.format(patch=result), "SYSTEM", self.model )
            prompt_user = f"{self.neo_format_prompt(self.user_prompt, 'USER', self.model, final=True)}"
            prompt_truncated = prompt_system + prompt_user
            answer = await self.inference(prompt_truncated, self.model_repository)
            answers_list.append(answer['output']['choices'][0]['text'])
            #print(f"Last output:{await self.inference(prompt_truncated, model_name)}")
        '''
        output =    together.Complete.create(
                    #model = "mistralai/Mixtral-8x7B-Instruct-v0.1", 
                    #model="cognitivecomputations/dolphin-2.5-mixtral-8x7b",
                    #model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",   
                    #model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT",   
                    #model="Qwen/Qwen1.5-14B-Chat",
                    #prompt=instruction.format(filename = self.filename, additions = self.additions, deletions = self.deletions, changes = self.changes, user_prompt = self.user_prompt),
                    prompt=universal_prompt.format(user_prompt=self.user_prompt),
                    #model="Qwen/Qwen2-72B-Instruct",
                    model="meta-llama/Llama-3-70b-chat-hf",
                    #model="meta-llama/Llama-3-8b-chat-hf",
                    #model="deepseek-ai/deepseek-coder-33b-instruct",
                    #prompt=prompt_3,
                    #prompt=PROMPT_SATORI_FINAL_L.format(query=query, hist=t2),
                    temperature=0.1,
                    top_p=0.85,
                    top_k=40,
                    max_tokens=128,  
                    repetition_penalty=1.1,
                    #stop = ["</s>", "[INST]", "<|im_end|>"]
                    stop = ["</s>", "[INST]", "User:", "assistant"]#, "`", "` "] #, '"', "\"\n\n", "</json>",'"\n']#, "01", "_v01"]
            )
        '''
        #print("\n")
        #print(f"Prompt unico usado:{instruction}")
        print("*"*25)
        #out_long = output['output']['choices'][0]['text']
        #print(out_long)
        #print(output["output"]["usage"])
        print("*"*25)
        return answers_list

    async def _arun_clash(
        self,
        answers: list,
        answer_model: str ="",
        max_tokens: int=128
    ):
        # Execute tool
        """Use the tool."""
        try:

            # Date and query params 
            #params = calculate_datetime_ranges(self.timezone)
            #params["query"] = parameters
            params={}
            #params["hist"] = self.hist
            '''
            adv_query = await async_chain_together_m(
                prompt=PROMPT_ADVANCED_QUERY_PULL_GITHUB,
                params=params,
                max_tokens=768,
                temperature=0.1,
                top_p=1,
                top_k=50,
                repetition_penalty=1.1,
                model="meta-llama/Llama-3-70b-chat-hf",
                stop = ["</s>", "[INST]", "<|EOT|>", '"'],
                tool_name="ClickUp Advanced Query Search",
                callback_manager=self.async_callback_manager
                )
            '''
            # Log step clickup improver
            #if self.verbose:
            #    step = {"type": "clickup_query", "properties": {"query": adv_query}}
            #    await self.async_callback_manager.queue.put(f"#STEP{json.dumps(step)}STEP#")
            # Run the task extractor
            #pulls = await self._arun(query=adv_query)
            # Log step clickup improver
            #if self.verbose:
            #    step = {"type": f"{self.name}", "results": tasks}
            #    await self.async_callback_manager.queue.put(f"#STEP_OUT{json_step.dumps(step, ignore_nan=True)}STEP_OUT#")
            
            token_counter = 0
            aggregated_answers = ""
            answers_list = []
            unite_aggregation_prompt = "Show me the main points yielded by the analysis."
            for answer in answers:
                
                if isinstance(answer, list): # the original file was split into parts, now the answers must be reassembled into one
                    string_answer="\n\n".join(answer)# reassemble all separated answers into one
                    string_answer = f"{string_answer}"
                    token_counter = self.token_count(string_answer)
                    aggregated_answers += f"\n\n<analysis> {string_answer} </analysis>"  
                    #TODO: Compress a list of answers that together exceed the limit of tokens. Could gather list into max sized chunks before using a prompt to extract their main points.
                    
                elif isinstance(answer, str): # the answer corresponds to one or multiple files, but it did not require to splitting before analysis
                    token_counter += self.token_count(answer)
                    if token_counter <= self.MAX_TOKENS: # if we exceed the token limit here, we can answer in parts, since we do not need to compress separate answers, we can split them into groups
                        aggregated_answers += f"\n\n<analysis> {answer} </analysis>"                
                    else: # The aggregation has exceeded the token limit, execute current aggregated prompt and start over with the aggregation
                        token_counter = self.token_count(answer)
                        system_prompt = self.neo_format_prompt(UNIVERSAL_UNION.format(analysis=aggregated_answers), "SYSTEM", self.model)
                        user_prompt_aggregation = f"{self.neo_format_prompt(unite_aggregation_prompt, 'USER', self.model, final=True)}"
                        prompt_truncated = system_prompt + user_prompt_aggregation
                        print(f"Prompt:{prompt_truncated}")
                        answer = await self.inference(prompt_truncated, self.model)
                        answers_list.append(answer['output']['choices'][0]['text'])
                        aggregated_answers = f"\n{answer}"    
                        
            if aggregated_answers != "": # the current aggregation never exceeded the limit, thus, it was never executed in the previous for cycle
                system_prompt = self.neo_format_prompt(UNIVERSAL_UNION.format(analysis=aggregated_answers), "SYSTEM", self.model)
                user_prompt_aggregation = f"{self.neo_format_prompt(unite_aggregation_prompt, 'USER', self.model, final=True)}"
                prompt_truncated = system_prompt + user_prompt_aggregation
                print(f"Prompt:{prompt_truncated}")
                answer = await self.inference(prompt_truncated, answer_model if answer_model != "" else self.model_repository)
                answers_list.append(answer['output']['choices'][0]['text'])
                
            '''            
            output =  together.Complete.create(
                    #model = "mistralai/Mixtral-8x7B-Instruct-v0.1", 
                    #model="cognitivecomputations/dolphin-2.5-mixtral-8x7b",
                    model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",   
                    #model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT",   
                    #model="Qwen/Qwen1.5-14B-Chat",
                    prompt=answer_inst,
                    #model="Qwen/Qwen2-72B-Instruct",
                    #model="meta-llama/Llama-3-70b-chat-hf",
                    #model="meta-llama/Llama-3-8b-chat-hf",
                    #model="deepseek-ai/deepseek-coder-33b-instruct",
                    #prompt=prompt_3,
                    #prompt=PROMPT_SATORI_FINAL_L.format(query=query, hist=t2),
                    temperature=0.1,
                    top_p=0.85,
                    top_k=40,
                    max_tokens=128,  
                    repetition_penalty=1.1,
                    #stop = ["</s>", "[INST]", "<|im_end|>"]
                    stop = ["</s>", "[INST]", "User:", "assistant"]#, "`", "` "] #, '"', "\"\n\n", "</json>",'"\n']#, "01", "_v01"]
            )
            '''
            #print("\n")
            #print(f"Prompt run clash (answer inst):{answer_inst}")
            #out_long = output['output']['choices'][0]['text']
            #print(f"Answer prompt output:\n{out_long}")
            #print(output["output"]["usage"])
            print("*"*25)
            # Export
            if self.export:
                await self.async_callback_manager.queue.put(f"\nExport: [{self.user_msg}]({tasks['export_file']})")
            #return {"finish": True, "response": output['output']['choices'][0]['text']}
            return {"finish":True, "response": answers_list}
        except Exception as e_tasks_clickup:
            ic(e_tasks_clickup)
            spaces = '- ' + '\n- '.join([f'{x}: {y}' for x, y in self.clickup_dicts['spaces'].items()])
            trace = traceback.format_exc()
            return {"error": "No tasks could be extracted.", "raise_msg": f"Is it likely the query used and the search parameters are not correct."}

class GithubReviewPullRequest():
    
    def __init__(self, token, repo_owner, repo_name, user_q, pull_number):
        self.token = token
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.user_q = user_q
        self.pull_number = pull_number
        self.export = False

    
    pr = PullRequests()
    
    parameters = '''
    | Parameter       | Type     | Description                                                                                   |
    |-----------------|----------|-----------------------------------------------------------------------------------------------|
    | commit_id       | string   | The SHA of the commit to review.                                                               |
    | body            | string   | Required when using REQUEST_CHANGES or COMMENT for the event parameter. The body text of the pull request review.                                                           |
    | event           | string   | The action to perform. Possible values: `APPROVE`, `REQUEST_CHANGES`, `COMMENT`.               |
    | comments        | array    | An array of comments associated with the review. Each comment should be an object with `path`, `position`, and `body` fields. |
    '''
    
    error_codes = '''
    | Status Code | Error Message                  | Description                                                                 |
    |--------------|--------------------------------|-----------------------------------------------------------------------------|
    | 200          | OK                             | The request was successful.                                                 |
    | 301          | Moved Permanently              | The resource has been moved to a new URL.                                   |
    | 304          | Not Modified                   | There was no new data to return.                                            |
    | 400          | Bad Request                    | The request was invalid or cannot be served.                                |
    | 401          | Unauthorized                   | Authentication is required and has failed or has not yet been provided.     |
    | 403          | Forbidden                      | The request is understood, but it has been refused or access is not allowed.|
    | 404          | Not Found                      | The requested resource could not be found.                                  |
    | 422          | Unprocessable Entity           | The request was well-formed but was unable to be followed due to semantic errors.|
    | 500          | Internal Server Error          | An error occurred on the server.                                            |
    | 502          | Bad Gateway                    | The server was acting as a gateway or proxy and received an invalid response from the upstream server.|
    | 503          | Service Unavailable            | The server is currently unavailable (because it is overloaded or down for maintenance).|
    | 504          | Gateway Timeout                | The server was acting as a gateway or proxy and did not receive a timely response from the upstream server.|
    '''

    ANSWER_SYSTEM_SUCCESS = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a helpful AI assistant, knowledgeable about the Github API. You will be reporting to the user about the information yielded by a previous pull request review submittion.<|eot_id|>
    '''
    ANSWER_SYSTEM_ERROR = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a helpful AI assistant, knowledgeable about the Github API. You will receive an error related to a query made to the Github API. Explain the error to the user, offer a suggestion to fix it. <|eot_id|>
    '''
    
    api_key = os.getenv("TOGETHERAI_MIXTRAL_API_KEY")
    if not api_key:
        raise ValueError("API key is missing or empty")
    together.api_key = api_key.strip()
    
    async def _aprep_clash(self,
        chat_history: list, 
        #agent_type: str, 
        #user_p: str, 
        #q_len: int
    ):
        
        self.hist = chat_history
    
        instruction = '''
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a helpful AI assistant, knowledgeable about the Github API. You will be asked to read text and extract parameter values used for queries to the Github API. Specifically, the queries will be related to Pull Request Reviews. The following contains all of the parameters you will need to look for inside the text:\n{parameters}\n
            Omit every parameter that is not mentioned. If no parameters are mentioned, return empty braces '{{}}'. Your answer must only include the relevant field name and its value in a json format. <|eot_id|><|start_header_id|>user<|end_header_id|>
            Approve the pull request<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            {{"event":"APPROVE"}}<|eot_id|><|start_header_id|>user<|end_header_id|>
            Add comment almost done, add error handling next<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            {{"event":"COMMENT",\n"body":"almost done, add error handling next"}}<|eot_id|><|start_header_id|>user<|end_header_id|>
            {user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|> 
            '''
        
        #example = "- Search for open pull requests with base branch tester\n- Search for all pull requests with head octocat:new-branch, sort them in ascending order."
        #user_p ="List page 2 of the pull request with 'tester' as base branch"
        #user_p = "List the fourth page of the file changes made to 675"
        #field = ['state="', 'head="', 'base="', 'sort="', 'direction="', 'page="', 'per_page="']
        output =    together.Complete.create(
                    #model = "mistralai/Mixtral-8x7B-Instruct-v0.1", 
                    #model="cognitivecomputations/dolphin-2.5-mixtral-8x7b",
                    #model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",   
                    #model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT",   
                    #model="Qwen/Qwen1.5-14B-Chat",
                    prompt=instruction.format(parameters = self.parameters, user_prompt = self.user_q),
                    #model="Qwen/Qwen2-72B-Instruct",
                    model="meta-llama/Llama-3-70b-chat-hf",
                    #model="meta-llama/Llama-3-8b-chat-hf",
                    #model="deepseek-ai/deepseek-coder-33b-instruct",
                    #prompt=prompt_3,
                    #prompt=PROMPT_SATORI_FINAL_L.format(query=query, hist=t2),
                    temperature=0.1,
                    top_p=0.85,
                    top_k=40,
                    max_tokens=128,  
                    repetition_penalty=1.1,
                    #stop = ["</s>", "[INST]", "<|im_end|>"]
                    stop = ["</s>", "[INST]", "User:", "assistant"]#, "`", "` "] #, '"', "\"\n\n", "</json>",'"\n']#, "01", "_v01"]
            )
        #print("\n")
        print(f"Prompt prep clash:{instruction}")
        print("*"*25)
        out_long = output['output']['choices'][0]['text']
        print(out_long)
        print(output["output"]["usage"])
        print("*"*25)
        return out_long
    
    async def _arun_clash(
        self,
        parameters: dict,
    ):
        # Execute tool
        """Use the tool."""
        try:

            # Date and query params 
            #params = calculate_datetime_ranges(self.timezone)
            #params["query"] = parameters
            params={}
            #params["hist"] = self.hist
            '''
            adv_query = await async_chain_together_m(
                prompt=PROMPT_ADVANCED_QUERY_PULL_GITHUB,
                params=params,
                max_tokens=768,
                temperature=0.1,
                top_p=1,
                top_k=50,
                repetition_penalty=1.1,
                model="meta-llama/Llama-3-70b-chat-hf",
                stop = ["</s>", "[INST]", "<|EOT|>", '"'],
                tool_name="ClickUp Advanced Query Search",
                callback_manager=self.async_callback_manager
                )
            '''
            # Log step clickup improver
            #if self.verbose:
            #    step = {"type": "clickup_query", "properties": {"query": adv_query}}
            #    await self.async_callback_manager.queue.put(f"#STEP{json.dumps(step)}STEP#")
            # Run the task extractor
            #pulls = await self._arun(query=adv_query)
            # Log step clickup improver
            #if self.verbose:
            #    step = {"type": f"{self.name}", "results": tasks}
            #    await self.async_callback_manager.queue.put(f"#STEP_OUT{json_step.dumps(step, ignore_nan=True)}STEP_OUT#")
            
            parameters["repo_owner"] = self.repo_owner
            parameters["repo_name"] = self.repo_name
            parameters["token"] = self.token
            parameters["pull_number"] = self.pull_number
            
            '''
            token = os.getenv("GITHUB_TOKEN")
            if not token:
                raise ValueError("Token is missing or empty")
            parameters["token"] = token.strip()
            '''
            review_result = self.pr.create_pull_request_review(**parameters)
            
            parameters.pop("token")
            if "status" not in review_result.keys(): # status means an error ocurred when submitting review
                #print("No pulls found")
                #print(f"Found {len(pr_list)} pulls")
                extra_export = ", finally, you must notify the user the tasks were exported into a csv file inside the session folder, this can be located on the right menu" if self.export else ""
                answer_inst = f'''{self.ANSWER_SYSTEM_SUCCESS}<|start_header_id|>user<|end_header_id|>
                Report succinctly to the user that the review was submitted succesfully<|eot_id|><|start_header_id|>assistant<|end_header_id|>'''
                    
            else:
                #print(pr_list)
                #print(f"Could not perform read pull operation:\nMessage: {pr_list['message']}\nError code: {pr_list['status']}")
                answer_inst = f'''{self.ANSWER_SYSTEM_ERROR}<|start_header_id|>user<|end_header_id|>
                An error ocurred when attempting to submit the pull request review. The most common errors are:\n```{self.error_codes}```\n.The message error is {review_result['message']} and the error code is {review_result['status']}.\n Explain what the possible cause of the error is and suggest a solution.<|eot_id|><|start_header_id|>assistant<|end_header_id|>'''              
            '''
            params = {
                "repo_owner": self.repo_owner,                
                "repo_name": self.repo_name,
                "hist": self.hist,
                #"lang": self.lang,
                "task_answer_instruction": answer_inst,
                #"query": self.user_msg,
                #"date": datetime.now(pytz.timezone(self.timezone)).strftime("%A, %d of %B with time %H:%M of %Y"),
                "init_lang": "Por supuesto!\n" if self.lang == "Spanish" else "Certainly!\n"
            }
            
            # Stream
            
            result = await achain_nirvana(
            prompt=general_prompt,
            params=params,
            max_tokens=2048,
            temperature=0.1,
            top_p=0.85,
            top_k=40,
            repetition_penalty=1.1,
            model = "together|meta-llama/Llama-3-70b-chat-hf",
            stop = ["</s>", "[INST]", "<|EOT|>", "<|eot_id|>"],
            tool_name="SQL Generation",
            stream=False,
            )
            
            msg_exec = await stream_chain_together_m(
                prompt=PROMPT_CLICKUP_ANSWER_SATORI,
                params=params,
                max_tokens=2048,
                callback_manager=self.async_callback_manager,
                model="meta-llama/Llama-3-8b-chat-hf",
                stop=["</s>", "[INST]", "<|im_end|>"],
                tool_name="ClickUp Tasks Answer",
            )
            '''
            
            output =  together.Complete.create(
                    #model = "mistralai/Mixtral-8x7B-Instruct-v0.1", 
                    #model="cognitivecomputations/dolphin-2.5-mixtral-8x7b",
                    model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",   
                    #model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT",   
                    #model="Qwen/Qwen1.5-14B-Chat",
                    prompt=answer_inst,
                    #model="Qwen/Qwen2-72B-Instruct",
                    #model="meta-llama/Llama-3-70b-chat-hf",
                    #model="meta-llama/Llama-3-8b-chat-hf",
                    #model="deepseek-ai/deepseek-coder-33b-instruct",
                    #prompt=prompt_3,
                    #prompt=PROMPT_SATORI_FINAL_L.format(query=query, hist=t2),
                    temperature=0.1,
                    top_p=0.85,
                    top_k=40,
                    max_tokens=128,  
                    repetition_penalty=1.1,
                    #stop = ["</s>", "[INST]", "<|im_end|>"]
                    stop = ["</s>", "[INST]", "User:", "assistant"]#, "`", "` "] #, '"', "\"\n\n", "</json>",'"\n']#, "01", "_v01"]
            )
            #print("\n")
            print(f"Prompt run clash (answer inst):{answer_inst}")
            out_long = output['output']['choices'][0]['text']
           # print(f"Answer prompt output:\n{out_long}")
            print(output["output"]["usage"])
            print("*"*25)
            # Export
            if self.export:
                await self.async_callback_manager.queue.put(f"\nExport: [{self.user_msg}]({tasks['export_file']})")
            return {"finish": True, "response": output['output']['choices'][0]['text']}
        except Exception as e_tasks_clickup:
            ic(e_tasks_clickup)
            spaces = '- ' + '\n- '.join([f'{x}: {y}' for x, y in self.clickup_dicts['spaces'].items()])
            trace = traceback.format_exc()
            return {"error": "No tasks could be extracted.", "raise_msg": f"Is it likely the query used and the search parameters are not correct. Explain this query to the user in natural language and ask for feedback. These are the spaces available:\n{spaces}\nQuestion: {query}\nSearch query: {adv_query}", "trace": trace}

class GithubMergePullRequest():

    def __init__(self, token, repo_owner, repo_name, user_q, pull_number):
        self.token = token
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.user_q = user_q
        self.pull_number = pull_number
        self.export = False

    
    pr = PullRequests()
    
    parameters = '''
    | Parameter         | Type     | Description                                                                                         |
    |-------------------|----------|-----------------------------------------------------------------------------------------------------|
    | commit_title      | string   | Title for the automatic commit message.                                                             |
    | commit_message    | string   | Extra detail to append to the automatic commit message.                                             |
    | sha               | string   | SHA that pull request head must match to allow merge.                                               |
    | merge_method      | string   | Merge method to use. Possible values are `merge`, `squash`, or `rebase`.                            |
    '''
    error_codes = '''
    | Status Code | Error Message                  | Description                                                                 |
    |--------------|--------------------------------|-----------------------------------------------------------------------------|
    | 200          | OK                             | The request was successful.                                                 |
    | 301          | Moved Permanently              | The resource has been moved to a new URL.                                   |
    | 304          | Not Modified                   | There was no new data to return.                                            |
    | 400          | Bad Request                    | The request was invalid or cannot be served.                                |
    | 401          | Unauthorized                   | Authentication is required and has failed or has not yet been provided.     |
    | 403          | Forbidden                      | The request is understood, but it has been refused or access is not allowed.|
    | 404          | Not Found                      | The requested resource could not be found.                                  |
    | 422          | Unprocessable Entity           | The request was well-formed but was unable to be followed due to semantic errors.|
    | 500          | Internal Server Error          | An error occurred on the server.                                            |
    | 502          | Bad Gateway                    | The server was acting as a gateway or proxy and received an invalid response from the upstream server.|
    | 503          | Service Unavailable            | The server is currently unavailable (because it is overloaded or down for maintenance).|
    | 504          | Gateway Timeout                | The server was acting as a gateway or proxy and did not receive a timely response from the upstream server.|
    '''

    ANSWER_SYSTEM_SUCCESS = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a helpful AI assistant, knowledgeable about the Github API. You will be reporting to the user about the information yielded by a previous merge to a pull request.<|eot_id|>
    '''
    ANSWER_SYSTEM_ERROR = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a helpful AI assistant, knowledgeable about the Github API. You will receive an error related to a query made to the Github API. Explain the error to the user, offer a suggestion to fix it. <|eot_id|>
    '''
    
    api_key = os.getenv("TOGETHERAI_MIXTRAL_API_KEY")
    if not api_key:
        raise ValueError("API key is missing or empty")
    together.api_key = api_key.strip()
    
    async def _aprep_clash(self,
        chat_history: list, 
        #agent_type: str, 
        #user_p: str, 
        #q_len: int
    ):
        
        self.hist = chat_history
    
        instruction = '''
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a helpful AI assistant, knowledgeable about the Github API. You will be asked to read text and extract parameter values used for queries to the Github API. Specifically, the queries will be related to a merge over a pull request. The following contains all of the parameters you will need to look for inside the text:\n{parameters}\n
            Omit every parameter that is not mentioned. If no parameters are mentioned, return empty braces '{{}}'. Your answer must only include the relevant field name and its value in a json format. <|eot_id|><|start_header_id|>user<|end_header_id|>
            Merge the pull request with commit title 'Tools testing' and message 'merged this PR using tools'<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            {{"commit_title":"Tools testing",\n"commit_message":"merged this PR using tools"}}<|eot_id|><|start_header_id|>user<|end_header_id|>
            Merge the PR with sha '123456abcd'<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            {{"sha":"123456abcd"}}<|eot_id|><|start_header_id|>user<|end_header_id|>
            {user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|> 
            '''
        
        #example = "- Search for open pull requests with base branch tester\n- Search for all pull requests with head octocat:new-branch, sort them in ascending order."
        #user_p ="List page 2 of the pull request with 'tester' as base branch"
        #user_p = "List the fourth page of the file changes made to 675"
        #field = ['state="', 'head="', 'base="', 'sort="', 'direction="', 'page="', 'per_page="']
        output =    together.Complete.create(
                    #model = "mistralai/Mixtral-8x7B-Instruct-v0.1", 
                    #model="cognitivecomputations/dolphin-2.5-mixtral-8x7b",
                    #model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",   
                    #model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT",   
                    #model="Qwen/Qwen1.5-14B-Chat",
                    prompt=instruction.format(parameters = self.parameters, user_prompt = self.user_q),
                    #model="Qwen/Qwen2-72B-Instruct",
                    model="meta-llama/Llama-3-70b-chat-hf",
                    #model="meta-llama/Llama-3-8b-chat-hf",
                    #model="deepseek-ai/deepseek-coder-33b-instruct",
                    #prompt=prompt_3,
                    #prompt=PROMPT_SATORI_FINAL_L.format(query=query, hist=t2),
                    temperature=0.1,
                    top_p=0.85,
                    top_k=40,
                    max_tokens=128,  
                    repetition_penalty=1.1,
                    #stop = ["</s>", "[INST]", "<|im_end|>"]
                    stop = ["</s>", "[INST]", "User:", "assistant"]#, "`", "` "] #, '"', "\"\n\n", "</json>",'"\n']#, "01", "_v01"]
            )
        #print("\n")
        print(f"Prompt prep clash:{instruction}")
        print("*"*25)
        out_long = output['output']['choices'][0]['text']
        print(out_long)
        print(output["output"]["usage"])
        print("*"*25)
        return out_long    
    
    async def _arun_clash(
        self,
        parameters: dict,
    ):
        # Execute tool
        """Use the tool."""
        try:

            # Date and query params 
            #params = calculate_datetime_ranges(self.timezone)
            #params["query"] = parameters
            params={}
            #params["hist"] = self.hist
            '''
            adv_query = await async_chain_together_m(
                prompt=PROMPT_ADVANCED_QUERY_PULL_GITHUB,
                params=params,
                max_tokens=768,
                temperature=0.1,
                top_p=1,
                top_k=50,
                repetition_penalty=1.1,
                model="meta-llama/Llama-3-70b-chat-hf",
                stop = ["</s>", "[INST]", "<|EOT|>", '"'],
                tool_name="ClickUp Advanced Query Search",
                callback_manager=self.async_callback_manager
                )
            '''
            # Log step clickup improver
            #if self.verbose:
            #    step = {"type": "clickup_query", "properties": {"query": adv_query}}
            #    await self.async_callback_manager.queue.put(f"#STEP{json.dumps(step)}STEP#")
            # Run the task extractor
            #pulls = await self._arun(query=adv_query)
            # Log step clickup improver
            #if self.verbose:
            #    step = {"type": f"{self.name}", "results": tasks}
            #    await self.async_callback_manager.queue.put(f"#STEP_OUT{json_step.dumps(step, ignore_nan=True)}STEP_OUT#")
            
            parameters["repo_owner"] = self.repo_owner
            parameters["repo_name"] = self.repo_name
            parameters["token"] = self.token
            parameters["pull_number"] = self.pull_number
            
            '''
            token = os.getenv("GITHUB_TOKEN")
            if not token:
                raise ValueError("Token is missing or empty")
            parameters["token"] = token.strip()
            '''
            merge_result = self.pr.merge(**parameters)
            
            parameters.pop("token")
            if "status" not in merge_result.keys(): # status means an error ocurred when submitting review
                #print("No pulls found")
                #print(f"Found {len(pr_list)} pulls")
                extra_export = ", finally, you must notify the user the tasks were exported into a csv file inside the session folder, this can be located on the right menu" if self.export else ""
                answer_inst = f'''{self.ANSWER_SYSTEM_SUCCESS}<|start_header_id|>user<|end_header_id|>
                Report succinctly to the user that the merge was executed succesfully<|eot_id|><|start_header_id|>assistant<|end_header_id|>'''
                    
            else:
                #print(pr_list)
                #print(f"Could not perform read pull operation:\nMessage: {pr_list['message']}\nError code: {pr_list['status']}")
                answer_inst = f'''{self.ANSWER_SYSTEM_ERROR}<|start_header_id|>user<|end_header_id|>
                An error ocurred when attempting to submit the merge request. The most common errors are:\n```{self.error_codes}```\n.The message error is {merge_result['message']} and the error code is {merge_result['status']}.\n Explain what the possible cause of the error is and suggest a solution.<|eot_id|><|start_header_id|>assistant<|end_header_id|>'''              
            '''
            params = {
                "repo_owner": self.repo_owner,                
                "repo_name": self.repo_name,
                "hist": self.hist,
                #"lang": self.lang,
                "task_answer_instruction": answer_inst,
                #"query": self.user_msg,
                #"date": datetime.now(pytz.timezone(self.timezone)).strftime("%A, %d of %B with time %H:%M of %Y"),
                "init_lang": "Por supuesto!\n" if self.lang == "Spanish" else "Certainly!\n"
            }
            
            # Stream
            
            result = await achain_nirvana(
            prompt=general_prompt,
            params=params,
            max_tokens=2048,
            temperature=0.1,
            top_p=0.85,
            top_k=40,
            repetition_penalty=1.1,
            model = "together|meta-llama/Llama-3-70b-chat-hf",
            stop = ["</s>", "[INST]", "<|EOT|>", "<|eot_id|>"],
            tool_name="SQL Generation",
            stream=False,
            )
            
            msg_exec = await stream_chain_together_m(
                prompt=PROMPT_CLICKUP_ANSWER_SATORI,
                params=params,
                max_tokens=2048,
                callback_manager=self.async_callback_manager,
                model="meta-llama/Llama-3-8b-chat-hf",
                stop=["</s>", "[INST]", "<|im_end|>"],
                tool_name="ClickUp Tasks Answer",
            )
            '''
            
            output =  together.Complete.create(
                    #model = "mistralai/Mixtral-8x7B-Instruct-v0.1", 
                    #model="cognitivecomputations/dolphin-2.5-mixtral-8x7b",
                    model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",   
                    #model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT",   
                    #model="Qwen/Qwen1.5-14B-Chat",
                    prompt=answer_inst,
                    #model="Qwen/Qwen2-72B-Instruct",
                    #model="meta-llama/Llama-3-70b-chat-hf",
                    #model="meta-llama/Llama-3-8b-chat-hf",
                    #model="deepseek-ai/deepseek-coder-33b-instruct",
                    #prompt=prompt_3,
                    #prompt=PROMPT_SATORI_FINAL_L.format(query=query, hist=t2),
                    temperature=0.1,
                    top_p=0.85,
                    top_k=40,
                    max_tokens=128,  
                    repetition_penalty=1.1,
                    #stop = ["</s>", "[INST]", "<|im_end|>"]
                    stop = ["</s>", "[INST]", "User:", "assistant"]#, "`", "` "] #, '"', "\"\n\n", "</json>",'"\n']#, "01", "_v01"]
            )
            #print("\n")
            print(f"Prompt run clash (answer inst):{answer_inst}")
            out_long = output['output']['choices'][0]['text']
           # print(f"Answer prompt output:\n{out_long}")
            print(output["output"]["usage"])
            print("*"*25)
            # Export
            if self.export:
                await self.async_callback_manager.queue.put(f"\nExport: [{self.user_msg}]({tasks['export_file']})")
            return {"finish": True, "response": output['output']['choices'][0]['text']}
        except Exception as e_tasks_clickup:
            ic(e_tasks_clickup)
            spaces = '- ' + '\n- '.join([f'{x}: {y}' for x, y in self.clickup_dicts['spaces'].items()])
            trace = traceback.format_exc()
            return {"error": "No tasks could be extracted.", "raise_msg": f"Is it likely the query used and the search parameters are not correct. Explain this query to the user in natural language and ask for feedback. These are the spaces available:\n{spaces}\nQuestion: {query}\nSearch query: {adv_query}", "trace": trace}

class GithubUpdatePullRequest():
    
    #token: str, repo_owner: str, repo_name: str, pull_number: int, state: str, body: str = "", title: str = "", base: str = ""
    def __init__(self, token, repo_owner, repo_name, user_q, pull_number):
        self.token = token
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.user_q = user_q
        self.pull_number = pull_number
        self.export = False

    
    pr = PullRequests()
    
    parameters = '''
    | Parameter       | Type     | Description                                                                                   |
    |-----------------|----------|-----------------------------------------------------------------------------------------------|
    | title           | string   | The title of the pull request.                                                                |
    | body            | string   | The contents of the pull request.                                                             |
    | state           | string   | State of the pull request. Possible values are `open`, `closed`.                              |
    | base            | string   | The name of the branch you want your changes pulled into. This should be an existing branch on the current repository. You cannot update the base branch on a pull request to point to another repository. |
    | maintainer_can_modify | boolean | Indicates whether maintainers can modify the pull request.                                  |
    '''
    error_codes = '''
    | Status Code | Error Message                  | Description                                                                 |
    |--------------|--------------------------------|-----------------------------------------------------------------------------|
    | 200          | OK                             | The request was successful.                                                 |
    | 301          | Moved Permanently              | The resource has been moved to a new URL.                                   |
    | 304          | Not Modified                   | There was no new data to return.                                            |
    | 400          | Bad Request                    | The request was invalid or cannot be served.                                |
    | 401          | Unauthorized                   | Authentication is required and has failed or has not yet been provided.     |
    | 403          | Forbidden                      | The request is understood, but it has been refused or access is not allowed.|
    | 404          | Not Found                      | The requested resource could not be found.                                  |
    | 422          | Unprocessable Entity           | The request was well-formed but was unable to be followed due to semantic errors.|
    | 500          | Internal Server Error          | An error occurred on the server.                                            |
    | 502          | Bad Gateway                    | The server was acting as a gateway or proxy and received an invalid response from the upstream server.|
    | 503          | Service Unavailable            | The server is currently unavailable (because it is overloaded or down for maintenance).|
    | 504          | Gateway Timeout                | The server was acting as a gateway or proxy and did not receive a timely response from the upstream server.|
    '''

    ANSWER_SYSTEM_SUCCESS = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a helpful AI assistant, knowledgeable about the Github API. You will be reporting to the user about the information yielded by a previous update to a pull request.<|eot_id|>
    '''
    ANSWER_SYSTEM_ERROR = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a helpful AI assistant, knowledgeable about the Github API. You will receive an error related to a query made to the Github API. Explain the error to the user, offer a suggestion to fix it. <|eot_id|>
    '''
    
    api_key = os.getenv("TOGETHERAI_MIXTRAL_API_KEY")
    if not api_key:
        raise ValueError("API key is missing or empty")
    together.api_key = api_key.strip()
    
    async def _aprep_clash(self,
        chat_history: list, 
        #agent_type: str, 
        #user_p: str, 
        #q_len: int
    ):
        
        self.hist = chat_history
    
        instruction = '''
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a helpful AI assistant, knowledgeable about the Github API. You will be asked to read text and extract parameter values used for queries to the Github API. Specifically, the queries will be related to an update for a pull request. The following contains all of the parameters you will need to look for inside the text:\n{parameters}\n
            Omit every parameter that is not mentioned. If no parameters are mentioned, return empty braces '{{}}'. Your answer must only include the relevant field name and its value in a json format. <|eot_id|><|start_header_id|>user<|end_header_id|>
            Set the title to 'Lorem ipsum' and the body to 'testing update tool'<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            {{"title":"Lorem ipsum",\n"body":"testing update tool"}}<|eot_id|><|start_header_id|>user<|end_header_id|>
            Close the PR, set the base branch to 'phobos'<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            {{"state":"closed",\n"base":"phobos"}}<|eot_id|><|start_header_id|>user<|end_header_id|>
            {user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|> 
            '''
        
        #example = "- Search for open pull requests with base branch tester\n- Search for all pull requests with head octocat:new-branch, sort them in ascending order."
        #user_p ="List page 2 of the pull request with 'tester' as base branch"
        #user_p = "List the fourth page of the file changes made to 675"
        #field = ['state="', 'head="', 'base="', 'sort="', 'direction="', 'page="', 'per_page="']
        output =    together.Complete.create(
                    #model = "mistralai/Mixtral-8x7B-Instruct-v0.1", 
                    #model="cognitivecomputations/dolphin-2.5-mixtral-8x7b",
                    #model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",   
                    #model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT",   
                    #model="Qwen/Qwen1.5-14B-Chat",
                    prompt=instruction.format(parameters = self.parameters, user_prompt = self.user_q),
                    #model="Qwen/Qwen2-72B-Instruct",
                    model="meta-llama/Llama-3-70b-chat-hf",
                    #model="meta-llama/Llama-3-8b-chat-hf",
                    #model="deepseek-ai/deepseek-coder-33b-instruct",
                    #prompt=prompt_3,
                    #prompt=PROMPT_SATORI_FINAL_L.format(query=query, hist=t2),
                    temperature=0.1,
                    top_p=0.85,
                    top_k=40,
                    max_tokens=128,  
                    repetition_penalty=1.1,
                    #stop = ["</s>", "[INST]", "<|im_end|>"]
                    stop = ["</s>", "[INST]", "User:", "assistant"]#, "`", "` "] #, '"', "\"\n\n", "</json>",'"\n']#, "01", "_v01"]
            )
        #print("\n")
        print(f"Prompt prep clash:{instruction}")
        print("*"*25)
        out_long = output['output']['choices'][0]['text']
        print(out_long)
        print(output["output"]["usage"])
        print("*"*25)
        return out_long    
    
    async def _arun_clash(
        self,
        parameters: dict,
    ):
        # Execute tool
        """Use the tool."""
        try:

            # Date and query params 
            #params = calculate_datetime_ranges(self.timezone)
            #params["query"] = parameters
            params={}
            #params["hist"] = self.hist
            '''
            adv_query = await async_chain_together_m(
                prompt=PROMPT_ADVANCED_QUERY_PULL_GITHUB,
                params=params,
                max_tokens=768,
                temperature=0.1,
                top_p=1,
                top_k=50,
                repetition_penalty=1.1,
                model="meta-llama/Llama-3-70b-chat-hf",
                stop = ["</s>", "[INST]", "<|EOT|>", '"'],
                tool_name="ClickUp Advanced Query Search",
                callback_manager=self.async_callback_manager
                )
            '''
            # Log step clickup improver
            #if self.verbose:
            #    step = {"type": "clickup_query", "properties": {"query": adv_query}}
            #    await self.async_callback_manager.queue.put(f"#STEP{json.dumps(step)}STEP#")
            # Run the task extractor
            #pulls = await self._arun(query=adv_query)
            # Log step clickup improver
            #if self.verbose:
            #    step = {"type": f"{self.name}", "results": tasks}
            #    await self.async_callback_manager.queue.put(f"#STEP_OUT{json_step.dumps(step, ignore_nan=True)}STEP_OUT#")
            
            parameters["repo_owner"] = self.repo_owner
            parameters["repo_name"] = self.repo_name
            parameters["token"] = self.token
            parameters["pull_number"] = self.pull_number
            
            '''
            token = os.getenv("GITHUB_TOKEN")
            if not token:
                raise ValueError("Token is missing or empty")
            parameters["token"] = token.strip()
            '''
            merge_result = self.pr.update_pull_request(**parameters)
            
            parameters.pop("token")
            if "status" not in merge_result.keys(): # status means an error ocurred when submitting review
                #print("No pulls found")
                #print(f"Found {len(pr_list)} pulls")
                extra_export = ", finally, you must notify the user the tasks were exported into a csv file inside the session folder, this can be located on the right menu" if self.export else ""
                answer_inst = f'''{self.ANSWER_SYSTEM_SUCCESS}<|start_header_id|>user<|end_header_id|>
                Report succinctly to the user that the update operation was executed succesfully.<|eot_id|><|start_header_id|>assistant<|end_header_id|>'''
                    
            else:
                #print(pr_list)
                #print(f"Could not perform read pull operation:\nMessage: {pr_list['message']}\nError code: {pr_list['status']}")
                answer_inst = f'''{self.ANSWER_SYSTEM_ERROR}<|start_header_id|>user<|end_header_id|>
                An error ocurred when attempting to submit the merge request. The most common errors are:\n```{self.error_codes}```\n.The message error is {merge_result['message']} and the error code is {merge_result['status']}.\n Explain what the possible cause of the error is and suggest a solution.<|eot_id|><|start_header_id|>assistant<|end_header_id|>'''              
            '''
            params = {
                "repo_owner": self.repo_owner,                
                "repo_name": self.repo_name,
                "hist": self.hist,
                #"lang": self.lang,
                "task_answer_instruction": answer_inst,
                #"query": self.user_msg,
                #"date": datetime.now(pytz.timezone(self.timezone)).strftime("%A, %d of %B with time %H:%M of %Y"),
                "init_lang": "Por supuesto!\n" if self.lang == "Spanish" else "Certainly!\n"
            }
            
            # Stream
            
            result = await achain_nirvana(
            prompt=general_prompt,
            params=params,
            max_tokens=2048,
            temperature=0.1,
            top_p=0.85,
            top_k=40,
            repetition_penalty=1.1,
            model = "together|meta-llama/Llama-3-70b-chat-hf",
            stop = ["</s>", "[INST]", "<|EOT|>", "<|eot_id|>"],
            tool_name="SQL Generation",
            stream=False,
            )
            
            msg_exec = await stream_chain_together_m(
                prompt=PROMPT_CLICKUP_ANSWER_SATORI,
                params=params,
                max_tokens=2048,
                callback_manager=self.async_callback_manager,
                model="meta-llama/Llama-3-8b-chat-hf",
                stop=["</s>", "[INST]", "<|im_end|>"],
                tool_name="ClickUp Tasks Answer",
            )
            '''
            
            output =  together.Complete.create(
                    #model = "mistralai/Mixtral-8x7B-Instruct-v0.1", 
                    #model="cognitivecomputations/dolphin-2.5-mixtral-8x7b",
                    model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",   
                    #model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT",   
                    #model="Qwen/Qwen1.5-14B-Chat",
                    prompt=answer_inst,
                    #model="Qwen/Qwen2-72B-Instruct",
                    #model="meta-llama/Llama-3-70b-chat-hf",
                    #model="meta-llama/Llama-3-8b-chat-hf",
                    #model="deepseek-ai/deepseek-coder-33b-instruct",
                    #prompt=prompt_3,
                    #prompt=PROMPT_SATORI_FINAL_L.format(query=query, hist=t2),
                    temperature=0.1,
                    top_p=0.85,
                    top_k=40,
                    max_tokens=128,  
                    repetition_penalty=1.1,
                    #stop = ["</s>", "[INST]", "<|im_end|>"]
                    stop = ["</s>", "[INST]", "User:", "assistant"]#, "`", "` "] #, '"', "\"\n\n", "</json>",'"\n']#, "01", "_v01"]
            )
            #print("\n")
            print(f"Prompt run clash (answer inst):{answer_inst}")
            out_long = output['output']['choices'][0]['text']
           # print(f"Answer prompt output:\n{out_long}")
            print(output["output"]["usage"])
            print("*"*25)
            # Export
            if self.export:
                await self.async_callback_manager.queue.put(f"\nExport: [{self.user_msg}]({tasks['export_file']})")
            return {"finish": True, "response": output['output']['choices'][0]['text']}
        except Exception as e_tasks_clickup:
            ic(e_tasks_clickup)
            spaces = '- ' + '\n- '.join([f'{x}: {y}' for x, y in self.clickup_dicts['spaces'].items()])
            trace = traceback.format_exc()
            return {"error": "No tasks could be extracted.", "raise_msg": f"Is it likely the query used and the search parameters are not correct. Explain this query to the user in natural language and ask for feedback. These are the spaces available:\n{spaces}\nQuestion: {query}\nSearch query: {adv_query}", "trace": trace}

class GithubAnalyzeDiff():
    
    def __init__(self, diff, user_prompt):
        self.patch = diff.get('patch', 'No changes')
        self.filename = diff.get('filename', '')
        self.deletions = diff.get('deletions', 0)
        self.additions = diff.get('additions', 0)
        self.changes = diff.get('changes', 0)
        self.user_prompt = user_prompt
        
    
    ANSWER_SYSTEM= '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a helpful AI assistant, knowledgeable about the Github API. You will be reporting to the user about the information yielded by a previous pull request query. Specifically, you will receive a diff patch containing the modifications made to files inside a given pull request. Your task is to give a concise report about the changes made to the file file.<|eot_id|>
    '''
    api_key = os.getenv("TOGETHERAI_MIXTRAL_API_KEY")
    if not api_key:
        raise ValueError("API key is missing or empty")
    together.api_key = api_key.strip()
    
    async def _aprep_clash(self,
        chat_history: list, 
        #agent_type: str, 
        #user_p: str, 
        #q_len: int
    ):
        
        self.hist = chat_history
    
        self.user_prompt+= f"\nThe changes are as follows:\n{self.patch}"
        instruction = '''
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a helpful AI assistant, knowledgeable about the Github API. You will be reporting to the user about the information yielded by a previous pull request query. Specifically, you will receive a diff patch containing the modifications made to a file inside a given pull request. Your task is to give a concise report about the changes made. Your answer must begin with the filename, and number of additions, deletions and changes. Then it must be a numbered list of the main modifications, each with a description. If there are no changes, state that the file is new.<|eot_id|><|start_header_id|>user<|end_header_id|>
        The filename is {filename}. Additions: {additions}, deletions: {deletions}, changes: {changes}. This is the diff patch for the file:´´´\n{user_prompt}\n´´´<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        '''
        
        universal_prompt='''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a helpful AI assistant, knowledgeable about the Github API. You will receive an instruction to perform certain analysis or synthesis task over information related to pull requests. Assume the user has a grasp on basic git concepts. Your answers must be concise and only contain critical information to the task at hand.<|eot_id|><|start_header_id|>user<|end_header_id|>
        {user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        '''
        
        #example = "- Search for open pull requests with base branch tester\n- Search for all pull requests with head octocat:new-branch, sort them in ascending order."
        #user_p ="List page 2 of the pull request with 'tester' as base branch"
        #user_p = "List the fourth page of the file changes made to 675"
        #field = ['state="', 'head="', 'base="', 'sort="', 'direction="', 'page="', 'per_page="']
        output =    together.Complete.create(
                    #model = "mistralai/Mixtral-8x7B-Instruct-v0.1", 
                    #model="cognitivecomputations/dolphin-2.5-mixtral-8x7b",
                    #model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",   
                    #model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT",   
                    #model="Qwen/Qwen1.5-14B-Chat",
                    #prompt=instruction.format(filename = self.filename, additions = self.additions, deletions = self.deletions, changes = self.changes, user_prompt = self.user_prompt),
                    prompt=universal_prompt.format(user_prompt=self.user_prompt),
                    #model="Qwen/Qwen2-72B-Instruct",
                    model="meta-llama/Llama-3-70b-chat-hf",
                    #model="meta-llama/Llama-3-8b-chat-hf",
                    #model="deepseek-ai/deepseek-coder-33b-instruct",
                    #prompt=prompt_3,
                    #prompt=PROMPT_SATORI_FINAL_L.format(query=query, hist=t2),
                    temperature=0.1,
                    top_p=0.85,
                    top_k=40,
                    max_tokens=128,  
                    repetition_penalty=1.1,
                    #stop = ["</s>", "[INST]", "<|im_end|>"]
                    stop = ["</s>", "[INST]", "User:", "assistant"]#, "`", "` "] #, '"', "\"\n\n", "</json>",'"\n']#, "01", "_v01"]
            )
        #print("\n")
        print(f"Prompt unico usado:{instruction}")
        print("*"*25)
        out_long = output['output']['choices'][0]['text']
        print(out_long)
        print(output["output"]["usage"])
        print("*"*25)
        return out_long

class GithubListPullRequest():
#class GithubListPullRequest(BaseTool):
    """_summary_

    Args:
        BaseTool (_type_): Base class for tools in OpenAI

    Raises:
        NotImplementedError: _description_
        NotImplementedError: _description_

    Returns:
        _type_: agent answer for query
    """
    
    def __init__(self, token, repo_owner, repo_name, user_q):
        self.token = token
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.user_q = user_q
    
    parameters = '''| Parameter         | Type    | Required | Description                                                                                                      |
            |-------------------|---------|----------|------------------------------------------------------------------------------------------------------------------|
            | `state`           | string  | No       | Indicates the state of the pull request. Can be either `open`, `closed`, or `all`. Default is `open`.             |
            | `head`            | string  | No       | Filter pulls by head user and branch name in the format of `user:ref-name`. Example: `github:new-script-format`.   |
            | `base`            | string  | No       | Filter pulls by base branch name. Example: `main`.                                                               |
            | `sort`            | string  | No       | What to sort results by. Can be either `created`, `updated`, `popularity`, or `long-running`. Default is `created`.|
            | `direction`       | string  | No       | The direction of the sort. Can be either `asc` or `desc`. Default is `desc`.                                     |
            | `per_page`        | integer | No       | The number of results per page (max 100). Default is 30.                                                         |
            | `page`            | integer | No       | Page number of the results to fetch. Default is 1.     															|
    
    '''
        
    error_codes = '''
    | Status Code | Error Message                  | Description                                                                 |
    |--------------|--------------------------------|-----------------------------------------------------------------------------|
    | 200          | OK                             | The request was successful.                                                 |
    | 301          | Moved Permanently              | The resource has been moved to a new URL.                                   |
    | 304          | Not Modified                   | There was no new data to return.                                            |
    | 400          | Bad Request                    | The request was invalid or cannot be served.                                |
    | 401          | Unauthorized                   | Authentication is required and has failed or has not yet been provided.     |
    | 403          | Forbidden                      | The request is understood, but it has been refused or access is not allowed.|
    | 404          | Not Found                      | The requested resource could not be found.                                  |
    | 422          | Unprocessable Entity           | The request was well-formed but was unable to be followed due to semantic errors.|
    | 500          | Internal Server Error          | An error occurred on the server.                                            |
    | 502          | Bad Gateway                    | The server was acting as a gateway or proxy and received an invalid response from the upstream server.|
    | 503          | Service Unavailable            | The server is currently unavailable (because it is overloaded or down for maintenance).|
    | 504          | Gateway Timeout                | The server was acting as a gateway or proxy and did not receive a timely response from the upstream server.|
    '''
        
    ANSWER_SYSTEM_SUCCESS = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a helpful AI assistant, knowledgeable about the Github API. You will be reporting to the user about the information yielded by a previous pull request query. You will receive a table with a summary of the findings, copy it in your answer.<|eot_id|>
    '''

    ANSWER_SYSTEM_ERROR = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a helpful AI assistant, knowledgeable about the Github API. You will be reporting to the user about the information yielded by a previous pull request query. You will receive an error code and message, your task is to explain the error to the user and provide suggestions to fix it. <|eot_id|>
    '''
    ANSWER_SYSTEM_NO_FINDINGS = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a helpful AI assistant, knowledgeable about the Github API. You will be reporting to the user about the information yielded by a previous pull request query. The pull request will be correctly executed, but it will yield no results given the filters applied. Provide a short explanation about what changes could be made for the query parameters to yield results. <|eot_id|>

    '''
    #repo_owner = "qgis"
    #repo_name = "QGIS"
    pr = PullRequests()
    session_path: str
    clickup_api: str
    workspace_id: int
    clickup_dicts: Dict[str, Dict]
    top_k: int = 10
    username: str
    user_id: str
    single = False
    user_msg: Optional[str] = ""
    export = False
    async_callback_manager: Optional[Any] = None
    lang: Optional[str] = "English"
    timezone: str
    verbose=False
    hist = ""
    
    api_key = os.getenv("TOGETHERAI_MIXTRAL_API_KEY")
    if not api_key:
        raise ValueError("API key is missing or empty")
    together.api_key = api_key.strip()
    


    
    async def _aprep_clash(self,
            chat_history: list, 
            #agent_type: str, 
            #user_p: str, 
            #q_len: int
        ):
        """Clash prepartion for run 

        Args:
            messages (str): Agent parsed memory
            agent_type (str): agent type string
            user_p (str): current query
        """
        """
        # Extract intention
        loop = asyncio.get_event_loop()
        self.hist = chat_history
        # Classify
        export = await async_chain_together_m(
                prompt=,
                params={"chat_history": parse_mixtral_memory(self.hist), "query": user_p},
                max_tokens=256,
                temperature=0.1,
                top_p=1,
                top_k=25,
                repetition_penalty=1.1,
                callback_manager=self.async_callback_manager,
                model="meta-llama/Llama-3-8b-chat-hf",
                stop = ["</s>", "[INST]", "<|EOT|>", '"'],
                tool_name=""
            )
        
        """
        #loop = asyncio.get_event_loop()
        self.hist = chat_history
    
 
        instruction = '''
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a helpful AI assistant, knowledgeable about the Github API. You will be asked to read text and extract parameter values used for queries to the Github API. Specifically, the queries will be related to Pull Requests. The following contains all of the parameters you will need to look for inside the text:\n{parameters}\n

            Omit every parameter that is not mentioned. If no parameters are mentioned, return empty braces '{{}}'. Your answer must only include the relevant field name and its value in a json format. <|eot_id|><|start_header_id|>user<|end_header_id|>
            List all of the open pull requests and sort them in descending order<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            {{"state":"open", "direction":"desc"}}<|eot_id|><|start_header_id|>user<|end_header_id|>
            Get page 2 of the pull request with 'tester' as base branch<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            {{"page":2,\n"base":"tester"}}<|eot_id|><|start_header_id|>user<|end_header_id|>
            Display all pull requests<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            {{}}<|eot_id|><|start_header_id|>user<|end_header_id|>
            {user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|> 
            '''
        
        #example = "- Search for open pull requests with base branch tester\n- Search for all pull requests with head octocat:new-branch, sort them in ascending order."
        #user_p ="List page 2 of the pull request with 'tester' as base branch"
        #user_p = "List all closed pull requests with base branch sysiphus, sort them by longest running"
        #field = ['state="', 'head="', 'base="', 'sort="', 'direction="', 'page="', 'per_page="']
        output =    together.Complete.create(
                    #model = "mistralai/Mixtral-8x7B-Instruct-v0.1", 
                    #model="cognitivecomputations/dolphin-2.5-mixtral-8x7b",
                    #model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",   
                    #model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT",   
                    #model="Qwen/Qwen1.5-14B-Chat",
                    prompt=instruction.format(parameters = self.parameters, user_prompt = self.user_q),
                    #model="Qwen/Qwen2-72B-Instruct",
                    model="meta-llama/Llama-3-70b-chat-hf",
                    #model="meta-llama/Llama-3-8b-chat-hf",
                    #model="deepseek-ai/deepseek-coder-33b-instruct",
                    #prompt=prompt_3,
                    #prompt=PROMPT_SATORI_FINAL_L.format(query=query, hist=t2),
                    temperature=0.1,
                    top_p=0.85,
                    top_k=40,
                    max_tokens=128,  
                    repetition_penalty=1.1,
                    #stop = ["</s>", "[INST]", "<|im_end|>"]
                    stop = ["</s>", "[INST]", "User:", "assistant"]#, "`", "` "] #, '"', "\"\n\n", "</json>",'"\n']#, "01", "_v01"]
            )
        #print("\n")
        print(f"Prompt prep clash:{instruction}")
        print("*"*25)
        out_long = output['output']['choices'][0]['text']
        print(out_long)
        print(output["output"]["usage"])
        print("*"*25)

        self.user_msg = output
        # Return json params
        return {"parameters": output['output']['choices'][0]['text']}
    
    async def _arun_clash(
            self,
            parameters: dict,
    ):
        # Execute tool
        """Use the tool."""
        try:

            # Date and query params 
            #params = calculate_datetime_ranges(self.timezone)
            #params["query"] = parameters
            params={}
            params["hist"] = self.hist
            '''
            adv_query = await async_chain_together_m(
                prompt=PROMPT_ADVANCED_QUERY_PULL_GITHUB,
                params=params,
                max_tokens=768,
                temperature=0.1,
                top_p=1,
                top_k=50,
                repetition_penalty=1.1,
                model="meta-llama/Llama-3-70b-chat-hf",
                stop = ["</s>", "[INST]", "<|EOT|>", '"'],
                tool_name="ClickUp Advanced Query Search",
                callback_manager=self.async_callback_manager
                )
            '''
            # Log step clickup improver
            #if self.verbose:
            #    step = {"type": "clickup_query", "properties": {"query": adv_query}}
            #    await self.async_callback_manager.queue.put(f"#STEP{json.dumps(step)}STEP#")
            # Run the task extractor
            #pulls = await self._arun(query=adv_query)
            # Log step clickup improver
            #if self.verbose:
            #    step = {"type": f"{self.name}", "results": tasks}
            #    await self.async_callback_manager.queue.put(f"#STEP_OUT{json_step.dumps(step, ignore_nan=True)}STEP_OUT#")
            
            parameters["repo_owner"] = self.repo_owner
            parameters["repo_name"] = self.repo_name
            parameters["token"] = self.token
            '''
            token = os.getenv("GITHUB_TOKEN")
            if not token:
                raise ValueError("Token is missing or empty")
            parameters["token"] = token.strip()
            '''
            pr_list = self.pr.list_all(**parameters)
            
            parameters.pop("token")
            if isinstance(pr_list, list):
                pr_number = len(pr_list)
                if len(pr_list) == 0:
                    answer_inst = f'''{self.ANSWER_SYSTEM_NO_FINDINGS}<|start_header_id|>user<|end_header_id|>
                    A total of 0 pull requests were retrieved. The parameters used for the query were {', '.join(f'{key}: {value}' for key, value in parameters.items())}.\n Explain what filters were used and ask for modifications that could yield results.<|eot_id|><|start_header_id|>assistant<|end_header_id|>'''
                    #print("No pulls found")
                else:
                    #print(f"Found {len(pr_list)} pulls")
                    extra_export = ", finally, you must notify the user the tasks were exported into a csv file inside the session folder, this can be located on the right menu" if self.export else ""
                    answer_inst = f'''{self.ANSWER_SYSTEM_SUCCESS}<|start_header_id|>user<|end_header_id|>
                    A total of {pr_number} pull requests were extracted.\n```{self.pr.pull_requests_to_markdown(pr_list)}```\n.<|eot_id|><|start_header_id|>assistant<|end_header_id|>'''
                    
            elif isinstance(pr_list, dict):
                #print(pr_list)
                #print(f"Could not perform read pull operation:\nMessage: {pr_list['message']}\nError code: {pr_list['status']}")
                answer_inst = f'''{self.ANSWER_SYSTEM_ERROR}<|start_header_id|>user<|end_header_id|>
                An error ocurred when attempting to retrieve the pull requests. The most common errors are:\n```{self.error_codes}```\n.The message error is {pr_list['message']} and the error code is {pr_list['status']}.\n Explain what the possible cause of the error is and suggest a solution.<|eot_id|><|start_header_id|>assistant<|end_header_id|>'''              
            
            params = {
                "repo_owner": self.repo_owner,                
                "repo_name": self.repo_name,
                "hist": self.hist,
                "lang": self.lang,
                "task_answer_instruction": answer_inst,
                "query": self.user_msg,
                #"date": datetime.now(pytz.timezone(self.timezone)).strftime("%A, %d of %B with time %H:%M of %Y"),
                "init_lang": "Por supuesto!\n" if self.lang == "Spanish" else "Certainly!\n"
            }
            
            # Stream
            '''
            result = await achain_nirvana(
            prompt=general_prompt,
            params=params,
            max_tokens=2048,
            temperature=0.1,
            top_p=0.85,
            top_k=40,
            repetition_penalty=1.1,
            model = "together|meta-llama/Llama-3-70b-chat-hf",
            stop = ["</s>", "[INST]", "<|EOT|>", "<|eot_id|>"],
            tool_name="SQL Generation",
            stream=False,
            )
            
            msg_exec = await stream_chain_together_m(
                prompt=PROMPT_CLICKUP_ANSWER_SATORI,
                params=params,
                max_tokens=2048,
                callback_manager=self.async_callback_manager,
                model="meta-llama/Llama-3-8b-chat-hf",
                stop=["</s>", "[INST]", "<|im_end|>"],
                tool_name="ClickUp Tasks Answer",
            )
            '''
            
            output =  together.Complete.create(
                    #model = "mistralai/Mixtral-8x7B-Instruct-v0.1", 
                    #model="cognitivecomputations/dolphin-2.5-mixtral-8x7b",
                    model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",   
                    #model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT",   
                    #model="Qwen/Qwen1.5-14B-Chat",
                    prompt=answer_inst,
                    #model="Qwen/Qwen2-72B-Instruct",
                    #model="meta-llama/Llama-3-70b-chat-hf",
                    #model="meta-llama/Llama-3-8b-chat-hf",
                    #model="deepseek-ai/deepseek-coder-33b-instruct",
                    #prompt=prompt_3,
                    #prompt=PROMPT_SATORI_FINAL_L.format(query=query, hist=t2),
                    temperature=0.1,
                    top_p=0.85,
                    top_k=40,
                    max_tokens=128,  
                    repetition_penalty=1.1,
                    #stop = ["</s>", "[INST]", "<|im_end|>"]
                    stop = ["</s>", "[INST]", "User:", "assistant"]#, "`", "` "] #, '"', "\"\n\n", "</json>",'"\n']#, "01", "_v01"]
            )
            #print("\n")
            print(f"Prompt run clash (answer inst):{answer_inst}")
            out_long = output['output']['choices'][0]['text']
           # print(f"Answer prompt output:\n{out_long}")
            print(output["output"]["usage"])
            print("*"*25)
            # Export
            if self.export:
                await self.async_callback_manager.queue.put(f"\nExport: [{self.user_msg}]({tasks['export_file']})")
            return {"finish": True, "response": output['output']['choices'][0]['text']}
        except Exception as e_tasks_clickup:
            ic(e_tasks_clickup)
            spaces = '- ' + '\n- '.join([f'{x}: {y}' for x, y in self.clickup_dicts['spaces'].items()])
            trace = traceback.format_exc()
            return {"error": "No tasks could be extracted.", "raise_msg": f"Is it likely the query used and the search parameters are not correct. Explain this query to the user in natural language and ask for feedback. These are the spaces available:\n{spaces}\nQuestion: {query}\nSearch query: {adv_query}", "trace": trace}

class GithubDiffPullRequest():
    
    def __init__(self, token, repo_owner, repo_name, user_q):
        self.token = token
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.pr = PullRequests()
        self.lang: Optional[str] = "English"
        self.export: Optional[str] = False
        self.user_q = user_q
        #self.pull_id = pull_id        
    
    parameters = '''
    | Parameter      | Type     | Description                                                              | Required |
    |----------------|----------|--------------------------------------------------------------------------|----------|
    | pull_number    | integer  | The number of the pull request.                                           | Yes      |
    | page           | integer  | Page number of the results to fetch.                                      | No       |
    | per_page       | integer  | Number of results per page (max 100).       
    '''
    
    error_codes = '''
    | Status Code | Error Message                  | Description                                                                 |
    |--------------|--------------------------------|-----------------------------------------------------------------------------|
    | 200          | OK                             | The request was successful.                                                 |
    | 301          | Moved Permanently              | The resource has been moved to a new URL.                                   |
    | 304          | Not Modified                   | There was no new data to return.                                            |
    | 400          | Bad Request                    | The request was invalid or cannot be served.                                |
    | 401          | Unauthorized                   | Authentication is required and has failed or has not yet been provided.     |
    | 403          | Forbidden                      | The request is understood, but it has been refused or access is not allowed.|
    | 404          | Not Found                      | The requested resource could not be found.                                  |
    | 422          | Unprocessable Entity           | The request was well-formed but was unable to be followed due to semantic errors.|
    | 500          | Internal Server Error          | An error occurred on the server.                                            |
    | 502          | Bad Gateway                    | The server was acting as a gateway or proxy and received an invalid response from the upstream server.|
    | 503          | Service Unavailable            | The server is currently unavailable (because it is overloaded or down for maintenance).|
    | 504          | Gateway Timeout                | The server was acting as a gateway or proxy and did not receive a timely response from the upstream server.|        
    '''
    
    ANSWER_SYSTEM_SUCCESS = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a helpful AI assistant, knowledgeable about the Github API. You will be reporting to the user about the information yielded by a previous pull request query. You will receive a table with a summary of the findings, copy it in your answer.<|eot_id|>
    '''

    ANSWER_SYSTEM_ERROR = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a helpful AI assistant, knowledgeable about the Github API. You will be reporting to the user about the information yielded by a previous pull request query. You will receive an error code and message, your task is to explain the error to the user and provide suggestions to fix it. <|eot_id|>
    '''
    
    api_key = os.getenv("TOGETHERAI_MIXTRAL_API_KEY")
    if not api_key:
        raise ValueError("API key is missing or empty")
    together.api_key = api_key.strip()
    

    def diff_files_to_markdown(self, diff_files):
        markdown = "| Filename | Status | Additions | Deletions | Changes |\n"
        markdown += "|----------|--------|-----------|-----------|---------|\n"
        
        for file in diff_files:
            filename = file['filename']
            status = file['status']
            additions = file['additions']
            deletions = file['deletions']
            changes = file['changes']
            
            markdown += f"| {filename} | {status} | {additions} | {deletions} | {changes} |\n"
        
        return markdown
    
    async def _aprep_clash(self,
            chat_history: list, 
            #agent_type: str, 
            #user_p: str, 
            #q_len: int
        ):
        
        self.hist = chat_history
    
 
        instruction = '''
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a helpful AI assitant, knowledgeable about the Github API. You will be asked to read text and extract parameter values used for queries to the Github API. Specifically, the queries will be related to Pull Requests. The following contains all of the parameters you will need to look for inside the text:\n{parameters}
            Omit every parameter that is not mentioned. If no parameters are mentioned, return empty braces '{{}}'. Your answer must only include the relevant field name and its value in a json format. <|eot_id|><|start_header_id|>user<|end_header_id|>
            Show me the modified files for 56741<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            {{"pull_number": 56741}}<|eot_id|><|start_header_id|>user<|end_header_id|>
            Get the second page for the file changes made to 12<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            {{"pull_number":12, "page": 2}}<|eot_id|><|start_header_id|>user<|end_header_id|>
            Display the first 10 file changes made in 20 <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            {{"pull_number":20, "per_page": 20}}<|eot_id|><|start_header_id|>user<|end_header_id|>
            {user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            
            '''
        
        #example = "- Search for open pull requests with base branch tester\n- Search for all pull requests with head octocat:new-branch, sort them in ascending order."
        #user_p ="List page 2 of the pull request with 'tester' as base branch"
        #user_p = "List the fourth page of the file changes made to 675"
        #field = ['state="', 'head="', 'base="', 'sort="', 'direction="', 'page="', 'per_page="']
        output =    together.Complete.create(
                    #model = "mistralai/Mixtral-8x7B-Instruct-v0.1", 
                    #model="cognitivecomputations/dolphin-2.5-mixtral-8x7b",
                    #model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",   
                    #model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT",   
                    #model="Qwen/Qwen1.5-14B-Chat",
                    prompt=instruction.format(parameters= self.parameters, user_prompt = self.user_q),
                    #model="Qwen/Qwen2-72B-Instruct",
                    model="meta-llama/Llama-3-70b-chat-hf",
                    #model="meta-llama/Llama-3-8b-chat-hf",
                    #model="deepseek-ai/deepseek-coder-33b-instruct",
                    #prompt=prompt_3,
                    #prompt=PROMPT_SATORI_FINAL_L.format(query=query, hist=t2),
                    temperature=0.1,
                    top_p=0.85,
                    top_k=40,
                    max_tokens=128,  
                    repetition_penalty=1.1,
                    #stop = ["</s>", "[INST]", "<|im_end|>"]
                    stop = ["</s>", "[INST]", "User:", "assistant"]#, "`", "` "] #, '"', "\"\n\n", "</json>",'"\n']#, "01", "_v01"]
            )
        #print("\n")
        out_long = output['output']['choices'][0]['text']
        print(f"Prompt prep clash:{instruction}")
        #print(out_long)
        #print(output["output"]["usage"])
        print("*"*25)
        return out_long
    
    async def _arun_clash(
        self,
        parameters: dict,
    ):
        # Execute tool
        """Use the tool."""
        try:

            # Date and query params 
            #params = calculate_datetime_ranges(self.timezone)
            #params["query"] = parameters
            params={}
            params["hist"] = self.hist
            '''
            adv_query = await async_chain_together_m(
                prompt=PROMPT_ADVANCED_QUERY_PULL_GITHUB,
                params=params,
                max_tokens=768,
                temperature=0.1,
                top_p=1,
                top_k=50,
                repetition_penalty=1.1,
                model="meta-llama/Llama-3-70b-chat-hf",
                stop = ["</s>", "[INST]", "<|EOT|>", '"'],
                tool_name="ClickUp Advanced Query Search",
                callback_manager=self.async_callback_manager
                )
            '''
            # Log step clickup improver
            #if self.verbose:
            #    step = {"type": "clickup_query", "properties": {"query": adv_query}}
            #    await self.async_callback_manager.queue.put(f"#STEP{json.dumps(step)}STEP#")
            # Run the task extractor
            #pulls = await self._arun(query=adv_query)
            # Log step clickup improver
            #if self.verbose:
            #    step = {"type": f"{self.name}", "results": tasks}
            #    await self.async_callback_manager.queue.put(f"#STEP_OUT{json_step.dumps(step, ignore_nan=True)}STEP_OUT#")
            
            parameters["repo_owner"] = self.repo_owner
            parameters["repo_name"] = self.repo_name
            parameters["token"] = self.token
            '''
            token = os.getenv("GITHUB_TOKEN")
            if not token:
                raise ValueError("Token is missing or empty")
            parameters["token"] = token.strip()
            '''
            pr_diff = self.pr.diff(**parameters)
            #print(f"PR Diff result:{pr_diff}")
            parameters.pop("token")
            if isinstance(pr_diff, list):
                pr_number = len(pr_diff)

                #print(f"Found {len(pr_list)} pulls")
                extra_export = ", finally, you must notify the user the tasks were exported into a csv file inside the session folder, this can be located on the right menu" if self.export else ""
                answer_inst = f'''{self.ANSWER_SYSTEM_SUCCESS}<|start_header_id|>user<|end_header_id|>
                A total of {pr_number} files were changed.\n```{self.diff_files_to_markdown(pr_diff)}```\n.<|eot_id|><|start_header_id|>assistant<|end_header_id|>'''
                    
            elif isinstance(pr_diff, dict):
                #print(pr_list)
                #print(f"Could not perform read pull operation:\nMessage: {pr_list['message']}\nError code: {pr_list['status']}")
                answer_inst = f'''{self.ANSWER_SYSTEM_ERROR}<|start_header_id|>user<|end_header_id|>
                An error ocurred when attempting to retrieve the diff files. The most common errors are:\n```{self.error_codes}```\n.The message error is {pr_diff['message']} and the error code is {pr_diff['status']}.\n Explain what the possible cause of the error is and suggest a solution.<|eot_id|><|start_header_id|>assistant<|end_header_id|>'''              
            
            params = {
                "repo_owner": self.repo_owner,                
                "repo_name": self.repo_name,
                "hist": self.hist,
                "lang": self.lang,
                "task_answer_instruction": answer_inst,
                #"query": self.user_msg,
                #"date": datetime.now(pytz.timezone(self.timezone)).strftime("%A, %d of %B with time %H:%M of %Y"),
                "init_lang": "Por supuesto!\n" if self.lang == "Spanish" else "Certainly!\n"
            }
            
            # Stream
            '''
            result = await achain_nirvana(
            prompt=general_prompt,
            params=params,
            max_tokens=2048,
            temperature=0.1,
            top_p=0.85,
            top_k=40,
            repetition_penalty=1.1,
            model = "together|meta-llama/Llama-3-70b-chat-hf",
            stop = ["</s>", "[INST]", "<|EOT|>", "<|eot_id|>"],
            tool_name="SQL Generation",
            stream=False,
            )
            
            '''
            
            output =  together.Complete.create(
                    #model = "mistralai/Mixtral-8x7B-Instruct-v0.1", 
                    #model="cognitivecomputations/dolphin-2.5-mixtral-8x7b",
                    #model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",   
                    #model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT",   
                    #model="Qwen/Qwen1.5-14B-Chat",
                    prompt=answer_inst,
                    #model="Qwen/Qwen2-72B-Instruct",
                    #model="meta-llama/Llama-3-70b-chat-hf",
                    model="meta-llama/Llama-3-8b-chat-hf",
                    #model="deepseek-ai/deepseek-coder-33b-instruct",
                    #prompt=prompt_3,
                    #prompt=PROMPT_SATORI_FINAL_L.format(query=query, hist=t2),
                    temperature=0.1,
                    top_p=0.85,
                    top_k=40,
                    max_tokens=128,  
                    repetition_penalty=1.1,
                    #stop = ["</s>", "[INST]", "<|im_end|>"]
                    stop = ["</s>", "[INST]", "User:", "assistant"]#, "`", "` "] #, '"', "\"\n\n", "</json>",'"\n']#, "01", "_v01"]
            )
            #print("\n")
            out_long = output['output']['choices'][0]['text']
            print(f"Prompt run clash:{answer_inst}")
            #print(f"Answer prompt output:\n{out_long}")
            #print(output["output"]["usage"])
            print("*"*25)
            # Export
            if self.export:
                await self.async_callback_manager.queue.put(f"\nExport: [{self.user_msg}]({tasks['export_file']})")
            return {"finish": True, "response": output['output']['choices'][0]['text'], "diff": pr_diff}
        except Exception as e_tasks_clickup:
            ic(e_tasks_clickup)
            spaces = '- ' + '\n- '.join([f'{x}: {y}' for x, y in self.clickup_dicts['spaces'].items()])
            trace = traceback.format_exc()
            return {"error": "No tasks could be extracted.", "raise_msg": f"Is it likely the query used and the search parameters are not correct. Explain this query to the user in natural language and ask for feedback. These are the spaces available:\n{spaces}\nQuestion: {query}\nSearch query: {adv_query}", "trace": trace}

class GithubCommit():
    
    def __init__(self, token, repo_owner, repo_name, user_q):
        self.token = token
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.commit = Commits()
        self.lang: Optional[str] = "English"
        self.export: Optional[str] = False
        self.user_q = user_q
        #self.pull_id = pull_id     
        self.parameters = '''
        | Parameter      | Type    | Description                                                              | Required/Optional |
        |----------------|---------|--------------------------------------------------------------------------|-------------------|
        | sha            | string  | SHA or branch to start listing commits from.                             | Optional          |
        | path           | string  | Only commits containing this file path will be returned.                 | Optional          |
        | author         | string  | GitHub login or email address by which to filter by commit author.       | Optional          |
        | since          | string  | Only commits after this date will be returned (ISO 8601 format).         | Optional          |
        | until          | string  | Only commits before this date will be returned (ISO 8601 format).        | Optional          |
        | per_page       | integer | Number of results per page (max 100).                                     | Optional          |
        | page           | integer | Page number of the results to fetch.                                      | Optional          |
        '''
        
        self.error_code = '''
        | Status Code | Error Message                  | Description                                                                 |
        |--------------|--------------------------------|-----------------------------------------------------------------------------|
        | 200          | OK                             | The request was successful.                                                 |
        | 301          | Moved Permanently              | The resource has been moved to a new URL.                                   |
        | 304          | Not Modified                   | There was no new data to return.                                            |
        | 400          | Bad Request                    | The request was invalid or cannot be served.                                |
        | 401          | Unauthorized                   | Authentication is required and has failed or has not yet been provided.     |
        | 403          | Forbidden                      | The request is understood, but it has been refused or access is not allowed.|
        | 404          | Not Found                      | The requested resource could not be found.                                  |
        | 422          | Unprocessable Entity           | The request was well-formed but was unable to be followed due to semantic errors.|
        | 500          | Internal Server Error          | An error occurred on the server.                                            |
        | 502          | Bad Gateway                    | The server was acting as a gateway or proxy and received an invalid response from the upstream server.|
        | 503          | Service Unavailable            | The server is currently unavailable (because it is overloaded or down for maintenance).|
        | 504          | Gateway Timeout                | The server was acting as a gateway or proxy and did not receive a timely response from the upstream server.|
        '''
        
    ANSWER_SYSTEM_SUCCESS = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a helpful AI assistant, knowledgeable about the Github API. You will receive data about a commit. You will receive a table with information about the commits, copy it to your answer. Do not mention the fact you have copied the table. You do not need to explain the information in your answer, assume the user has a grasp on basic concepts about commits. <|eot_id|>
    '''

    ANSWER_SYSTEM_ERROR = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a helpful AI assistant, knowledgeable about the Github API. You will receive an error related to a query made to the Github API. Explain the error to the user, offer a suggestion to fix it. <|eot_id|>
    '''
    
    api_key = os.getenv("TOGETHERAI_MIXTRAL_API_KEY")
    if not api_key:
        raise ValueError("API key is missing or empty")
    together.api_key = api_key.strip()
    
    def commits_to_markdown(self, commits):
        markdown = "| Commit SHA | Author | Message |\n"
        markdown += "|------------|--------|---------|\n"
        
        for commit in commits:
            sha = commit['sha']
            author = commit['commit']['author']['name']
            message = commit['commit']['message'].replace('\n', ' ').replace('|', '\|')
            
            markdown += f"| {sha} | {author} | {message} |\n"
        
        return markdown

    async def _aprep_clash(self,
        chat_history: list, 
        #agent_type: str, 
        #user_p: str, 
        #q_len: int
    ):
        
        self.hist = chat_history
    
 
        instruction = '''
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a helpful AI assitant, knowledgeable about the Github API. You will be asked to read text and extract parameter values used for queries to the Github API. Specifically, the queries will be related to Commits. The following contains all of the parameters you will need to look for inside the text:\n{parameters}
            Omit every parameter that is not mentioned. If no parameters are mentioned, return empty braces '{{}}'. Your answer must only include the relevant field name and its value in a json format. <|eot_id|><|start_header_id|>user<|end_header_id|>
            List the last 5 commits made to the main branch<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            {{"per_page": 5, "sha": "main"}}<|eot_id|><|start_header_id|>user<|end_header_id|>
            Show me all the commits made to the branch test since january 2023<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            {{"sha": "test", "since": "2023-01-01T00:00:00Z"}}<|eot_id|><|start_header_id|>user<|end_header_id|>
            Display the commits made since march 2 2020 by mike-p containig the path /root/api <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            {{"until":"2020-02-03T00:00:00Z", "author": "mike-p", "path"="/root/api"}}<|eot_id|><|start_header_id|>user<|end_header_id|>
            {user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            
            '''
        
        #example = "- Search for open pull requests with base branch tester\n- Search for all pull requests with head octocat:new-branch, sort them in ascending order."
        #user_p ="List page 2 of the pull request with 'tester' as base branch"
        #user_p = "List the fourth page of the file changes made to 675"
        #field = ['state="', 'head="', 'base="', 'sort="', 'direction="', 'page="', 'per_page="']
        output =    together.Complete.create(
                    #model = "mistralai/Mixtral-8x7B-Instruct-v0.1", 
                    #model="cognitivecomputations/dolphin-2.5-mixtral-8x7b",
                    #model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",   
                    #model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT",   
                    #model="Qwen/Qwen1.5-14B-Chat",
                    prompt=instruction.format(parameters= self.parameters, user_prompt = self.user_q),
                    #model="Qwen/Qwen2-72B-Instruct",
                    model="meta-llama/Llama-3-70b-chat-hf",
                    #model="meta-llama/Llama-3-8b-chat-hf",
                    #model="deepseek-ai/deepseek-coder-33b-instruct",
                    #prompt=prompt_3,
                    #prompt=PROMPT_SATORI_FINAL_L.format(query=query, hist=t2),
                    temperature=0.1,
                    top_p=0.85,
                    top_k=40,
                    max_tokens=128,  
                    repetition_penalty=1.1,
                    #stop = ["</s>", "[INST]", "<|im_end|>"]
                    stop = ["</s>", "[INST]", "User:", "assistant"]#, "`", "` "] #, '"', "\"\n\n", "</json>",'"\n']#, "01", "_v01"]
            )
        #print("\n")
        print(f"Prompt prep clash:{instruction}")
        out_long = output['output']['choices'][0]['text']
        print(out_long)
        print(output["output"]["usage"])
        print("*"*25)
        return out_long
    
    async def _arun_clash(
        self,
        parameters: dict,
    ):
        # Execute tool
        """Use the tool."""
        try:

            # Date and query params 
            #params = calculate_datetime_ranges(self.timezone)
            #params["query"] = parameters
            params={}
            params["hist"] = self.hist
            '''
            adv_query = await async_chain_together_m(
                prompt=PROMPT_ADVANCED_QUERY_PULL_GITHUB,
                params=params,
                max_tokens=768,
                temperature=0.1,
                top_p=1,
                top_k=50,
                repetition_penalty=1.1,
                model="meta-llama/Llama-3-70b-chat-hf",
                stop = ["</s>", "[INST]", "<|EOT|>", '"'],
                tool_name="ClickUp Advanced Query Search",
                callback_manager=self.async_callback_manager
                )
            '''
            # Log step clickup improver
            #if self.verbose:
            #    step = {"type": "clickup_query", "properties": {"query": adv_query}}
            #    await self.async_callback_manager.queue.put(f"#STEP{json.dumps(step)}STEP#")
            # Run the task extractor
            #pulls = await self._arun(query=adv_query)
            # Log step clickup improver
            #if self.verbose:
            #    step = {"type": f"{self.name}", "results": tasks}
            #    await self.async_callback_manager.queue.put(f"#STEP_OUT{json_step.dumps(step, ignore_nan=True)}STEP_OUT#")
            
            parameters["repo_owner"] = self.repo_owner
            parameters["repo_name"] = self.repo_name
            parameters["token"] = self.token
            '''
            token = os.getenv("GITHUB_TOKEN")
            if not token:
                raise ValueError("Token is missing or empty")
            parameters["token"] = token.strip()
            '''
            commits = self.commit.list_all(**parameters)
            #print(f"Commit result:{commits}")
            parameters.pop("token")
            if isinstance(commits, list):
                #print(f"Found {len(pr_list)} pulls")
                extra_export = ", finally, you must notify the user the tasks were exported into a csv file inside the session folder, this can be located on the right menu" if self.export else ""
                answer_inst = f'''{self.ANSWER_SYSTEM_SUCCESS}<|start_header_id|>user<|end_header_id|>
                A total of {len(commits)} files were changed.\n```{self.commits_to_markdown(commits)}```\n.<|eot_id|><|start_header_id|>assistant<|end_header_id|>'''
                    
            elif isinstance(commits, dict):
                #print(pr_list)
                #print(f"Could not perform read pull operation:\nMessage: {pr_list['message']}\nError code: {pr_list['status']}")
                answer_inst = f'''{self.ANSWER_SYSTEM_ERROR}<|start_header_id|>user<|end_header_id|>
                An error ocurred when attempting to retrieve the diff files. The most common errors are:\n```{self.error_code}```\n.The message error is {commits['message']} and the error code is {commits['status']}.\n Explain what the possible cause of the error is and suggest a solution.<|eot_id|><|start_header_id|>assistant<|end_header_id|>'''              
            
            params = {
                "repo_owner": self.repo_owner,                
                "repo_name": self.repo_name,
                "hist": self.hist,
                "lang": self.lang,
                "task_answer_instruction": answer_inst,
                #"query": self.user_msg,
                #"date": datetime.now(pytz.timezone(self.timezone)).strftime("%A, %d of %B with time %H:%M of %Y"),
                "init_lang": "Por supuesto!\n" if self.lang == "Spanish" else "Certainly!\n"
            }
            
            # Stream
            '''
            result = await achain_nirvana(
            prompt=general_prompt,
            params=params,
            max_tokens=2048,
            temperature=0.1,
            top_p=0.85,
            top_k=40,
            repetition_penalty=1.1,
            model = "together|meta-llama/Llama-3-70b-chat-hf",
            stop = ["</s>", "[INST]", "<|EOT|>", "<|eot_id|>"],
            tool_name="SQL Generation",
            stream=False,
            )
            
            '''
            
            output =  together.Complete.create(
                    #model = "mistralai/Mixtral-8x7B-Instruct-v0.1", 
                    #model="cognitivecomputations/dolphin-2.5-mixtral-8x7b",
                    #model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",   
                    #model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT",   
                    #model="Qwen/Qwen1.5-14B-Chat",
                    prompt=answer_inst,
                    #model="Qwen/Qwen2-72B-Instruct",
                    model="meta-llama/Llama-3-70b-chat-hf",
                    #model="meta-llama/Llama-3-8b-chat-hf",
                    #model="deepseek-ai/deepseek-coder-33b-instruct",
                    #prompt=prompt_3,
                    #prompt=PROMPT_SATORI_FINAL_L.format(query=query, hist=t2),
                    temperature=0.1,
                    top_p=0.85,
                    top_k=40,
                    max_tokens=128,  
                    repetition_penalty=1.1,
                    #stop = ["</s>", "[INST]", "<|im_end|>"]
                    stop = ["</s>", "[INST]", "User:", "assistant"]#, "`", "` "] #, '"', "\"\n\n", "</json>",'"\n']#, "01", "_v01"]
            )
            #print("\n")
            print(f"Prompt run clash (answer inst):{answer_inst}")
            out_long = output['output']['choices'][0]['text']
            #print(f"Answer prompt output:\n{out_long}")
            print(output["output"]["usage"])
            print("*"*25)
            # Export
            if self.export:
                await self.async_callback_manager.queue.put(f"\nExport: [{self.user_msg}]({tasks['export_file']})")
            return {"finish": True, "response": output['output']['choices'][0]['text']}
        except Exception as e_tasks_clickup:
            ic(e_tasks_clickup)
            spaces = '- ' + '\n- '.join([f'{x}: {y}' for x, y in self.clickup_dicts['spaces'].items()])
            trace = traceback.format_exc()
            return {"error": "No tasks could be extracted.", "raise_msg": f"Is it likely the query used and the search parameters are not correct. Explain this query to the user in natural language and ask for feedback. These are the spaces available:\n{spaces}\nQuestion: {query}\nSearch query: {adv_query}", "trace": trace}

class GithubBranch():
    
    def __init__(self, token, repo_owner, repo_name, user_q):
        self.token = token
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.commit = Commits()
        self.lang: Optional[str] = "English"
        self.export: Optional[str] = False
        self.branches = Branches()
        self.user_q = user_q
        #self.pull_id = pull_id     
        self.parameters = '''
        | Parameter   | Type    | Description                                                                                 | Required/Optional |
        |-------------|---------|---------------------------------------------------------------------------------------------|-------------------|
        | protected   | boolean | Only show branches that are protected (true) or not protected (false).                      | Optional          |
        | per_page    | integer | Number of results per page (maximum 100).                                                   | Optional          |
        | page        | integer | Page number of the results to fetch.                                                        | Optional          |
        '''
        
        self.error_code = '''
        | Status Code | Error Message                  | Description                                                                 |
        |--------------|--------------------------------|-----------------------------------------------------------------------------|
        | 200          | OK                             | The request was successful.                                                 |
        | 301          | Moved Permanently              | The resource has been moved to a new URL.                                   |
        | 304          | Not Modified                   | There was no new data to return.                                            |
        | 400          | Bad Request                    | The request was invalid or cannot be served.                                |
        | 401          | Unauthorized                   | Authentication is required and has failed or has not yet been provided.     |
        | 403          | Forbidden                      | The request is understood, but it has been refused or access is not allowed.|
        | 404          | Not Found                      | The requested resource could not be found.                                  |
        | 422          | Unprocessable Entity           | The request was well-formed but was unable to be followed due to semantic errors.|
        | 500          | Internal Server Error          | An error occurred on the server.                                            |
        | 502          | Bad Gateway                    | The server was acting as a gateway or proxy and received an invalid response from the upstream server.|
        | 503          | Service Unavailable            | The server is currently unavailable (because it is overloaded or down for maintenance).|
        | 504          | Gateway Timeout                | The server was acting as a gateway or proxy and did not receive a timely response from the upstream server.|
        '''
        
    ANSWER_SYSTEM_SUCCESS = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a helpful AI assistant, knowledgeable about the Github API. You will receive data about branches from a repository. You will receive a table with information about the branches, copy it to your answer. Do not mention the fact you have copied the table. You do not need to explain the information in your answer, assume the user has a grasp on basic concepts about github. <|eot_id|>
    '''

    ANSWER_SYSTEM_ERROR = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a helpful AI assistant, knowledgeable about the Github API. You will receive an error related to a query made to the Github API. Explain the error to the user, offer a suggestion to fix it. <|eot_id|>
    '''
    
    api_key = os.getenv("TOGETHERAI_MIXTRAL_API_KEY")
    if not api_key:
        raise ValueError("API key is missing or empty")
    together.api_key = api_key.strip()   
    
    def branches_to_markdown(self, branches):
        markdown = "| Branch Name | Protected | Commit SHA |\n"
        markdown += "|-------------|-----------|------------|\n"
        
        for branch in branches:
            name = branch['name']
            protected = branch['protected']
            sha = branch['commit']['sha']
            
            markdown += f"| {name} | {protected} | {sha} |\n"
        
        return markdown
    
    async def _aprep_clash(self,
        chat_history: list, 
        #agent_type: str, 
        #user_p: str, 
        #q_len: int
    ):
        
        self.hist = chat_history
    
 
        instruction = '''
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a helpful AI assitant, knowledgeable about the Github API. You will be asked to read text and extract parameter values used for queries to the Github API. Specifically, the queries will be related to Branches. The following contains all of the parameters you will need to look for inside the text:\n{parameters}
            Omit every parameter that is not mentioned. If no parameters are mentioned, return empty braces '{{}}'. Your answer must only include the relevant field name and its value in a json format. <|eot_id|><|start_header_id|>user<|end_header_id|>
            List the protected branches<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            {{"protected": True}}<|eot_id|><|start_header_id|>user<|end_header_id|>
            Show page 2 of the protected branches<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            {{"protected": True, "page": 2}}<|eot_id|><|start_header_id|>user<|end_header_id|>
            {user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            
            '''
        
        #example = "- Search for open pull requests with base branch tester\n- Search for all pull requests with head octocat:new-branch, sort them in ascending order."
        #user_p ="List page 2 of the pull request with 'tester' as base branch"
        #user_p = "List the fourth page of the file changes made to 675"
        #field = ['state="', 'head="', 'base="', 'sort="', 'direction="', 'page="', 'per_page="']
        output =    together.Complete.create(
                    #model = "mistralai/Mixtral-8x7B-Instruct-v0.1", 
                    #model="cognitivecomputations/dolphin-2.5-mixtral-8x7b",
                    #model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",   
                    #model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT",   
                    #model="Qwen/Qwen1.5-14B-Chat",
                    prompt=instruction.format(parameters= self.parameters, user_prompt = self.user_q),
                    #model="Qwen/Qwen2-72B-Instruct",
                    model="meta-llama/Llama-3-70b-chat-hf",
                    #model="meta-llama/Llama-3-8b-chat-hf",
                    #model="deepseek-ai/deepseek-coder-33b-instruct",
                    #prompt=prompt_3,
                    #prompt=PROMPT_SATORI_FINAL_L.format(query=query, hist=t2),
                    temperature=0.1,
                    top_p=0.85,
                    top_k=40,
                    max_tokens=128,  
                    repetition_penalty=1.1,
                    #stop = ["</s>", "[INST]", "<|im_end|>"]
                    stop = ["</s>", "[INST]", "User:", "assistant"]#, "`", "` "] #, '"', "\"\n\n", "</json>",'"\n']#, "01", "_v01"]
            )
        #print("\n")
        print(f"Prompt prep clash:{instruction}")
        out_long = output['output']['choices'][0]['text']
        print(out_long)
        print(output["output"]["usage"])
        print("*"*25)
        return out_long
    
    async def _arun_clash(
        self,
        parameters: dict,
    ):
        # Execute tool
        """Use the tool."""
        try:

            # Date and query params 
            #params = calculate_datetime_ranges(self.timezone)
            #params["query"] = parameters
            params={}
            params["hist"] = self.hist
            '''
            adv_query = await async_chain_together_m(
                prompt=PROMPT_ADVANCED_QUERY_PULL_GITHUB,
                params=params,
                max_tokens=768,
                temperature=0.1,
                top_p=1,
                top_k=50,
                repetition_penalty=1.1,
                model="meta-llama/Llama-3-70b-chat-hf",
                stop = ["</s>", "[INST]", "<|EOT|>", '"'],
                tool_name="ClickUp Advanced Query Search",
                callback_manager=self.async_callback_manager
                )
            '''
            # Log step clickup improver
            #if self.verbose:
            #    step = {"type": "clickup_query", "properties": {"query": adv_query}}
            #    await self.async_callback_manager.queue.put(f"#STEP{json.dumps(step)}STEP#")
            # Run the task extractor
            #pulls = await self._arun(query=adv_query)
            # Log step clickup improver
            #if self.verbose:
            #    step = {"type": f"{self.name}", "results": tasks}
            #    await self.async_callback_manager.queue.put(f"#STEP_OUT{json_step.dumps(step, ignore_nan=True)}STEP_OUT#")
            
            parameters["repo_owner"] = self.repo_owner
            parameters["repo_name"] = self.repo_name
            parameters["token"] = self.token
            '''
            token = os.getenv("GITHUB_TOKEN")
            if not token:
                raise ValueError("Token is missing or empty")
            parameters["token"] = token.strip()
            '''
            branches = self.branches.list_all(**parameters)
            #print(f"Commit result:{commits}")
            parameters.pop("token")
            if isinstance(branches, list):
                #print(f"Found {len(pr_list)} pulls")
                extra_export = ", finally, you must notify the user the tasks were exported into a csv file inside the session folder, this can be located on the right menu" if self.export else ""
                answer_inst = f'''{self.ANSWER_SYSTEM_SUCCESS}<|start_header_id|>user<|end_header_id|>
                A total of {len(branches)} files were changed.\n```{self.branches_to_markdown(branches)}```\n.<|eot_id|><|start_header_id|>assistant<|end_header_id|>'''
                    
            elif isinstance(branches, dict):
                #print(pr_list)
                #print(f"Could not perform read pull operation:\nMessage: {pr_list['message']}\nError code: {pr_list['status']}")
                answer_inst = f'''{self.ANSWER_SYSTEM_ERROR}<|start_header_id|>user<|end_header_id|>
                An error ocurred when attempting to retrieve the diff files. The most common errors are:\n```{self.error_code}```\n.The message error is {branches['message']} and the error code is {branches['status']}.\n Explain what the possible cause of the error is and suggest a solution.<|eot_id|><|start_header_id|>assistant<|end_header_id|>'''              
            
            params = {
                "repo_owner": self.repo_owner,                
                "repo_name": self.repo_name,
                "hist": self.hist,
                "lang": self.lang,
                "task_answer_instruction": answer_inst,
                #"query": self.user_msg,
                #"date": datetime.now(pytz.timezone(self.timezone)).strftime("%A, %d of %B with time %H:%M of %Y"),
                "init_lang": "Por supuesto!\n" if self.lang == "Spanish" else "Certainly!\n"
            }
            
            # Stream
            '''
            result = await achain_nirvana(
            prompt=general_prompt,
            params=params,
            max_tokens=2048,
            temperature=0.1,
            top_p=0.85,
            top_k=40,
            repetition_penalty=1.1,
            model = "together|meta-llama/Llama-3-70b-chat-hf",
            stop = ["</s>", "[INST]", "<|EOT|>", "<|eot_id|>"],
            tool_name="SQL Generation",
            stream=False,
            )
            
            '''
            
            output =  together.Complete.create(
                    #model = "mistralai/Mixtral-8x7B-Instruct-v0.1", 
                    #model="cognitivecomputations/dolphin-2.5-mixtral-8x7b",
                    #model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",   
                    #model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT",   
                    #model="Qwen/Qwen1.5-14B-Chat",
                    prompt=answer_inst,
                    #model="Qwen/Qwen2-72B-Instruct",
                    model="meta-llama/Llama-3-70b-chat-hf",
                    #model="meta-llama/Llama-3-8b-chat-hf",
                    #model="deepseek-ai/deepseek-coder-33b-instruct",
                    #prompt=prompt_3,
                    #prompt=PROMPT_SATORI_FINAL_L.format(query=query, hist=t2),
                    temperature=0.1,
                    top_p=0.85,
                    top_k=40,
                    max_tokens=128,  
                    repetition_penalty=1.1,
                    #stop = ["</s>", "[INST]", "<|im_end|>"]
                    stop = ["</s>", "[INST]", "User:", "assistant"]#, "`", "` "] #, '"', "\"\n\n", "</json>",'"\n']#, "01", "_v01"]
            )
            #print("\n")
            print(f"Prompt run clash (answer inst):{answer_inst}")
            out_long = output['output']['choices'][0]['text']
            #print(f"Answer prompt output:\n{out_long}")
            print(output["output"]["usage"])
            print("*"*25)
            # Export
            if self.export:
                await self.async_callback_manager.queue.put(f"\nExport: [{self.user_msg}]({tasks['export_file']})")
            return {"finish": True, "response": output['output']['choices'][0]['text']}
        except Exception as e_tasks_clickup:
            ic(e_tasks_clickup)
            spaces = '- ' + '\n- '.join([f'{x}: {y}' for x, y in self.clickup_dicts['spaces'].items()])
            trace = traceback.format_exc()
            return {"error": "No tasks could be extracted.", "raise_msg": f"Is it likely the query used and the search parameters are not correct. Explain this query to the user in natural language and ask for feedback. These are the spaces available:\n{spaces}\nQuestion: {query}\nSearch query: {adv_query}", "trace": trace}

class GithubSearchFile():
    
    def __init__(self, token, repo_owner, repo_name, user_q):
        self.parameters = '''
        | Parameter   | Type    | Description                                                                                           | Required/Optional |
        |-------------|---------|-------------------------------------------------------------------------------------------------------|-------------------|
        | q           | string  | The search query. Can include qualifiers such as `in:file`, `language:python`, `repo:owner/repo`, etc. | Required          |
        | sort        | string  | The sort field. Can be `indexed` (the default, sorts by the last indexed time).                        | Optional          |
        | order       | string  | The sort order if `sort` parameter is provided. Can be either `asc` or `desc`.                        | Optional          |
        | per_page    | integer | The number of results per page (max 100).                                                              | Optional          |
        | page        | integer | The page number of the results to fetch.                                                               | Optional          |
        '''
        self.syntax = '''
        | Qualifier          | Description                                                                                           | Example Usage                                          |
        |--------------------|-------------------------------------------------------------------------------------------------------|--------------------------------------------------------|
        | `q`                | The search query. Can include text and qualifiers.                                                    | `q=addClass`                                           |
        | `in`               | Qualifies which fields are searched. Can be `file`, `path`, or `readme`.                              | `q=addClass in:file`                                   |
        | `language`         | Searches for code of a specific language.                                                             | `q=addClass language:javascript`                       |
        | `repo`             | Limits searches to a specific repository.                                                            | `q=addClass repo:jquery/jquery`                        |
        | `org`              | Limits searches to a specific organization.                                                          | `q=addClass org:github`                                |
        | `user`             | Limits searches to a specific user.                                                                  | `q=addClass user:octocat`                              |
        | `path`             | Limits searches to a specific path.                                                                  | `q=addClass path:src`                                  |
        | `filename`         | Searches for a specific file by name.                                                                | `q=addClass filename:index.js`                         |
        | `extension`        | Searches for files with a specific extension.                                                        | `q=addClass extension:js`                              |
        | `size`             | Searches for files based on size (bytes).                                                            | `q=addClass size:1000..2000`                           |
        | `fork`             | Includes forked repositories. Use `true` or `only` to include/exclude.                               | `q=addClass fork:true`                                 |
        | `created`          | Searches based on file creation date (ISO 8601 format or relative date).                             | `q=addClass created:>=2020-01-01`                      |
        | `pushed`           | Searches based on when code was last pushed (ISO 8601 format or relative date).                      | `q=addClass pushed:>=2020-01-01`                       |
        | `stars`            | Searches based on the number of stars a repository has.                                              | `q=addClass stars:>100`                                |
        | `sort`             | Sorts the results. Can be `indexed`.                                                                 | `sort=indexed`                                         |
        | `order`            | The order of the sorted results. Can be `asc` or `desc`.                                             | `order=desc`                                           |
        | `per_page`         | Number of results per page (max 100).                                                                | `per_page=10`                                          |
        | `page`             | Page number of the results to fetch.                                                                 | `page=1`                                               |   
        '''
        self.token = token
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.user_q = user_q
        self.lang: Optional[str] = "English"
        self.export: Optional[str] = False
        self.search = Files()
        self.error_code = '''
        | Status Code | Error Message                  | Description                                                                 |
        |--------------|--------------------------------|-----------------------------------------------------------------------------|
        | 200          | OK                             | The request was successful.                                                 |
        | 400          | Bad Request                    | The request was invalid or cannot be served.                                |
        | 401          | Unauthorized                   | Authentication is required and has failed or has not yet been provided.     |
        | 403          | Forbidden                      | The request is understood, but it has been refused or access is not allowed.|
        | 404          | Not Found                      | The requested resource could not be found.                                  |
        | 422          | Unprocessable Entity           | The request was well-formed but was unable to be followed due to semantic errors.|
        | 500          | Internal Server Error          | An error occurred on the server.                                            |
        | 502          | Bad Gateway                    | The server was acting as a gateway or proxy and received an invalid response from the upstream server.|
        | 503          | Service Unavailable            | The server is currently unavailable (because it is overloaded or down for maintenance).|
        | 504          | Gateway Timeout                | The server was acting as a gateway or proxy and did not receive a timely response from the upstream server.|
        '''       
        
    ANSWER_SYSTEM_SUCCESS = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a helpful AI assistant, knowledgeable about the Github API. You will receive data about a search query inside a repository. You will receive a table with information about the results, copy it to your answer. Do not mention the fact that you have copied the table. You do not need to explain the information in your answer, assume the user has a grasp on basic concepts about github. <|eot_id|>
    '''

    ANSWER_SYSTEM_ERROR = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a helpful AI assistant, knowledgeable about the Github API. You will receive an error related to a query made to the Github API. Explain the error to the user, offer a suggestion to fix it. <|eot_id|>
    '''
    
    api_key = os.getenv("TOGETHERAI_MIXTRAL_API_KEY")
    if not api_key:
        raise ValueError("API key is missing or empty")
    together.api_key = api_key.strip()   
    
    
    def results_to_markdown(self, results):
        items = results['items'] if 'items' in results else []
        

        # Markdown table header
        markdown_table = "| Filename | Size (bytes) | Path |\n|----------|--------------|------|\n"

        # Append each file's information to the markdown table
        for item in items:
            #print(item.keys())
            filename = item['name']  # accessing file name
            size = item.get('size', 'N/A') # accessing file size
            path = item.get('path', 'N/A')
            markdown_table += f"| {filename} | {size} | {path} |\n"

        #print(markdown_table)
        return markdown_table

    
    async def _aprep_clash(self,
        chat_history: list, 
        #agent_type: str, 
        #user_p: str, 
        #q_len: int
    ):
        
        self.hist = chat_history
    
 
        instruction = '''
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a helpful AI assitant, knowledgeable about the Github API. You will be asked to read text and form a query for the Github API. Specifically, the queries will be related to search content inside a repository. The repository owner is {owner}, the repository name is {repo}. The github API endpoint is https://api.github.com/search. Your answer must always contain the endpoint URL, repository owner and repository name. The following contains the parameters you will need to look for inside the text:\n{parameters}\n This next table contains the syntax used for the query corresponding to the parameter 'q' from the previous table:\n{syntax}\n
            Omit every parameter that is not mentioned. If no parameters are mentioned, return an empty string ''. Your answer must include the string query corresponding to the text. <|eot_id|><|start_header_id|>user<|end_header_id|>
            Search for the term 'handler' in python files<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            "https://api.github.com/search/code?q=handler+in:file+language:python+repo:{owner}/{repo}"<|eot_id|><|start_header_id|>user<|end_header_id|>
            Search for files containing 'handler' in their name or content, pushed before march 3rd 2021<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            "https://api.github.com/search/code?q=handler+in:path,file+pushed:<2021-03-03+repo:{owner}/{repo}"<|eot_id|><|start_header_id|>user<|end_header_id|>
            Search for files lighter than 100KB, including forked repositories, with .exe extension and sort them by file creation date in descending order<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            "https://api.github.com/search/code?q=size:<100000+fork:true+extension:exe+repo:{owner}/{repo}&sort=created&order=desc"<|eot_id|><|start_header_id|>user<|end_header_id|>
            {user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            '''
        
        #example = "- Search for open pull requests with base branch tester\n- Search for all pull requests with head octocat:new-branch, sort them in ascending order."
        #user_p ="List page 2 of the pull request with 'tester' as base branch"
        #user_p = "List the fourth page of the file changes made to 675"
        #field = ['state="', 'head="', 'base="', 'sort="', 'direction="', 'page="', 'per_page="']
        output =    together.Complete.create(
                    #model = "mistralai/Mixtral-8x7B-Instruct-v0.1", 
                    #model="cognitivecomputations/dolphin-2.5-mixtral-8x7b",
                    #model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",   
                    #model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT",   
                    #model="Qwen/Qwen1.5-14B-Chat",
                    prompt=instruction.format(owner = self.repo_owner, repo=self.repo_name, parameters= self.parameters, syntax = self.syntax, user_prompt = self.user_q),
                    #model="Qwen/Qwen2-72B-Instruct",
                    model="meta-llama/Llama-3-70b-chat-hf",
                    #model="meta-llama/Llama-3-8b-chat-hf",
                    #model="deepseek-ai/deepseek-coder-33b-instruct",
                    #prompt=prompt_3,
                    #prompt=PROMPT_SATORI_FINAL_L.format(query=query, hist=t2),
                    temperature=0.1,
                    top_p=0.85,
                    top_k=40,
                    max_tokens=128,  
                    repetition_penalty=1.1,
                    #stop = ["</s>", "[INST]", "<|im_end|>"]
                    stop = ["</s>", "[INST]", "User:", "assistant"]#, "`", "` "] #, '"', "\"\n\n", "</json>",'"\n']#, "01", "_v01"]
            )
        print(f"Prompt prep clash:{instruction}")
        #print("\n")
        out_long = output['output']['choices'][0]['text']
        print(out_long)
        print(output["output"]["usage"])
        print("*"*25)
        return out_long
    
    async def _arun_clash(
    self,
    url: str,
    ):
        # Execute tool
        """Use the tool."""
        try:

            # Date and query params 
            #params = calculate_datetime_ranges(self.timezone)
            #params["query"] = parameters
            params={}
            params["hist"] = self.hist
            '''
            adv_query = await async_chain_together_m(
                prompt=PROMPT_ADVANCED_QUERY_PULL_GITHUB,
                params=params,
                max_tokens=768,
                temperature=0.1,
                top_p=1,
                top_k=50,
                repetition_penalty=1.1,
                model="meta-llama/Llama-3-70b-chat-hf",
                stop = ["</s>", "[INST]", "<|EOT|>", '"'],
                tool_name="ClickUp Advanced Query Search",
                callback_manager=self.async_callback_manager
                )
            '''
            # Log step clickup improver
            #if self.verbose:
            #    step = {"type": "clickup_query", "properties": {"query": adv_query}}
            #    await self.async_callback_manager.queue.put(f"#STEP{json.dumps(step)}STEP#")
            # Run the task extractor
            #pulls = await self._arun(query=adv_query)
            # Log step clickup improver
            #if self.verbose:
            #    step = {"type": f"{self.name}", "results": tasks}
            #    await self.async_callback_manager.queue.put(f"#STEP_OUT{json_step.dumps(step, ignore_nan=True)}STEP_OUT#")
            
            '''
            token = os.getenv("GITHUB_TOKEN")
            if not token:
                raise ValueError("Token is missing or empty")
            parameters["token"] = token.strip()
            '''
            search_results = self.search.search_file(self.token, url)
            #print(search_results)
            #print(f"Commit result:{commits}")
            if "status" not in search_results.keys(): #query executed successfully
                items = search_results['items'] if 'items' in search_results else []
                #print(f"Found {len(pr_list)} pulls")
                extra_export = ", finally, you must notify the user the tasks were exported into a csv file inside the session folder, this can be located on the right menu" if self.export else ""
                answer_inst = f'''{self.ANSWER_SYSTEM_SUCCESS}<|start_header_id|>user<|end_header_id|>
                A total of {len(items)} files were found in the search, mention that exact number in your answer.\n```{self.results_to_markdown(search_results)}```\n.<|eot_id|><|start_header_id|>assistant<|end_header_id|>'''
                    
            else: # an error ocurred when calling the api endpoint
                #print(pr_list)
                #print(f"Could not perform read pull operation:\nMessage: {pr_list['message']}\nError code: {pr_list['status']}")
                answer_inst = f'''{self.ANSWER_SYSTEM_ERROR}<|start_header_id|>user<|end_header_id|>
                An error ocurred when attempting to execute the search query. The most common errors are:\n```{self.error_code}```\n.The message error is {search_results['message']} and the error code is {search_results['status']}.\n Explain what the possible cause of the error is and suggest a solution.<|eot_id|><|start_header_id|>assistant<|end_header_id|>'''              
            
            params = {
                "repo_owner": self.repo_owner,                
                "repo_name": self.repo_name,
                "hist": self.hist,
                "lang": self.lang,
                "task_answer_instruction": answer_inst,
                #"query": self.user_msg,
                #"date": datetime.now(pytz.timezone(self.timezone)).strftime("%A, %d of %B with time %H:%M of %Y"),
                "init_lang": "Por supuesto!\n" if self.lang == "Spanish" else "Certainly!\n"
            }
            
            # Stream
            '''
            result = await achain_nirvana(
            prompt=general_prompt,
            params=params,
            max_tokens=2048,
            temperature=0.1,
            top_p=0.85,
            top_k=40,
            repetition_penalty=1.1,
            model = "together|meta-llama/Llama-3-70b-chat-hf",
            stop = ["</s>", "[INST]", "<|EOT|>", "<|eot_id|>"],
            tool_name="SQL Generation",
            stream=False,
            )
            
            '''
            
            output =  together.Complete.create(
                    #model = "mistralai/Mixtral-8x7B-Instruct-v0.1", 
                    #model="cognitivecomputations/dolphin-2.5-mixtral-8x7b",
                    #model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",   
                    #model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT",   
                    #model="Qwen/Qwen1.5-14B-Chat",
                    prompt=answer_inst,
                    #model="Qwen/Qwen2-72B-Instruct",
                    model="meta-llama/Llama-3-70b-chat-hf",
                    #model="meta-llama/Llama-3-8b-chat-hf",
                    #model="deepseek-ai/deepseek-coder-33b-instruct",
                    #prompt=prompt_3,
                    #prompt=PROMPT_SATORI_FINAL_L.format(query=query, hist=t2),
                    temperature=0.1,
                    top_p=0.85,
                    top_k=40,
                    max_tokens=128,  
                    repetition_penalty=1.1,
                    #stop = ["</s>", "[INST]", "<|im_end|>"]
                    stop = ["</s>", "[INST]", "User:", "assistant"]#, "`", "` "] #, '"', "\"\n\n", "</json>",'"\n']#, "01", "_v01"]
            )
            #print("\n")
            print(f"Prompt run clash (answer inst):{answer_inst}")
            out_long = output['output']['choices'][0]['text']
            #print(f"Answer prompt output:\n{out_long}")
            print(output["output"]["usage"])
            print("*"*25)
            # Export
            if self.export:
                await self.async_callback_manager.queue.put(f"\nExport: [{self.user_msg}]({tasks['export_file']})")
            return {"finish": True, "response": output['output']['choices'][0]['text']}
        except Exception as e_tasks_clickup:
            ic(e_tasks_clickup)
            spaces = '- ' + '\n- '.join([f'{x}: {y}' for x, y in self.clickup_dicts['spaces'].items()])
            trace = traceback.format_exc()
            return {"error": "No tasks could be extracted.", "raise_msg": f"Is it likely the query used and the search parameters are not correct. Explain this query to the user in natural language and ask for feedback. These are the spaces available:\n{spaces}\nQuestion: {query}\nSearch query: {adv_query}", "trace": trace}

class GithubIssue():
    
    def __init__(self, token, repo_owner, repo_name, user_q):

        self.parameters = '''
        | Parameter     | Description                                                                                   | Example                                    |
        |---------------|-----------------------------------------------------------------------------------------------|--------------------------------------------|
        | `q`           | The search query.                                                                              | `q=repo:owner/repo is:open bug`           |
        | `sort`        | The sort field. Can be `comments`, `created`, `updated`, or `reactions`.                       | `sort=created`                             |
        | `order`       | The sort order. Can be `asc` (ascending) or `desc` (descending).                               | `order=desc`                               |
        | `per_page`    | Number of items per page. Default is 30.                                                       | `per_page=50`                              |
        | `page`        | The page number to retrieve. Default is 1.                                                     | `page=2`                                   |
        | `since`       | Only issues updated at or after this time are returned.                                        | `since=2023-01-01T00:00:00Z`               |
        | `created`     | Only issues created at or after this time are returned.                                         | `created:>=2023-01-01`                     |
        | `updated`     | Only issues updated at or after this time are returned.                                         | `updated:>=2023-01-01`                     |
        | `comments`    | Filters issues based on the number of comments.                                                 | `comments:>=10`                            |
        | `reactions`   | Filters issues based on the number of reactions.                                                | `reactions:>=5`                            |
        | `user`        | Finds issues created by a certain user.                                                         | `user:octocat`                             |
        | `repo`        | Limits searches to a specific repository.                                                       | `repo:owner/repo`                          |
        | `org`         | Limits searches to a specific organization.                                                     | `org:github`                               |
        | `assignee`    | Finds issues assigned to a certain user.                                                        | `assignee:octocat`                         |
        | `mentions`    | Finds issues mentioning a certain user.                                                         | `mentions:octocat`                         |
        | `team`        | Finds issues that are assigned to teams within an organization.                                 | `team:github/org-team`                     |
        | `is`          | Filters issues based on certain states. Possible values are `open`, `closed`, or `merged`.      | `is:open`                                  |
        | `label`       | Filters issues based on labels.                                                                 | `label:bug`                                |
        | `milestone`   | Filters issues based on milestones.                                                             | `milestone:1`                              |
        | `project`     | Filters issues based on projects.                                                               | `project:"Project Name"`                   |
        | `type`        | Filters issues based on issue type. Possible values are `issue` or `pr` (pull request).         | `type:pr`                                  |
        | `in`          | Filters issues based on the location of the search term. Possible values are `title`, `body`, or `comments`. | `in:title`                         |
        | `language`    | Filters issues based on the programming language of the repository.                             | `language:python`                          |
        | `locked`      | Filters issues based on whether they are locked. Possible values are `true` or `false`.         | `locked:true`                              |
        | `state`       | Filters issues based on whether they are `open`, `closed`, or `all` (default).                  | `state:closed`                             |  
        '''
        self.token = token
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.user_q = user_q
        self.lang: Optional[str] = "English"
        self.export: Optional[str] = False
        self.search = Issues()
        self.error_code = '''
        | Status Code | Error Message                  | Description                                                                 |
        |--------------|--------------------------------|-----------------------------------------------------------------------------|
        | 200          | OK                             | The request was successful.                                                 |
        | 400          | Bad Request                    | The request was invalid or cannot be served.                                |
        | 401          | Unauthorized                   | Authentication is required and has failed or has not yet been provided.     |
        | 403          | Forbidden                      | The request is understood, but it has been refused or access is not allowed.|
        | 404          | Not Found                      | The requested resource could not be found.                                  |
        | 422          | Unprocessable Entity           | The request was well-formed but was unable to be followed due to semantic errors.|
        | 500          | Internal Server Error          | An error occurred on the server.                                            |
        | 502          | Bad Gateway                    | The server was acting as a gateway or proxy and received an invalid response from the upstream server.|
        | 503          | Service Unavailable            | The server is currently unavailable (because it is overloaded or down for maintenance).|
        | 504          | Gateway Timeout                | The server was acting as a gateway or proxy and did not receive a timely response from the upstream server.|
        '''       
        
    ANSWER_SYSTEM_SUCCESS = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a helpful AI assistant, knowledgeable about the Github API. You will receive data about a search query for issues inside a repository. You will receive a table with information about the results, copy it to your answer. Do not mention the fact that you have copied the table. You do not need to explain the information in your answer, assume the user has a grasp on basic concepts about github. <|eot_id|>
    '''

    ANSWER_SYSTEM_ERROR = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a helpful AI assistant, knowledgeable about the Github API. You will receive an error related to a query made to the Github API. Explain the error to the user, offer a suggestion to fix it. <|eot_id|>
    '''
    
    api_key = os.getenv("TOGETHERAI_MIXTRAL_API_KEY")
    if not api_key:
        raise ValueError("API key is missing or empty")
    together.api_key = api_key.strip()   
    
        
    def issues_to_markdown(self, issues):
        """
        Function to print GitHub issues in a Markdown table format.

        Parameters:
        - issues (list): List of issues returned from the GitHub API.
        """
        # Print table header
        table ="| Issue Number | Title | URL |"
        table+="|--------------|-------|-----|"

        listado = issues['items'] if 'items' in issues else []
        # Print each issue
        for issue in listado:
            #print(f"Keys:{issue.keys()}")
            issue_number = issue['number']
            title = issue['title']
            url = issue['html_url']

            table+=f"| {issue_number} | {title} | [Link]({url}) |"
        return table
    async def _aprep_clash(self,
        chat_history: list, 
        #agent_type: str, 
        #user_p: str, 
        #q_len: int
    ):
        
        self.hist = chat_history
    
 
        instruction = '''
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a helpful AI assitant, knowledgeable about the Github API. You will be asked to read text and form a query for the Github API. Specifically, the queries will be related to search issues inside a repository. The repository owner is {owner}, the repository name is {repo}. The github API endpoint is https://api.github.com/search/issues?q=is:issue. Your answer must always contain the endpoint URL, repository owner and repository name. The following contains the parameters you will need to look for inside the text:\n{parameters}\n Your task is to find the parameters to be used inside the query based on the previous table.
            Omit every parameter that is not mentioned. Your answer must be the string query corresponding to the text. <|eot_id|><|start_header_id|>user<|end_header_id|>
            Search for open issues that contain crash in the title labeled as bugs<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            https://api.github.com/search/issues?q=is:issue+is:open+label:bug+crash+in:title+repo:{owner}/{repo}<|eot_id|><|start_header_id|>user<|end_header_id|>
            Search all locked issues for the beta milestone created by xiphos since march 1 2022<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            https://api.github.com/search/issues?q=is:issue+is:locked+milestone:beta+author:xiphos+created:>=2022-03-01+repo:{owner}/{repo}<|eot_id|><|start_header_id|>user<|end_header_id|>
            Search for all bugs with token count in the title, content or messages<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            https://api.github.com/search/issues?q=is:issue+label:bug+token+count+in:title,body,comments+repo:{owner}/{repo}&sort=created&order=desc<|eot_id|><|start_header_id|>user<|end_header_id|>
            {user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            '''
        
        #example = "- Search for open pull requests with base branch tester\n- Search for all pull requests with head octocat:new-branch, sort them in ascending order."
        #user_p ="List page 2 of the pull request with 'tester' as base branch"
        #user_p = "List the fourth page of the file changes made to 675"
        #field = ['state="', 'head="', 'base="', 'sort="', 'direction="', 'page="', 'per_page="']
        output =    together.Complete.create(
                    #model = "mistralai/Mixtral-8x7B-Instruct-v0.1", 
                    #model="cognitivecomputations/dolphin-2.5-mixtral-8x7b",
                    #model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",   
                    #model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT",   
                    #model="Qwen/Qwen1.5-14B-Chat",
                    prompt=instruction.format(owner = self.repo_owner, repo=self.repo_name, parameters= self.parameters, user_prompt = self.user_q),
                    #model="Qwen/Qwen2-72B-Instruct",
                    model="meta-llama/Llama-3-70b-chat-hf",
                    #model="meta-llama/Llama-3-8b-chat-hf",
                    #model="deepseek-ai/deepseek-coder-33b-instruct",
                    #prompt=prompt_3,
                    #prompt=PROMPT_SATORI_FINAL_L.format(query=query, hist=t2),
                    temperature=0.1,
                    top_p=0.85,
                    top_k=40,
                    max_tokens=128,  
                    repetition_penalty=1.1,
                    #stop = ["</s>", "[INST]", "<|im_end|>"]
                    stop = ["</s>", "[INST]", "User:", "assistant"]#, "`", "` "] #, '"', "\"\n\n", "</json>",'"\n']#, "01", "_v01"]
            )
        #print("\n")
        print(f"Prompt prep clash:{instruction}")
        out_long = output['output']['choices'][0]['text']
        #print(out_long)
        #print(output["output"]["usage"])
        print("*"*100)
        return out_long  
    
    
    async def _arun_clash(
    self,
    url: str,
    ):
        # Execute tool
        """Use the tool."""
        try:

            # Date and query params 
            #params = calculate_datetime_ranges(self.timezone)
            #params["query"] = parameters
            params={}
            params["hist"] = self.hist
            '''
            adv_query = await async_chain_together_m(
                prompt=PROMPT_ADVANCED_QUERY_PULL_GITHUB,
                params=params,
                max_tokens=768,
                temperature=0.1,
                top_p=1,
                top_k=50,
                repetition_penalty=1.1,
                model="meta-llama/Llama-3-70b-chat-hf",
                stop = ["</s>", "[INST]", "<|EOT|>", '"'],
                tool_name="ClickUp Advanced Query Search",
                callback_manager=self.async_callback_manager
                )
            '''
            # Log step clickup improver
            #if self.verbose:
            #    step = {"type": "clickup_query", "properties": {"query": adv_query}}
            #    await self.async_callback_manager.queue.put(f"#STEP{json.dumps(step)}STEP#")
            # Run the task extractor
            #pulls = await self._arun(query=adv_query)
            # Log step clickup improver
            #if self.verbose:
            #    step = {"type": f"{self.name}", "results": tasks}
            #    await self.async_callback_manager.queue.put(f"#STEP_OUT{json_step.dumps(step, ignore_nan=True)}STEP_OUT#")
            
            '''
            token = os.getenv("GITHUB_TOKEN")
            if not token:
                raise ValueError("Token is missing or empty")
            parameters["token"] = token.strip()
            '''
            search_results = self.search.search_issues(self.token, url)
            #print(search_results)
            #print(f"Commit result:{commits}")
            if "status" not in search_results.keys(): #query executed successfully
                items = search_results['items'] if 'items' in search_results else []
                #print(f"Found {len(pr_list)} pulls")
                extra_export = ", finally, you must notify the user the tasks were exported into a csv file inside the session folder, this can be located on the right menu" if self.export else ""
                answer_inst = f'''{self.ANSWER_SYSTEM_SUCCESS}<|start_header_id|>user<|end_header_id|>
                A total of {len(items)} issues were found in the search, mention that exact number in your answer.\n```{self.issues_to_markdown(search_results)}```\n.<|eot_id|><|start_header_id|>assistant<|end_header_id|>'''
                    
            else: # an error ocurred when calling the api endpoint
                #print(pr_list)
                #print(f"Could not perform read pull operation:\nMessage: {pr_list['message']}\nError code: {pr_list['status']}")
                answer_inst = f'''{self.ANSWER_SYSTEM_ERROR}<|start_header_id|>user<|end_header_id|>
                An error ocurred when attempting to execute the search query. The most common errors are:\n```{self.error_code}```\n.The message error is {search_results['message']} and the error code is {search_results['status']}.\n Explain what the possible cause of the error is and suggest a solution.<|eot_id|><|start_header_id|>assistant<|end_header_id|>'''              
            
            params = {
                "repo_owner": self.repo_owner,                
                "repo_name": self.repo_name,
                "hist": self.hist,
                "lang": self.lang,
                "task_answer_instruction": answer_inst,
                #"query": self.user_msg,
                #"date": datetime.now(pytz.timezone(self.timezone)).strftime("%A, %d of %B with time %H:%M of %Y"),
                "init_lang": "Por supuesto!\n" if self.lang == "Spanish" else "Certainly!\n"
            }
            
            # Stream
            '''
            result = await achain_nirvana(
            prompt=general_prompt,
            params=params,
            max_tokens=2048,
            temperature=0.1,
            top_p=0.85,
            top_k=40,
            repetition_penalty=1.1,
            model = "together|meta-llama/Llama-3-70b-chat-hf",
            stop = ["</s>", "[INST]", "<|EOT|>", "<|eot_id|>"],
            tool_name="SQL Generation",
            stream=False,
            )
            
            '''
            
            output =  together.Complete.create(
                    #model = "mistralai/Mixtral-8x7B-Instruct-v0.1", 
                    #model="cognitivecomputations/dolphin-2.5-mixtral-8x7b",
                    #model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",   
                    #model="NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT",   
                    #model="Qwen/Qwen1.5-14B-Chat",
                    prompt=answer_inst,
                    #model="Qwen/Qwen2-72B-Instruct",
                    model="meta-llama/Llama-3-70b-chat-hf",
                    #model="meta-llama/Llama-3-8b-chat-hf",
                    #model="deepseek-ai/deepseek-coder-33b-instruct",
                    #prompt=prompt_3,
                    #prompt=PROMPT_SATORI_FINAL_L.format(query=query, hist=t2),
                    temperature=0.1,
                    top_p=0.85,
                    top_k=40,
                    max_tokens=128,  
                    repetition_penalty=1.1,
                    #stop = ["</s>", "[INST]", "<|im_end|>"]
                    stop = ["</s>", "[INST]", "User:", "assistant"]#, "`", "` "] #, '"', "\"\n\n", "</json>",'"\n']#, "01", "_v01"]
            )
            #print("\n")
            print(f"Prompt run clash (answer inst):{answer_inst}")
            out_long = output['output']['choices'][0]['text']
            #print(f"Answer prompt output:\n{out_long}")
            print(output["output"]["usage"])
            print("*"*25)
            # Export
            if self.export:
                await self.async_callback_manager.queue.put(f"\nExport: [{self.user_msg}]({tasks['export_file']})")
            return {"finish": True, "response": output['output']['choices'][0]['text']}
        except Exception as e_tasks_clickup:
            ic(e_tasks_clickup)
            spaces = '- ' + '\n- '.join([f'{x}: {y}' for x, y in self.clickup_dicts['spaces'].items()])
            trace = traceback.format_exc()
            return {"error": "No tasks could be extracted.", "raise_msg": f"Is it likely the query used and the search parameters are not correct. Explain this query to the user in natural language and ask for feedback. These are the spaces available:\n{spaces}\nQuestion: {query}\nSearch query: {adv_query}", "trace": trace}


token = os.getenv("GITHUB_TOKEN")
if not token:
    raise ValueError("Token is missing or empty")

'''
user_q = "List all closed pull requests with sort them by longest running"
pr = GithubListPullRequest(token.strip(), "qgis", "QGIS", user_q)
string_json = asyncio.run(pr._aprep_clash([""]))
print(type(string_json))
diccionario = ast.literal_eval(string_json["parameters"])
print(diccionario)
res = asyncio.run(pr._arun_clash(diccionario))
'''

'''
user_q = "I want to see the modifications made to 6"
pr = GithubDiffPullRequest(token.strip(), "xiphos-dev", "Githubapp-demo", user_q)
string_json = asyncio.run(pr._aprep_clash([""]))
#print(string_json)
diccionario = ast.literal_eval(string_json)
#print(diccionario)
#print(diccionario)
res = asyncio.run(pr._arun_clash(diccionario))

analisis = GithubAnalyzeDiff(res["diff"][3])
res_analisis = asyncio.run(analisis._aprep_clash([""]))
print(res_analisis)
'''

'''
user_q = "List all of the commits made into the branch release-2_18"
pr = GithubCommit(token.strip(), "qgis", "QGIS", user_q)
string_json = asyncio.run(pr._aprep_clash([""]))
print(string_json)
diccionario = ast.literal_eval(string_json)
print(diccionario)
print(diccionario)
res = asyncio.run(pr._arun_clash(diccionario))
'''
'''
user_q = "Display all protected branches"
pr = GithubBranch(token.strip(), "qgis", "QGIS", user_q)
string_json = asyncio.run(pr._aprep_clash([""]))
print(string_json)
diccionario = ast.literal_eval(string_json)
print(diccionario)
res = asyncio.run(pr._arun_clash(diccionario))
'''
'''
#user_q = "Search for files with 'lorem' in their name or content, created since march 1 2023, sort them by date created in ascending order"
user_q = "Search for files with extension .py"
pr = GithubSearchFile(token.strip(), "qgis", "QGIS", user_q)
string_json = asyncio.run(pr._aprep_clash([""]))
print(string_json)

#diccionario = ast.literal_eval(string_json)
#print(diccionario)
res = asyncio.run(pr._arun_clash(string_json.replace('"','')))
#print(res)
'''

'''
user_q = "Search for bugs containing python in the title or comments"
pr = GithubIssue(token.strip(), "qgis", "QGIS", user_q)
string_json = asyncio.run(pr._aprep_clash([""]))
print(f"Query:{string_json}")
res = asyncio.run(pr._arun_clash(string_json))
'''

#file_path = "./message (74).txt"
#with open(file_path, 'r') as file:
 #   lines = file.readlines()

#segundo_prompt = "".join(lines)

#model = "meta-llama/Meta-Llama-3-70B"
#model= "meta-llama/Meta-Llama-3.1-70B-Instruct"
#user_prompt = "List the main changes mades"
#gdiff = GithubDiff("", user_prompt, model)
#prompt_system = gdiff.neo_format_prompt(UNIVERSAL_ANALYSIS.format(patch=None), "SYSTEM", "Llama" )
#print(prompt_system)
#res = asyncio.run(gdiff._aprep_clash([""]))
#res = asyncio.run(gdiff.inference(segundo_prompt, model))
#print(res['output']['choices'][0]['text'])