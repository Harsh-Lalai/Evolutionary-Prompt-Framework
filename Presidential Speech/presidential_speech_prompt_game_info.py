import openai
from openai import OpenAI
import os
import csv
import time
import random
from random import sample
import json
from typing import Tuple, List
import boto3
from botocore.config import Config
import traceback
import pandas as pd
import math
import concurrent.futures
from evaluate import load
bertscore = load("bertscore")
rogue = load("rouge")
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import download
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bertopic import BERTopic
from nltk.tokenize import sent_tokenize
from transformers import pipeline
from gensim.models import LdaModel
from gensim import corpora
from nrclex import NRCLex
from collections import Counter
import math
import textstat
import tiktoken

download('punkt')
download('averaged_perceptron_tagger')
download('stopwords')
download('vader_lexicon')
download('wordnet')
sia = SentimentIntensityAnalyzer()
sentiment_pipeline = pipeline("text-classification", model="siebert/sentiment-roberta-large-english")  
tokenizer = tiktoken.get_encoding("o200k_base")
emotion_pipeline = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")



# Set the API key from the environment variable
client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)


# session = boto3.Session(profile_name="AuraIntern") # use this if you are using Ada. 

# # session = boto3.Session()


# config = Config(

#     read_timeout=120, # corresponds to inference time limit set for Bedrock 

#     connect_timeout=120,

#     retries={

#         "max_attempts": 5,

#     },

# )


# bedrock = session.client(

#     service_name='bedrock-runtime',

#     region_name="us-east-1",

#     endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com",

#     config=config

# )

# Dictionary to store the ELO ratings of the prompts
elo_ratings_dict = {}
# Dictionary to store the history of the games in the leagues
league_dict = {}
#league number
current_league = 1
# Dictionary to store the top and bottom k prompts from each league
top_bottom_prompts_dict = {}
# Dictionary to store the human prompts which have been used in a league already
human_prompts_used_dict = {}
# Dictonary to store top k prompts across leagues for each role by using the final league
top_bottom_prompts_dict_across_league = {}
origin_league_prompts_dict = {}
top_bottom_prompts_dict_poc = {}


#Function for the API calls
def api_call_openai(model_name, prompt, temp, max_tokens):
    response = client.chat.completions.create(
        model= model_name,
        temperature = temp,
        messages=[{"role": "user", "content": prompt}],
        max_tokens= max_tokens
    )
    return response.choices[0].message.content.strip()

def api_call_openai_json(model_name, prompt, temp, max_tokens):
    response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=temp,
                max_tokens=max_tokens
            )
    

    try:
        return json.loads(response.choices[0].message.content.strip())
    except:
        return None


def api_call_claude(model, prompt,temp = 0.5, max_tokens=2048):

    api_template = {

        "modelId": "anthropic.claude-3-sonnet-20240229-v1:0",

        "contentType": "application/json",

        "accept": "application/json",

        "body": ""

    }



    body = {

        "max_tokens":  max_tokens,

        "stop_sequences": ["\n\nHuman:"],

        "anthropic_version": "bedrock-2023-05-31",

        "temperature": 0.5,

        "messages": [{"role": "user", "content": prompt}]

    }



    api_template["body"] = json.dumps(body)



    success = False

    for i in range(5):

        try:

            response = bedrock.invoke_model(**api_template)

            success = True

            break

        except:

            

            traceback.print_exc()

            time.sleep(1)



    if success:

        response_body = json.loads(response.get('body').read())

        return response_body["content"][0]["text"].strip()

    else:

        print("***exception!!!!!")

        return ""

# def get_debate_topic(no_of_topics, temp, max_tokens, file_path):
#     """
#     Parameters
#     ----------
#     no_of_topics: int
#         Number of topics required for the Debate Battle
    
#     Returns
#     --------
#     topics: list
#         List of topics generated for the Debate Battle
#     """

#     topics = []
#     topic_set = set()
#     last_topic_id = get_last_topic_id(file_path)

#     instruction_for_topics = """Please generate a unique debate topic which is debatable and interesting.
    
#     Make sure that the response starts with 'Topic: ' followed by the topic text"""
    
#     while not check_topics(no_of_topics, file_path):

#         response_text = api_call_openai("gpt-4o-mini-2024-07-18", instruction_for_topics, temp, max_tokens)

#         # Store all the prompts in a set to remove duplicates and store them in a csv file
#         if response_text.startswith("Topic:"):
#             topic = response_text[len("Topic:"):].strip().strip('"').strip('"')
#             if topic not in topic_set:
#                 topic_set.add(topic)
#                 with open(file_path, "a", newline="") as f:
#                     writer = csv.writer(f)
#                     if f.tell() == 0:
#                         writer.writerow(["Topic ID", "Topic"])
#                     last_topic_id += 1
#                     writer.writerow([last_topic_id, topic])

#         else:
#             print(response_text)
#             print("Response did not start with 'Topic:', retrying...")
        
#     topics = pd.read_csv(file_path)["Topic"].to_list()
#     # print(topics[:5])

#     # print(topics)
#     return topics

# def check_topics(no_of_topics, file_path):
#     """
#     Parameters
#     ----------
#     no_of_topics: int
#         Number of topics required for the Debate Battle
#     file_name: str
#         Name of the file to store the topics
        
#     Returns
#     --------
#     bool
#         True if the number of topics required have been generated, False otherwise
#     """

#     if not os.path.exists(file_path):
#         return False
    
#     df = pd.read_csv(file_path)

#     df.drop_duplicates("Topic")

#     df["Topic ID"] = range(0, len(df))

#     df.to_csv(file_path, index=False)

#     if len(df)>= no_of_topics+1:

#         return True

#     else:

#         return False

# def get_last_topic_id(file_path):
#     """
#     Parameters
#     ----------
#     file_name: str
#         Name of the file to read the last topic ID from
        
#     Returns
#     --------
#     last_topic_id: int
#         The last used topic ID
#     """

#     if not os.path.exists(file_path):
#         return 0
    
#     with open(file_path, "r") as f:
#         reader = csv.reader(f)
#         data = list(reader)
#         if len(data) > 1:
#             return int(data[-1][0])
#         else:
#             return 0

def get_prompts(no_of_prompts, league_no, temp, max_tokens, top_k_prompts=None, bottom_k_prompts=None, human=False, human_file=None):
    """
    Parameters
    ----------
    no_of_prompts: int
        Number of prompts required for the Debate Battle for the specific role
    league_no: int
        The league number for which the prompts are to be generated
    temp: float
        Temperature setting for the LLM
    max_tokens: int
        Maximum tokens setting for the LLM
    top_k_prompts: list
        List of top k prompts from the previous league for the specific role
    bottom_k_prompts: list
        List of bottom k prompts from the previous league for the specific role
    human: bool
        True if the prompts are to be generated by humans, False if the prompts are to be generated by LLM
    human_file: str
        The name of the file where the human prompts are stored

    Returns
    --------
    prompts: list
        List of prompts generated by LLM for the Debate Battle to be given to the specific role
    """

    global human_prompts_used_dict
    
    if human and not human_file:
        print("Error in input! Please provide the human_file for the human prompts to be read.")
        return None
    


    already_generated_prompts = set()
    # Check if there exists prompts{i}.csv file for all i <= league_no
    for i in range(1, league_no+1):
        if os.path.exists(f"prompts_{i}.csv"):
            already_generated_prompts.update(pd.read_csv(f"prompts_{i}.csv")["Prompt"].to_list())

    existing_prompts = set()
    if os.path.exists(f"prompts_{league_no}.csv"):
        # Read existing prompts from the CSV file
        with open(f"prompts_{league_no}.csv", mode='r') as file:  
            csv_reader = csv.reader(file)
            for row in csv_reader:
                if row[0] != "Prompt ID":
                    existing_prompts.add(row[1])
    
    if human:
        if (check_league(1)):
            # Get the specified role prompts from the league_results.json file
            with open('league_results.json', 'r') as json_file:
                data = json.load(json_file)

            prompts = set()

            for battle_no, battle_dict in data[str(league_no)].items():
                
                prompt_1 = battle_dict['prompt_1']['prompt']
                prompts.add(prompt_1)
                prompt_2 = battle_dict['prompt_2']['prompt']
                prompts.add(prompt_2)

            prompts = list(prompts)

            return prompts


        else: 
            prompts = set()
            with open(human_file, mode='r') as file:  
                csv_reader = csv.reader(file)
                for row in csv_reader:
                    if row[0] != "Prompt ID":
                        prompts.add(row[1])
            prompts = list(prompts)

            
            prompts = [prompt for prompt in prompts if prompt not in existing_prompts and prompt not in human_prompts_used_dict.values()]
            

            prompts = sample(prompts, no_of_prompts)
            return prompts

    last_prompt_id = get_last_prompt_id(f"prompts_{league_no}.csv")

    prompts = []
    prompt_set = existing_prompts.copy()
    
    
    while not check_prompts(no_of_prompts, f"prompts_{league_no}.csv"):
        times_runned = 0
        candidate_prompts_set = set()
        
        while(len(candidate_prompts_set)<4):
            # Generate the prompt using those features
            json_response = api_call_openai_json("gpt-4o-mini-2024-07-18", instruction_for_campaign_speech1(existing_prompts, candidate_prompts_set, top_k_prompts, bottom_k_prompts), temp, max_tokens)
            prompt = extract_prompt2(json_response)
            # Check if the prompt is none
            if prompt is None:
                print(json_response)
                print("Prompt not generated in correct format, retrying...")
                times_runned += 1
                continue

            if prompt not in prompt_set and prompt not in already_generated_prompts:
                candidate_prompts_set.add(prompt)
                # prompt_set.add(prompt)
                # existing_prompts.add(prompt)
                # last_prompt_id += 1
                # with open(f"prompts_{league_no}.csv", "a", newline="") as f:
                #     writer = csv.writer(f)
                #     if f.tell() == 0:
                #         writer.writerow(["Prompt ID", "Prompt", "Role"])
                #     writer.writerow([last_prompt_id, prompt, role])

            else:
                print("Prompt already exists in the prompt set or already generated prompts set, retrying...")
                times_runned += 1

        best_prompt, reason = finding_best_prompt_for_campaign(existing_prompts, list(candidate_prompts_set), temp, max_tokens, top_k_prompts, bottom_k_prompts)
        if best_prompt not in prompt_set and best_prompt is not None:
            prompt_set.add(best_prompt)
            existing_prompts.add(best_prompt)
            last_prompt_id += 1
            with open(f"prompts_{league_no}.csv", "a", newline="") as f:
                writer = csv.writer(f)
                if f.tell() == 0:
                    writer.writerow(["Prompt ID", "Prompt"])
                writer.writerow([last_prompt_id, best_prompt])
        else:
            print("Best prompt already exists in the prompt set, retrying...")
            times_runned += 1
        
    # print("Prompts generated successfully!")
    with open(f"prompts_{league_no}.csv", mode='r') as file:  
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if row[0] != "Prompt ID":
                prompts.append(row[1])

    # rogue_score()
    return prompts

# def bert_score():
#     """
#     Calculate the BERT score for the prompts generated using the prompts csv files and store it in a json file.
#     The index of the prompt is used as the key in the JSON file.
#     """
#     # Calculate the BERT score for the prompts_{current_league}.csv file with each other 
#     df_pro = pd.read_csv(f"prompts_{current_league}.csv")
#     df_pro = df_pro[df_pro["Role"] == "Pro"]
#     df_con = pd.read_csv(f"prompts_{current_league}.csv")
#     df_con = df_con[df_con["Role"] == "Con"]
    
#     pro_prompts = df_pro["Prompt"].to_list()
#     con_prompts = df_con["Prompt"].to_list()

#     # Function to check if a key exists in JSON file
#     def key_exists_in_json(filename, key):
#         if os.path.exists(filename):
#             with open(filename, "r") as f:
#                 data = json.load(f)
#             return str(key) in data
#         return False

#     # Process Pro Prompts
#     for i in range(len(pro_prompts) - 1):
#         references = pro_prompts[i + 1:]
#         predictions = [pro_prompts[i]] * len(references)
        
#         # Check if the key for this prompt already exists in the JSON file
#         if key_exists_in_json(f"rogue_score_{current_league}_pro.json", i):
#             print(f"Skipping calculation for pro prompt {i}, already exists in the file.")
#             continue
        
#         print("going to calculate results")
#         # Compute the BERT score
#         results = bertscore.compute(predictions=predictions, references=references, lang="en", model_type="distilbert-base-uncased")
#         print("results computed")
#         print(i)

#         # Check if the JSON file exists and update it
#         if os.path.exists(f"rogue_score_{current_league}_pro.json"):
#             with open(f"rogue_score_{current_league}_pro.json", "r") as f:
#                 data = json.load(f)
#             data[str(i)] = results  # Store the results under the index `i`
#         else:
#             data = {str(i): results}

#         # Write back to the JSON file
#         with open(f"rogue_score_{current_league}_pro.json", "w") as f:
#             json.dump(data, f)
    
#     # Process Con Prompts
#     for i in range(len(con_prompts) - 1):
#         references = con_prompts[i + 1:]
#         predictions = [con_prompts[i]] * len(references)

#         # Check if the key for this prompt already exists in the JSON file
#         if key_exists_in_json(f"bert_score_{current_league}_con.json", i):
#             print(f"Skipping calculation for con prompt {i}, already exists in the file.")
#             continue
        
#         # Compute the BERT score
#         results = bertscore.compute(predictions=predictions, references=references, lang="en", model_type="distilbert-base-uncased")

#         # Check if the JSON file exists and update it
#         if os.path.exists(f"bert_score_{current_league}_con.json"):
#             with open(f"bert_score_{current_league}_con.json", "r") as f:
#                 data = json.load(f)
#             data[str(i)] = results  # Store the results under the index `i`
#         else:
#             data = {str(i): results}

#         # Write back to the JSON file
#         with open(f"bert_score_{current_league}_con.json", "w") as f:
#             json.dump(data, f)

#     # Process cross-league prompts
#     for i in range(1, current_league):
#         if os.path.exists(f"prompts_{current_league-i}.csv"):
#             df_pro_prev = pd.read_csv(f"prompts_{current_league-i}.csv")
#             df_pro_prev = df_pro_prev[df_pro_prev["Role"] == "Pro"]
#             df_con_prev = pd.read_csv(f"prompts_{current_league-i}.csv")
#             df_con_prev = df_con_prev[df_con_prev["Role"] == "Con"]
            
#             pro_prompts_prev = df_pro_prev["Prompt"].to_list()
#             con_prompts_prev = df_con_prev["Prompt"].to_list()

#             # Process cross-league pro prompts
#             for j in range(len(pro_prompts)):
#                 if key_exists_in_json(f"rogue_score_{current_league}_pro_{current_league-i}.json", j):
#                     print(f"Skipping calculation for cross-league pro prompt {j}, already exists in the file.")
#                     continue
                
#                 references = pro_prompts_prev
#                 predictions = [pro_prompts[j]] * len(references)
                
#                 results = bertscore.compute(predictions=predictions, references=references, lang="en", model_type="distilbert-base-uncased")
                
#                 # Check if the JSON file exists and update it
#                 if os.path.exists(f"rogue_score_{current_league}_pro_{current_league-i}.json"):
#                     with open(f"rogue_score_{current_league}_pro_{current_league-i}.json", "r") as f:
#                         data = json.load(f)
#                     data[str(j)] = results  # Store the results under the index `j`
#                 else:
#                     data = {str(j): results}

#                 # Write back to the JSON file
#                 with open(f"rogue_score_{current_league}_pro_{current_league-i}.json", "w") as f:
#                     json.dump(data, f)
            
#             # Process cross-league con prompts
#             for j in range(len(con_prompts)):
#                 if key_exists_in_json(f"bert_score_{current_league}_con_{current_league-i}.json", j):
#                     print(f"Skipping calculation for cross-league con prompt {j}, already exists in the file.")
#                     continue
                
#                 references = con_prompts_prev
#                 predictions = [con_prompts[j]] * len(references)
                
#                 results = bertscore.compute(predictions=predictions, references=references, lang="en", model_type="distilbert-base-uncased")
                
#                 # Check if the JSON file exists and update it
#                 if os.path.exists(f"bert_score_{current_league}_con_{current_league-i}.json"):
#                     with open(f"bert_score_{current_league}_con_{current_league-i}.json", "r") as f:
#                         data = json.load(f)
#                     data[str(j)] = results  # Store the results under the index `j`
#                 else:
#                     data = {str(j): results}

#                 # Write back to the JSON file
#                 with open(f"bert_score_{current_league}_con_{current_league-i}.json", "w") as f:
#                     json.dump(data, f)
# def rogue_score():
#     """
#     Calculate the ROUGE score for the prompts generated using the prompts csv files and store it in a json file.
#     The index of the prompt is used as the key in the JSON file.
#     Separate calculations for 'Pro' and 'Con' roles.
#     """
#     # Load the current league prompts for "Pro" and "Con"
#     df_pro = pd.read_csv(f"prompts_{current_league}.csv")
#     df_pro = df_pro[df_pro["Role"] == "Pro"]
#     df_con = pd.read_csv(f"prompts_{current_league}.csv")
#     df_con = df_con[df_con["Role"] == "Con"]
    
#     pro_prompts = df_pro["Prompt"].to_list()
#     con_prompts = df_con["Prompt"].to_list()

#     # Function to check if a key exists in JSON file
#     def key_exists_in_json(filename, key):
#         if os.path.exists(filename):
#             with open(filename, "r") as f:
#                 data = json.load(f)
#             return str(key) in data
#         return False

#     # Process Pro Prompts: Compare each "Pro" prompt with subsequent "Pro" prompts
#     for i in range(len(pro_prompts) - 1):
#         for j in range(i + 1, len(pro_prompts)):  # Compare pro_prompts[i] with pro_prompts[j] individually
#             key = f"{i}_{j}"

#             # Check if the key for this prompt pair already exists in the JSON file
#             if key_exists_in_json(f"rogue_score_{current_league}_pro.json", key):
#                 print(f"Skipping calculation for pro prompt pair {i}-{j}, already exists in the file.")
#                 continue
            
#             # Compute the ROUGE score between pro_prompts[i] and pro_prompts[j]
#             results = rogue.compute(predictions=[pro_prompts[i]], references=[pro_prompts[j]])

#             # Check if the JSON file exists and update it
#             if os.path.exists(f"rogue_score_{current_league}_pro.json"):
#                 with open(f"rogue_score_{current_league}_pro.json", "r") as f:
#                     data = json.load(f)
#                 data[key] = results  # Store the results under the key `i_j`
#             else:
#                 data = {key: results}

#             # Write back to the JSON file
#             with open(f"rogue_score_{current_league}_pro.json", "w") as f:
#                 json.dump(data, f)

#     # Process Con Prompts: Compare each "Con" prompt with subsequent "Con" prompts
#     for i in range(len(con_prompts) - 1):
#         for j in range(i + 1, len(con_prompts)):  # Compare con_prompts[i] with con_prompts[j] individually
#             key = f"{i}_{j}"

#             # Check if the key for this prompt pair already exists in the JSON file
#             if key_exists_in_json(f"rogue_score_{current_league}_con.json", key):
#                 print(f"Skipping calculation for con prompt pair {i}-{j}, already exists in the file.")
#                 continue
            
#             # Compute the ROUGE score between con_prompts[i] and con_prompts[j]
#             results = rogue.compute(predictions=[con_prompts[i]], references=[con_prompts[j]])

#             # Check if the JSON file exists and update it
#             if os.path.exists(f"rogue_score_{current_league}_con.json"):
#                 with open(f"rogue_score_{current_league}_con.json", "r") as f:
#                     data = json.load(f)
#                 data[key] = results  # Store the results under the key `i_j`
#             else:
#                 data = {key: results}

#             # Write back to the JSON file
#             with open(f"rogue_score_{current_league}_con.json", "w") as f:
#                 json.dump(data, f)

#     # Process cross-league "Pro" prompts
#     for i in range(1, current_league):
#         if os.path.exists(f"prompts_{current_league-i}.csv"):
#             df_pro_prev = pd.read_csv(f"prompts_{current_league-i}.csv")
#             df_pro_prev = df_pro_prev[df_pro_prev["Role"] == "Pro"]
#             pro_prompts_prev = df_pro_prev["Prompt"].to_list()

#             # Compare each current "Pro" prompt with previous league's "Pro" prompts
#             for j in range(len(pro_prompts)):
#                 for k in range(len(pro_prompts_prev)):
#                     key = f"{j}_{k}"
                    
#                     if key_exists_in_json(f"rogue_score_{current_league}_pro_{current_league-i}.json", key):
#                         print(f"Skipping calculation for cross-league pro prompt pair {j}-{k}, already exists in the file.")
#                         continue

#                     results = rogue.compute(predictions=[pro_prompts[j]], references=[pro_prompts_prev[k]])

#                     # Check if the JSON file exists and update it
#                     if os.path.exists(f"rogue_score_{current_league}_pro_{current_league-i}.json"):
#                         with open(f"rogue_score_{current_league}_pro_{current_league-i}.json", "r") as f:
#                             data = json.load(f)
#                         data[key] = results
#                     else:
#                         data = {key: results}

#                     # Write back to the JSON file
#                     with open(f"rogue_score_{current_league}_pro_{current_league-i}.json", "w") as f:
#                         json.dump(data, f)

#     # Process cross-league "Con" prompts
#     for i in range(1, current_league):
#         if os.path.exists(f"prompts_{current_league-i}.csv"):
#             df_con_prev = pd.read_csv(f"prompts_{current_league-i}.csv")
#             df_con_prev = df_con_prev[df_con_prev["Role"] == "Con"]
#             con_prompts_prev = df_con_prev["Prompt"].to_list()

#             # Compare each current "Con" prompt with previous league's "Con" prompts
#             for j in range(len(con_prompts)):
#                 for k in range(len(con_prompts_prev)):
#                     key = f"{j}_{k}"

#                     if key_exists_in_json(f"rogue_score_{current_league}_con_{current_league-i}.json", key):
#                         print(f"Skipping calculation for cross-league con prompt pair {j}-{k}, already exists in the file.")
#                         continue

#                     results = rogue.compute(predictions=[con_prompts[j]], references=[con_prompts_prev[k]])

#                     # Check if the JSON file exists and update it
#                     if os.path.exists(f"rogue_score_{current_league}_con_{current_league-i}.json"):
#                         with open(f"rogue_score_{current_league}_con_{current_league-i}.json", "r") as f:
#                             data = json.load(f)
#                         data[key] = results
#                     else:
#                         data = {key: results}

#                     # Write back to the JSON file
#                     with open(f"rogue_score_{current_league}_con_{current_league-i}.json", "w") as f:
#                         json.dump(data, f)

# Function to extract prompt text from JSON response
def extract_prompt1(json_response):
    """
    Extracts the prompt text from the JSON response if it meets all criteria.
    """
    # Define the required keys
    required_keys = ["Reason for uniqueness", "Reason for effectiveness", "Prompt Text"]

    # Check if all required keys are present and their values are non-zero or non-empty
    if all(key in json_response for key in required_keys) and all(json_response[key] for key in required_keys):
        # Extract the prompt text
        prompt_text = json_response["Prompt Text"].strip().strip('"').strip("'").strip().strip("**").strip("*").strip().strip('"').strip()
        return prompt_text
    else:
        # If any key is missing or has an empty/zero value, return None
        return None

def instruction_for_campaign_speech1(existing_prompts=None, candidates=None, top_k_prompts=None, bottom_k_prompts=None):

    game_info_win_top_k = {}
    game_info_lose_top_k = {}
    game_info_win_bot_k = {}
    game_info_lose_bot_k = {}

    

    if top_k_prompts is not None and bottom_k_prompts is not None:
        with open('league_results.json', 'r') as json_file:
            data = json.load(json_file)
        # Get a winning and losing game for each of the top and bottom k prompt and store it in a dictionary. Get these from the league_results.json file from the league which has the prompt required. So you have to iterate through the league no from 1 to current league -1
        for i in range(len(top_k_prompts)):
            for j in range (1, current_league):
                if str(j) in data:
                    for battle_no, battle_dict in data[str(j)].items():
                        if battle_dict['prompt_1']['prompt'] == top_k_prompts[i]:
                            if battle_dict["winner_prompt"] == top_k_prompts[i]:
                                game_info_win_top_k[i] = battle_dict
                            else:
                                game_info_lose_top_k[i] = battle_dict
                            
                        elif battle_dict['prompt_2']['prompt'] == top_k_prompts[i]:
                            if battle_dict["winner_prompt"] == top_k_prompts[i]:
                                game_info_win_top_k[i] = battle_dict
                            else:
                                game_info_lose_top_k[i] = battle_dict
        
        for i in range(len(bottom_k_prompts)):
            for j in range (1, current_league):
                if str(j) in data:
                    for battle_no, battle_dict in data[str(j)].items():
                        if battle_dict['prompt_1']['prompt'] == bottom_k_prompts[i]:
                            if battle_dict["winner_prompt"] == bottom_k_prompts[i]:
                                game_info_win_bot_k[i] = battle_dict
                            else:
                                game_info_lose_bot_k[i] = battle_dict
                            
                        elif battle_dict['prompt_2']['prompt'] == bottom_k_prompts[i]:
                            if battle_dict["winner_prompt"] == bottom_k_prompts[i]:
                                game_info_win_bot_k[i] = battle_dict
                            else:
                                game_info_lose_bot_k[i] = battle_dict
        

    # Initialize text variables
    existing_prompts_text = ""
    top_prompts_text = ""
    bottom_prompts_text = ""

    # Prepare existing prompts text if available
    if existing_prompts:
        existing_and_candidate_prompts = existing_prompts.union(candidates or set())
        if existing_and_candidate_prompts:
            existing_prompts_text = "\n".join([f'{i+1}. "{prompt}"' for i, prompt in enumerate(existing_and_candidate_prompts)])

    # Prepare top and bottom prompts text if available in which for each of the top and bottom k prompts, you have to provide the winning and losing game information which includes their prompt speech, opponent prompt and speech, winner with reason. First you have to extract the prompt, speech, winner with reason from the game info dict.
    if top_k_prompts is not None and bottom_k_prompts is not None:
        for i in range(len(top_k_prompts)):

            top_prompts_text += f'{i+1}. "{top_k_prompts[i]}"\n'

            top_prompt = top_k_prompts[i]
            if i in game_info_win_top_k:
                game_info = game_info_win_top_k[i]
                # Check if the prompt is prompt_1 or prompt_2 and then extract own and opponent speech
                if game_info['prompt_1']['prompt'] == top_prompt:
                    opponent_prompt = game_info['prompt_2']['prompt']
                    opponent_speech = game_info["speech_2"]["speech_2_text"]
                    own_speech = game_info['speech_1']['speech_1_text']
                
                elif game_info['prompt_2']['prompt'] == top_prompt:
                    opponent_prompt = game_info['prompt_1']['prompt']
                    opponent_speech = game_info["speech_1"]["speech_1_text"]
                    own_speech = game_info['speech_2']['speech_2_text']
                    

                reason_for_win = game_info['reason']

                # Add the extracted information to the top_prompts_text
                # First add a string winning game then the speech, followed by opponent speech, followed by reason for win
                top_prompts_text += f"Winning Speech:\nOwn Prompt: {top_prompt}\nOwn Speech: {own_speech}\nOpponent Prompt: {opponent_prompt}\nOpponent Speech: {opponent_speech}\nReason for Win: {reason_for_win}\n\n"
            
            if i in game_info_lose_top_k:
                game_info = game_info_lose_top_k[i]
                # Check if the prompt is prompt_1 or prompt_2 and then extract own and opponent speech
                if game_info['prompt_1']['prompt'] == top_prompt:
                    opponent_prompt = game_info['prompt_2']['prompt']
                    opponent_speech = game_info["speech_2"]["speech_2_text"]
                    own_speech = game_info['speech_1']['speech_1_text']
                
                elif game_info['prompt_2']['prompt'] == top_prompt:
                    opponent_prompt = game_info['prompt_1']['prompt']
                    opponent_speech = game_info["speech_1"]["speech_1_text"]
                    own_speech = game_info['speech_2']['speech_2_text']


                reason_for_lose = game_info['reason']

                # Add the extracted information to the top_prompts_text
                # First add a string winning game then the speech, followed by opponent speech, followed by reason for win
                top_prompts_text += f"Losing Speech:\nOwn Prompt: {top_prompt}\nOwn Speech: {own_speech}\nOpponent Prompt: {opponent_prompt}\nOpponent Speech: {opponent_speech}\nReason for Lose: {reason_for_lose}\n\n"


        for i in range(len(bottom_k_prompts)):
            bottom_prompts_text += f'{i+1}. "{bottom_k_prompts[i]}"\n'

            bottom_prompt = bottom_k_prompts[i]
            if i in game_info_win_bot_k:
                game_info = game_info_win_bot_k[i]
                # Check if the prompt is prompt_1 or prompt_2 and then extract own and opponent speech
                if game_info['prompt_1']['prompt'] == bottom_prompt:
                    opponent_prompt = game_info['prompt_2']['prompt']
                    opponent_speech = game_info["speech_2"]["speech_2_text"]
                    own_speech = game_info['speech_1']['speech_1_text']
                    
                
                elif game_info['prompt_2']['prompt'] == bottom_prompt:
                    opponent_prompt = game_info['prompt_1']['prompt']
                    opponent_speech = game_info["speech_1"]["speech_1_text"]
                    own_speech = game_info['speech_2']['speech_2_text']
                    

                reason_for_win = game_info['reason']

                # Add the extracted information to the bottom_prompts_text
                # First add a string winning game then the speech, followed by opponent speech, followed by reason for win
                bottom_prompts_text += f"Winning Speech:\nOwn Prompt: {bottom_prompt}\nOwn Speech: {own_speech}\nOpponent Prompt: {opponent_prompt}\nOpponent Speech: {opponent_speech}\nReason for Win: {reason_for_win}\n\n"
            
            if i in game_info_lose_bot_k:
                game_info = game_info_lose_bot_k[i]
                # Check if the prompt is prompt_1 or prompt_2 and then extract own and opponent speech
                if game_info['prompt_1']['prompt'] == bottom_prompt:
                    opponent_prompt = game_info['prompt_2']['prompt']
                    opponent_speech = game_info["speech_2"]["speech_2_text"]
                    own_speech = game_info['speech_1']['speech_1_text']
                    
                
                elif game_info['prompt_2']['prompt'] == bottom_prompt:
                    opponent_prompt = game_info['prompt_1']['prompt']
                    opponent_speech = game_info["speech_1"]["speech_1_text"]
                    own_speech = game_info['speech_2']['speech_2_text']
                    

                reason_for_lose = game_info['reason']

                # Add the extracted information to the bottom_prompts_text
                # First add a string winning game then the speech, followed by opponent speech, followed by reason for win
                bottom_prompts_text += f"Losing Speech:\nOwn Prompt: {bottom_prompt}\nOwn Speech: {own_speech}\nOpponent Prompt: {opponent_prompt}\nOpponent Speech: {opponent_speech}\nReason for Lose: {reason_for_lose}\n\n"


    # Build instructions based on conditions
    if top_k_prompts is None and bottom_k_prompts is None:
        if not existing_prompts:
            instructions = (
                "Help me create a new and effective prompt to guide an AI model in generating a campaign speech for a candidate for presidency aiming to secure majority votes.\n\n"
                "We have a framework that evaluates the effectiveness of prompts.\n"
                "Format your response in JSON with the following fields:\n"
                "{\n"
                '  "Reason for uniqueness": "[Explain why this prompt is unique]",\n'
                '  "Reason for effectivess": "[Explain why this prompt is effective for the task]",\n'
                '  "Prompt Text": "[Write the new prompt here without starting with \'Prompt:\']"\n'
                "}\n\n"
                "Ensure all fields are filled with meaningful content."
            )
        else:
            instructions = (
                "Help me create a new and effective prompt to guide an AI model in generating a campaign speech for a candidate for presidency aiming to secure majority votes.\n\n"
                "We have a framework that evaluates the effectiveness of prompts.\n"
                "Here is a list of existing prompts that have already been created:\n"
                f"{existing_prompts_text}\n\n"
                "- Do not make it structurally or syntactically similar to the existing prompts.\n"
                "- The prompt should be more effective than the existing prompts.\n"
                "Format your response in JSON with the following fields:\n"
                "{\n"
                '  "Reason for uniqueness": "[Explain why this prompt is unique]",\n'
                '  "Reason for effectiveness": "[Explain why this prompt is more effective for the task than the existing prompts]",\n'
                '  "Prompt Text": "[Write the new prompt here without starting with \'Prompt:\']"\n'
                "}\n\n"
                "Ensure all fields are filled with meaningful content."
            )
    else:
        if not existing_prompts:
            instructions = (
                "Help me create a new and effective prompt to guide an AI model in generating a campaign speech for a candidate for presidency aiming to secure majority votes.\n\n"
                "We have a framework that evaluates the effectiveness of prompts.\n"
                "Here are some of the most and least effective prompts already generated along with their winning and losing speeches if any:\n\n"
                f"Most effective prompts:\n{top_prompts_text}\n\Least effective prompts:\n{bottom_prompts_text}\n\n"
                "- Get inspired by the ideas from the most effective prompts but do not make it structurally or syntactically similar to the prompts listed above.\n"
                "- The prompt should be more effective than the prompts listed above.\n"
                "Format your response in JSON with the following fields:\n"
                "{\n"
                '  "Reason for uniqueness": "[Explain why this prompt is unique]",\n'
                '  "Reason for effectiveness": "[Explain why this prompt is more effective for the task than the prompts listed above]",\n'
                '  "Prompt Text": "[Write the new prompt here without starting with \'Prompt:\']"\n'
                "}\n\n"
                "Ensure all fields are filled with meaningful content."
            )


        else:
            instructions = (
                "Help me create a new and effective prompt to guide an AI model in generating a campaign speech for a candidate for presidency aiming to secure majority votes.\n\n"
                "We have a framework that evaluates the effectiveness of prompts.\n"
                "Here are some of the most and least effective prompts already generated along with their winning and losing speeches if any:\n\n"
                f"Most effective prompts:\n{top_prompts_text}\n\Least effective prompts:\n{bottom_prompts_text}\n\n"
                "Here is a list of existing prompts that have already been created:\n"
                f"{existing_prompts_text}\n\n"
                "- Get inspired by the ideas from the most effective prompts but do not make it structurally or syntactically similar to the prompts listed above.\n"
                "- The prompt should be more effective than the prompts listed above.\n"
                "Format your response in JSON with the following fields:\n"
                "{\n"
                '  "Reason for uniqueness": "[Explain why this prompt is unique]",\n'
                '  "Reason for effectiveness": "[Explain why this prompt is more effective for the task than the prompts listed above]",\n'
                '  "Prompt Text": "[Write the new prompt here without starting with \'Prompt:\']"\n'
                "}\n\n"
                "Ensure all fields are filled with meaningful content."
            )

    return instructions

def instructions_for_campaign_speech2(top_k_prompts):

    # Prepare top and bottom prompts text if available
    top_prompts_text = ""
    if top_k_prompts:
        top_prompts_text = "\n".join([f'{i+1}. "{prompt}"' for i, prompt in enumerate(top_k_prompts)])
    
    # Build instructions based on conditions
    instructions = (
    "We have a framework that evaluates the effectiveness of prompts for generating a campaign speech for a presidential candidate aiming to secure the majority of votes.\n\n"
    "Here are some of the most effective prompts already generated:\n\n"
    "Most effective prompts:\n"
    f"{top_prompts_text}\n\n"
    "Now, I'd like you to analyze and identify the common, underlying strategy behind ALL the prompts that makes them perform better than alternatives."
    "Prepare your response in a way that will be useful as guidance for a prompt engineer to write a prompt that works effectively for campaign speeches. You should not provide the specific structure of the prompts, but instead, focus on the overarching gist or strategy that works."
    "Format your response as follows:\n"
    "Strategy: [Write the common strategy here]\n\n"
    )

    return instructions


# Function to extract the strategy from the response
def extract_strategy(response):
    # Check if the response starts with "Strategy:" and also check if there is a list of features (each strings)
    # First strip the response
    response = response.strip()
    if response.startswith("Strategy:") or response.startswith("***Strategy:***") or response.startswith("**Strategy:**"):
        # Extract the strategy
        strategy = response.split(":", 1)[1].strip()
        return strategy
    else: 
        return None

# Function to return a prompt which includes strategy of top prompts and asking to generate a prompt which it feels can beat the top prompts
def instruction_for_campaign_speech3(strategy):
    
    instructions = (
        "You are tasked with coming up with a prompt that will be used by an AI model in generating a campaign speech for a candidate for presidency aiming to secure majority votes. Abide by the following strategy:"
        f"{strategy}\n\n"
        "Format your response in JSON with the following fields:\n"
        "{\n"
        '  "Reason for effectiveness": "[Explain why this prompt is effective]",\n'
        '  "Prompt Text": "[Write the new prompt here without starting with \'Prompt:\']"\n'
        "}\n\n"
        "Ensure all fields are filled with meaningful content."
    )

    return instructions
    
# Function to extract prompt text from JSON response
def extract_prompt2(json_response):
    """
    Extracts the prompt text from the JSON response if it meets all criteria.
    """

    if json_response == None:
        return None
    # Define the required keys
    required_keys = ["Reason for effectiveness", "Prompt Text"]

    # Check if all required keys are present and their values are non-zero or non-empty
    if all(key in json_response for key in required_keys) and all(json_response[key] for key in required_keys):
        # Extract the prompt text
        prompt_text = json_response["Prompt Text"].strip().strip('"').strip("'").strip().strip("**").strip("*").strip().strip('"').strip()
        return prompt_text
    else:
        # If any key is missing or has an empty/zero value, return None
        return None

def check_prompts(no_of_prompts, file_name):
    """
    Parameters
    ----------
    no_of_prompts: int
        Number of prompts required for the Debate Battle for the specific role
    file_name: str
        The name of the file where the prompts are stored
    role: str
        The role for which the prompts are to be generated

    Returns
    --------
    bool
        True if the number of prompts stored in the prompts.csv are more than or equal to the number of prompts required, False otherwise
    """

    if not os.path.exists(file_name):
        return False
    
    with open(file_name, "r") as f:
        reader = csv.reader(f)
        data = list(reader)
        count = 0
        for row in data:
            if row[0] != "Prompt ID":
                count += 1
        if count >= no_of_prompts:
            return True
        else:
            return False

def get_last_prompt_id(file_name):
    """
    Parameters
    ----------
    file_name: str
        Name of the file to read the last prompt ID from
    role: str
        The role for which the last prompt ID is to be fetched

    Returns
    --------
    last_prompt_id: int
        The last used prompt ID for the specific role
    """

    if not os.path.exists(file_name):
        return 0
    
    with open(file_name, "r") as f:
        reader = csv.reader(f)
        data = list(reader)
        last_id = 0
        for row in data:
            if row[0] != "Prompt ID":
                last_id = int(row[0])
        return last_id
    
def extract_best_prompt(json_response):
    """
    Extracts the best prompt and reason from the JSON response if it meets all criteria.
    """

    if json_response == None:
        return None, None
    
    # Define the required keys
    required_keys = ["Reason for the prompt being most effective", "Prompt Text"]

    # Check if all required keys are present and their values are non-zero or non-empty
    if all(key in json_response for key in required_keys) and all(json_response[key] for key in required_keys):
        # Extract the prompt text and reason
        reason = json_response["Reason for the prompt being most effective"].strip().strip('"').strip("'").strip().strip("**").strip("*").strip().strip('"').strip()
        prompt_text = json_response["Prompt Text"].strip().strip('"').strip("'").strip().strip("**").strip("*").strip().strip('"').strip()
        return prompt_text, reason
    else:
        # If any key is missing or has an empty/zero value, return None
        return None, None


def finding_best_prompt_for_campaign(existing_prompts, candidates, temp, max_tokens, top_k_prompts=None, bottom_k_prompts=None):
    """
    Function to find the best prompt for generating a campaign speech as a candidate for president.

    Parameters
    ----------
    existing_prompts: list
        List of existing prompts already generated.
    candidates: list
        List of candidate prompts to evaluate (should contain exactly 3 candidates).
    temp: float
        The temperature setting for the OpenAI API call.
    max_tokens: int
        The maximum tokens for the OpenAI API call.
    top_k_prompts: list, optional
        List of top-performing prompts.
    bottom_k_prompts: list, optional
        List of worst-performing prompts.

    Returns
    -------
    best_prompt: str
        The best candidate prompt selected.
    reason: str
        The reason for selecting the best prompt.
    """

    # Prepare existing prompts text if available
    existing_prompts_text = ""
    if existing_prompts:
        existing_prompts_text = "\n".join([f'{i+1}. "Prompt: {prompt}"' for i, prompt in enumerate(existing_prompts)])
    
    # Prepare top and bottom prompts text if available
    top_prompts_text = ""
    bottom_prompts_text = ""
    if top_k_prompts:
        top_prompts_text = "\n".join([f'{i+1}. "Prompt: {prompt}"' for i, prompt in enumerate(top_k_prompts)])
    if bottom_k_prompts:
        bottom_prompts_text = "\n".join([f'{i+1}. "Prompt: {prompt}"' for i, prompt in enumerate(bottom_k_prompts)])
    
    # Build instruction based on conditions
    if top_k_prompts is None and bottom_k_prompts is None:
        if existing_prompts:
            instruction_to_select_best_prompt = (
                "Help me select a prompt from the 3 candidate prompts that I can give to an AI model in generating a campaign speech for a candidate for presidency aiming to secure majority votes.\n\n"
                "We have a framework that evaluates the effectiveness of prompts.\n"
                "Here are the 3 candidate prompts:\n"
                f"1. \"{candidates[0]}\"\n"
                f"2. \"{candidates[1]}\"\n"
                f"3. \"{candidates[2]}\"\n\n"
                "The existing prompts are:\n"
                f"{existing_prompts_text}\n\n"
                "Consider the following criteria while evaluating each response:\n"
                "- **Quality**: The prompt should demonstrate superior effectiveness over "
                "any of the other candidate prompts as well as the existing prompts.\n\n"
                "Make sure that the response is in JSON format with the following fields:\n"
                "{\n"
                '  "Reason for the prompt being most effective": "[Explain why this prompt is most effective among the candidates]",\n'
                '  "Prompt Text": "[Write only your prompt here and don\'t start with \'Prompt:\']"\n'
                "}\n\n"
                "Ensure all fields are provided and have non-zero or non-empty values."
            )
        else:
            instruction_to_select_best_prompt = (
                "Help me select a prompt from the 3 candidate prompts that I can give to an AI model in generating a campaign speech for a candidate for presidency aiming to secure majority votes.\n\n"
                "We have a framework that evaluates the effectiveness of prompts.\n"
                "Here are the 3 candidate prompts:\n"
                f"1. \"{candidates[0]}\"\n"
                f"2. \"{candidates[1]}\"\n"
                f"3. \"{candidates[2]}\"\n\n"
                "Consider the following criteria while evaluating each response:\n"
                "- **Quality**: The prompt should demonstrate superior effectiveness over "
                "any of the other candidate prompts.\n\n"
                "Make sure that the response is in JSON format with the following fields:\n"
                "{\n"
                '  "Reason for the prompt being most effective": "[Explain why this prompt is most effective among the candidates]",\n'
                '  "Prompt Text": "[Write only your prompt here and don\'t start with \'Prompt:\']"\n'
                "}\n\n"
                "Ensure all fields are provided and have non-zero or non-empty values."
            )
    else:
        if existing_prompts:
            instruction_to_select_best_prompt = (
                "Help me select a prompt from the 3 candidate prompts that I can give to an AI model in generating a campaign speech for a candidate for presidency aiming to secure majority votes.\n\n"
                "We have a framework that evaluates the effectiveness of prompts.\n"
                "Here are the 3 candidate prompts:\n"
                f"1. \"{candidates[0]}\"\n"
                f"2. \"{candidates[1]}\"\n"
                f"3. \"{candidates[2]}\"\n\n"
                "The existing prompts are:\n"
                f"{existing_prompts_text}\n\n"
                "Additionally, here are some examples of the most and least effective prompts:\n\n"
                "Most effective prompts:\n"
                f"{top_prompts_text}\n\n"
                "Least effective prompts:\n"
                f"{bottom_prompts_text}\n\n"
                "Consider the following criteria while evaluating each response:\n"
                "- **Quality**: The prompt should demonstrate superior effectiveness over "
                "all other candidate prompts and the prompts listed above.\n\n"
                "Make sure that the response is in JSON format with the following fields:\n"
                "{\n"
                '  "Reason for the prompt being most effective": "[Explain why this prompt is most effective among the candidates]",\n'
                '  "Prompt Text": "[Write only your prompt here and don\'t start with \'Prompt:\']"\n'
                "}\n\n"
                "Ensure all fields are provided and have non-zero or non-empty values."
            )
        else:
            instruction_to_select_best_prompt = (
                "Help me select a prompt from the 3 candidate prompts that I can give to an AI model in generating a campaign speech for a candidate for presidency aiming to secure majority votes.\n\n"
                "We have a framework that evaluates the effectiveness of prompts.\n"
                "Here are the 3 candidate prompts:\n"
                f"1. \"{candidates[0]}\"\n"
                f"2. \"{candidates[1]}\"\n"
                f"3. \"{candidates[2]}\"\n\n"
                "Additionally, here are some examples of the most and least effective prompts:\n\n"
                "Most effective prompts:\n"
                f"{top_prompts_text}\n\n"
                "Least effective prompts:\n"
                f"{bottom_prompts_text}\n\n"
                "Consider the following criteria while evaluating each response:\n"
                "- **Quality**: The prompt should demonstrate superior effectiveness over "
                "all other candidate prompts and the prompts listed above\n\n"
                "Make sure that the response is in JSON format with the following fields:\n"
                "{\n"
                '  "Reason for the prompt being most effective": "[Explain why this prompt is most effective among the candidates]",\n'
                '  "Prompt Text": "[Write only your prompt here and don\'t start with \'Prompt:\']"\n'
                "}\n\n"
                "Ensure all fields are provided and have non-zero or non-empty values."
            )

    # API calls to evaluate and select the best prompt
    times_runned = 0
    while times_runned < 10:

        json_response = api_call_openai_json("gpt-4o-mini-2024-07-18", instruction_to_select_best_prompt, temp, max_tokens)
        best_prompt, reason = extract_best_prompt(json_response)
        if best_prompt is None or reason is None:
            print(json_response)
            print("Response did not contain the required keys, retrying...")
            times_runned += 1
            continue

        if times_runned == 10:
            print("Failed to generate the best prompt.")
            return None, None

        # Store the best prompt with its details
        best_prompt_dict = {best_prompt: {"reason": reason, "candidates": candidates}}

        if os.path.exists("selected_prompt_dict_campaign.json"):
            with open("selected_prompt_dict_campaign.json", "r") as f:
                data = json.load(f)
            data.update(best_prompt_dict)
            with open("selected_prompt_dict_campaign.json", "w") as f:
                json.dump(data, f)
        else:
            with open("selected_prompt_dict_campaign.json", "w") as f:
                json.dump(best_prompt_dict, f)

        return best_prompt, reason



# Function to play the Negotiation game
def game(prompt_1, prompt_2, temp, max_tokens):
    
        times_runned = 0
        # Initialize the arguments for both the Pro and Con side debaters
        speech_1 = ""
        speech_2 = ""

        start_time = time.time()
        

        while True:
            if(times_runned>10):
                print("Failed to generate speech 1")
                speech_1 = ""
                break

            speech_1_text = api_call_openai("gpt-4o-mini-2024-07-18", prompt_to_speaker(prompt_1), temp, max_tokens).replace('\n', '').replace('\r', '').strip()

            if speech_1_text.startswith("Speech:") or speech_1_text.startswith("***Speech:***") or speech_1_text.startswith("**Speech:**"):
                speech_1 = speech_1_text.strip().split("Speech:")[-1].strip().strip('"').strip('*').replace('\n\n', ' ').replace('  ', ' ').strip()
                break

            else:
                print(speech_1_text)
                print("Speech 1 did not start with 'Speech:', retrying...")
                times_runned+=1
            
        times_runned = 0

        while True:
            if(times_runned>10):
                print("Failed to generate speech 2")
                speech_2 = ""
                break

            speech_2_text = api_call_openai("gpt-4o-mini-2024-07-18", prompt_to_speaker(prompt_2), temp, max_tokens).replace('\n', '').replace('\r', '').strip()

            if speech_2_text.startswith("Speech:") or speech_2_text.startswith("***Speech:***") or speech_2_text.startswith("**Speech:**"):
                speech_2 = speech_2_text.strip().split("Speech:")[-1].strip().strip('"').strip('*').replace('\n\n', ' ').replace('  ', ' ').strip()
                break

            else:
                print(speech_2_text)
                print("Speech 2 did not start with 'Speech:', retrying...")
                times_runned+=1

        # get the winner of speech 1 and 2
        times_runned = 0
        while True:
            if(times_runned>10):
                print("Failed to generate winner")
                winner = "draw"
                break

            json_response = api_call_openai_json("gpt-4o-mini-2024-07-18", prompt_to_judge_speeches(speech_1, speech_2), temp, max_tokens)
            winner, reason = extract_winner(json_response)
            if winner is not None and reason is not None:
                break
            else:
                print(json_response)
                print("Winner Response did not contain the required keys, retrying...")
                times_runned+=1
            

        end_time = time.time()
        print(f"Time taken to complete the game: {end_time - start_time} seconds")
        results = [prompt_1, prompt_2, speech_1, speech_2, winner, reason]
        return results

# Function to extract the winner and reason from the JSON response
def extract_winner(json_response):
    """
    Extracts the winner and reason from the JSON response if it meets all criteria.
    """

    if json_response == None:
        return None, None
    # Define the required keys
    required_keys = ["winner", "reason"]

    # Check if all required keys are present and their values are non-zero or non-empty and winner is either speech 1 or speech 2
    if all(key in json_response for key in required_keys) and all(json_response[key] for key in required_keys):
        # Check if winner is either speech 1 or speech 2
        if json_response["winner"].strip().strip('"').strip("'").strip().strip("**").strip("*").strip().strip('"').strip() == "Speech 1" or json_response["winner"].strip().strip('"').strip("'").strip().strip("**").strip("*").strip().strip('"').strip() == "Speech 2":
            winner = json_response["winner"].strip().strip('"').strip("'").strip().strip("**").strip("*").strip().strip('"').strip()
            reason = json_response["reason"].strip().strip('"').strip("'").strip().strip("**").strip("*").strip().strip('"').strip()
            return winner, reason
    else:
        # If any key is missing or has an empty/zero value, return None
        return None, None
# Function for prompting the Campaign Speaker
def prompt_to_speaker(speaker_prompt):
    """
    Parameters
    ----------
    speaker_prompt: str
        The prompt for the speaker giving a campaign speech.
    
    Returns
    --------
    prompt_speaker: str
        The final prompt for the speaker.
    """

    # Prompt the campaign speaker
    prompt_speaker = f"{speaker_prompt}"

    prompt_speaker += f"\n\nYou are expected to deliver a campaign speech as a candidate for presidency aiming to secure majority votes"

    prompt_speaker += f"\n\nMake sure that the response starts with 'Speech: ' followed by your speech."

    return prompt_speaker

def prompt_to_judge_speeches(speech_1, speech_2, criteria=None):
    """
    Parameters
    ----------
    speech_1: str
        The first campaign speech to be evaluated.
    speech_2: str
        The second campaign speech to be evaluated.
    criteria: str, optional
        The criteria based on which the judge has to decide the winner.
    
    Returns
    -------
    prompt_judge: str
        The instruction for the judge to evaluate the speeches and select the winner.
    """

    # Can look at both with and without criteria for eval....
    # Prompt with criteria
    if criteria is not None:
        prompt_judge = (
            f"You are tasked with evaluating two campaign speeches given by candidates for president to determine the winner. Below are the speeches:\n\n"
            f"Speech 1:\n{speech_1}\n\n"
            f"Speech 2:\n{speech_2}\n\n"
            f"Use the following criteria to make your decision: {criteria}.\n\n"
            "You must select only one winner, either Speech 1 or Speech 2, based on which speech better fulfills the given criteria.\n\n"
            "Here is the format you need to respond in (in JSON format):\n\n"
            "{\n"
            '    "reason": "[Your explanation of why you selected the winning speech]",\n'
            '    "winner": "[Speech 1/Speech 2]"\n'
            "}\n\n"
            "Ensure that all fields are provided and contain meaningful, non-empty values."
            "Note that for the 'winner' field, you should enter 'Speech 1' or 'Speech 2' not the actual speech content."
        )

    # Prompt without criteria
    else:
        prompt_judge = (
            f"You are tasked with evaluating two campaign speeches given by candidates for president to determine the winner. Below are the speeches:\n\n"
            f"Speech 1:\n{speech_1}\n\n"
            f"Speech 2:\n{speech_2}\n\n"
            "Based on the overall effectiveness, inspiration, and impact of the speeches, select the winner.\n\n"
            "You must select only one winner, either Speech 1 or Speech 2.\n\n"
            "Here is the format you need to respond in (in JSON format):\n\n"
            "{\n"
            '    "reason": "[Your explanation of why you selected the winning speech]",\n'
            '    "winner": "[Speech 1/Speech 2]"\n'
            "}\n\n"
            "Ensure that all fields are provided and contain meaningful, non-empty values."
            "Note that for the 'winner' field, you should enter 'Speech 1' or 'Speech 2' not the actual speech content."
        )

    return prompt_judge




# finding the ELO rating of the prompts after the battle
def elo_rating(prompt_1, prompt_2, temp, max_tokens, game_over = False, results = None, custom_dict = None):
    
    global elo_ratings_dict
    
    # Get the results of the game
    if game_over == False:
        results = game(prompt_1, prompt_2, temp, max_tokens)
        if results == None:
            print("Failed to generate results")
            winner = "draw"
            results = [prompt_1, prompt_2, "","" , winner, None]
            return results, winner

    if game_over == True:
        if results == None:
            print("Failed to generate results")
            # Don't update the ELO ratings if the results are not generated
            winner = "draw"
            results = [prompt_1, prompt_2, "", "", winner, None]
            return results, winner
                
    # Get the winner of the game
    winner = results[4]
    if (custom_dict == None):
        #If it is a draw, then calculate the ELO ratings of the prompts and print the updated ELO ratings
        if winner == "draw":
            # Find the ELO scores of the winner and loser
            winner_rating = elo_ratings_dict[prompt_1]
            loser_rating = elo_ratings_dict[prompt_2]
            # Calculate the expected score of the winner and loser
            expected_winner = 1 / (1 + 10 ** ((loser_rating - winner_rating) / 400))
            expected_loser = 1 / (1 + 10 ** ((winner_rating - loser_rating) / 400))
            # Update the ELO ratings of the winner and loser
            winner_rating = winner_rating + 32 * (0.5 - expected_winner)
            loser_rating = loser_rating + 32 * (0.5 - expected_loser)
            # Update the ELO ratings in the dictionary
            elo_ratings_dict[prompt_1] = winner_rating
            elo_ratings_dict[prompt_2] = loser_rating
            # Print the updated ELO ratings
            print("Updated ELO Ratings:")
            print(f"{prompt_1}: {winner_rating}")
            print(f"{prompt_2}: {loser_rating}")

        #If the winner is the Pro side debater, then calculate the ELO ratings of the prompts and print the updated ELO ratings

        elif winner == "Speech 1":
            # Find the ELO scores of the winner and loser
            winner_rating = elo_ratings_dict[prompt_1]
            loser_rating = elo_ratings_dict[prompt_2]
            # Calculate the expected score of the winner and loser
            expected_winner = 1 / (1 + 10 ** ((loser_rating - winner_rating) / 400))
            expected_loser = 1 / (1 + 10 ** ((winner_rating - loser_rating) / 400))
            # Update the ELO ratings of the winner and loser
            winner_rating = winner_rating + 32 * (1 - expected_winner)
            loser_rating = loser_rating + 32 * (0 - expected_loser)
            # Update the ELO ratings in the dictionary
            elo_ratings_dict[prompt_1] = winner_rating
            elo_ratings_dict[prompt_2] = loser_rating
            # Print the updated ELO ratings
            print("Updated ELO Ratings:")
            print(f"{prompt_1}: {winner_rating}")
            print(f"{prompt_2}: {loser_rating}")

        #If the winner is the Con side debater, then calculate the ELO ratings of the prompts and print the updated ELO ratings

        elif winner == "Speech 2":
            # Find the ELO scores of the winner and loser
            winner_rating = elo_ratings_dict[prompt_2]
            loser_rating = elo_ratings_dict[prompt_1]
            # Calculate the expected score of the winner and loser
            expected_winner = 1 / (1 + 10 ** ((loser_rating - winner_rating) / 400))
            expected_loser = 1 / (1 + 10 ** ((winner_rating - loser_rating) / 400))
            # Update the ELO ratings of the winner and loser
            winner_rating = winner_rating + 32 * (1 - expected_winner)
            loser_rating = loser_rating + 32 * (0 - expected_loser)
            # Update the ELO ratings in the dictionary
            elo_ratings_dict[prompt_2] = winner_rating
            elo_ratings_dict[prompt_1] = loser_rating
            # Print the updated ELO ratings
            print("Updated ELO Ratings:")
            print(f"{prompt_2}: {winner_rating}")
            print(f"{prompt_1}: {loser_rating}")
    
    elif(custom_dict!=None):
        #If it is a draw, then calculate the ELO ratings of the prompts and print the updated ELO ratings
        if winner == "draw":
            # Find the ELO scores of the winner and loser
            winner_rating = custom_dict[prompt_1]
            loser_rating = custom_dict[prompt_2]
            # Calculate the expected score of the winner and loser
            expected_winner = 1 / (1 + 10 ** ((loser_rating - winner_rating) / 400))
            expected_loser = 1 / (1 + 10 ** ((winner_rating - loser_rating) / 400))
            # Update the ELO ratings of the winner and loser
            winner_rating = winner_rating + 32 * (0.5 - expected_winner)
            loser_rating = loser_rating + 32 * (0.5 - expected_loser)
            # Update the ELO ratings in the dictionary
            custom_dict[prompt_1] = winner_rating
            custom_dict[prompt_2] = loser_rating
            # Print the updated ELO ratings
            print("Updated ELO Ratings:")
            print(f"{prompt_1}: {winner_rating}")
            print(f"{prompt_2}: {loser_rating}")

        #If the winner is the Pro side debater, then calculate the ELO ratings of the prompts and print the updated ELO ratings

        elif winner == "Speech 1":
            # Find the ELO scores of the winner and loser
            winner_rating = custom_dict[prompt_1]
            loser_rating = custom_dict[prompt_2]
            # Calculate the expected score of the winner and loser
            expected_winner = 1 / (1 + 10 ** ((loser_rating - winner_rating) / 400))
            expected_loser = 1 / (1 + 10 ** ((winner_rating - loser_rating) / 400))
            # Update the ELO ratings of the winner and loser
            winner_rating = winner_rating + 32 * (1 - expected_winner)
            loser_rating = loser_rating + 32 * (0 - expected_loser)
            # Update the ELO ratings in the dictionary
            custom_dict[prompt_1] = winner_rating
            custom_dict[prompt_2] = loser_rating
            # # Print the updated ELO ratings
            print("Updated ELO Ratings:")
            print(f"{prompt_1}: {winner_rating}")
            print(f"{prompt_2}: {loser_rating}")

        #If the winner is the Con side debater, then calculate the ELO ratings of the prompts and print the updated ELO ratings

        elif winner == "Speech 2":
            # Find the ELO scores of the winner and loser
            winner_rating = custom_dict[prompt_2]
            loser_rating = custom_dict[prompt_1]
            # Calculate the expected score of the winner and loser
            expected_winner = 1 / (1 + 10 ** ((loser_rating - winner_rating) / 400))
            expected_loser = 1 / (1 + 10 ** ((winner_rating - loser_rating) / 400))
            # Update the ELO ratings of the winner and loser
            winner_rating = winner_rating + 32 * (1 - expected_winner)
            loser_rating = loser_rating + 32 * (0 - expected_loser)
            # Update the ELO ratings in the dictionary
            custom_dict[prompt_2] = winner_rating
            custom_dict[prompt_1] = loser_rating
            # Print the updated ELO ratings
            print("Updated ELO Ratings:")
            print(f"{prompt_2}: {winner_rating}")
            print(f"{prompt_1}: {loser_rating}")
    
    return results, winner

# Function to check if the i'th league is already over or not to avoid playing the league again by checking if the league number is present in the dictionary of leage_results.json file
def check_league(league_no):
    
        """
        Parameters
        ----------
        league_no: int
            The league number to be checked if it is already over or not
    
        Returns
        --------
        Bool: True if the league is already over, False otherwise
    
        """
    
        # Load the dictionary from the JSON file

        if os.path.exists('league_results.json') == False:
            return False
        
        with open('league_results.json', 'r') as json_file:
            data = json.load(json_file)
        
        # Check if the league number is present in the dictionary
        if str(league_no) in data:
            return True
        else:
            return False



# Function which takes all the prompts and objects as input and plays a round robin league and updates the elo ratings of the prompts and returns the top k and bottom k prompts after the league
def league(prompts, k, league_no, temp, max_tokens, human = False, human_file = None):

    
    global elo_ratings_dict
    global league_dict
    global top_bottom_prompts_dict
    global human_prompts_used_dict
    global origin_league_prompts_dict

    for prompt in prompts:
        origin_league_prompts_dict[prompt] = league_no


    #Check if the league is already palyed or not
    if check_league(league_no):

        print(f"League {league_no} is already over!")
        # Get the top and bottom k prompts along with their final elo scores from the league_results.json file for that league_no and also put it in the top_bottom_prompts_dict
        with open('league_results.json', 'r') as json_file:
            data = json.load(json_file)
        
        # Collect all prompts with their final ELO scores
        all_prompts = {}
        for battle_no, battle_dict in data[str(league_no)].items():
            prompt_1 = battle_dict['prompt_1']['prompt']
            prompt_2 = battle_dict['prompt_2']['prompt']
            prompt_1_final_elo = battle_dict['prompt_1']['final_elo']
            prompt_2_final_elo = battle_dict['prompt_2']['final_elo']
            all_prompts[prompt_1] = prompt_1_final_elo
            all_prompts[prompt_2] = prompt_2_final_elo

        # Store all the prompts in the human_prompts_used_dict
        if human == True and human_file != None:
            human_prompts_used_dict[league_no] = list(all_prompts.keys())

        # Store all the elo scores in the elo_ratings_dict with the key has prompt and value as elo score
        for prompt, elo in all_prompts.items():
            elo_ratings_dict[prompt] = elo

        # Sort the prompts based on their ELO scores
        sorted_prompts= sorted(all_prompts.items(), key=lambda x: x[1], reverse=True)

        # Get the top k and bottom k prompts
        top_k_prompts = sorted_prompts[:k]
        bottom_k_prompts = sorted_prompts[-k:]

        # Store the top and bottom k prompts in the dictionary for the league and also rank them by 1,2,3..,k
        top_bottom_prompts_dict[league_no] = {
            'top_k_prompts': {i+1: {'prompt': prompt, 'elo': elo} for i, (prompt, elo) in enumerate(top_k_prompts)},
            'bottom_k_prompts': {i+1: {'prompt': prompt, 'elo': elo} for i, (prompt, elo) in enumerate(bottom_k_prompts)}
        }

        # Dump the top and bottom k prompts for the Pro side and Con side debators in a json file 
        with open('top_bottom_prompts_dict.json', 'w') as json_file:
            json.dump(top_bottom_prompts_dict, json_file, indent=4)

        return top_k_prompts, bottom_k_prompts
    
    temp_league_dict = {}
    temp_league_dict[league_no] = {}

    for prompt in prompts:
        elo_ratings_dict[prompt] = 1200

    battle_no = 0

    round_robin_list = [(i, j) for i in range(len(prompts)) for j in range(len(prompts)) if i < j]
    random.shuffle(round_robin_list)

    # To store the results for ELO updates after parallel processing in the correct order
    results_list = [None] * len(round_robin_list)  # Preallocate list to ensure correct order
    
    start_time = time.time()

    # Use ThreadPoolExecutor to run games in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=7) as executor:
        future_to_battle = {
            executor.submit(game, prompts[i], prompts[j], temp, max_tokens): (index, i, j)
            for index, (i, j) in enumerate(round_robin_list)
        }

        for future in concurrent.futures.as_completed(future_to_battle):
            index, i, j = future_to_battle[future]
            try:
                result = future.result()
                results_list[index] = (i, j, result)  # Store results in the correct order
            except Exception as exc:
                print(f'Game {i} vs {j} generated an exception: {exc}')

    # Process results in the order of the original round robin list
    for (i, j, result) in results_list:

        # Get initial ELO ratings
        initial_elo_rating_1 = elo_ratings_dict[prompts[i]]
        initial_elo_rating_2 = elo_ratings_dict[prompts[j]]

        # Get speech1 and speech2
        speech_1 = result[2]
        speech_2 = result[3]

        # Check if speech 1 is not none then get the analysis of speech 1 using analysis function
        if speech_1 != "":
            analysis_1 = analysis(speech_1)
        else:
            analysis_1 = [None]*22

        # Check if speech 2 is not none then get the analysis of speech 2 using analysis function
        if speech_2 != "":
            analysis_2 = analysis(speech_2)
        else:
            analysis_2 = [None]*22
        
        # Update ELO ratings using elo_rating function
        result, winner = elo_rating(prompts[i], prompts[j], temp, max_tokens, game_over=True, results=result, custom_dict=None)
        
        # Get final ELO ratings
        final_elo_rating1 = elo_ratings_dict[prompts[i]]
        final_elo_rating2 = elo_ratings_dict[prompts[j]]

        # Increment battle number and prepare battle dictionary
        battle_no += 1
        battle_dict = {
        'prompt_1': {
            'prompt': prompts[i],
            'initial_elo': initial_elo_rating_1,
            'final_elo': final_elo_rating1
        },
        'prompt_2': {
            'prompt': prompts[j],
            'initial_elo': initial_elo_rating_2,
            'final_elo': final_elo_rating2
        },
        'speech_1': {
            'speech_1_text': result[2],
            'sentiment': analysis_1[0],
            'reason_for_sentiment': analysis_1[1],
            'length': analysis_1[2],
            'num_tokens': analysis_1[3],
            'personal_count': analysis_1[4],
            'audience_count': analysis_1[5],
            'personal_ratio': analysis_1[6],
            'audience_ratio': analysis_1[7],
            'male_features_count': analysis_1[8],
            'female_features_count': analysis_1[9],
            'male_features_ratio': analysis_1[10],
            'female_features_ratio': analysis_1[11],
            'topic': analysis_1[12],
            'emotional_tone': analysis_1[13],
            'emotion_scores': analysis_1[14],
            'ttr': analysis_1[15],
            'lexical_density': analysis_1[16],
            'entropy': analysis_1[17],
            'formality': analysis_1[18],
            'reason': analysis_1[19],
            'flesch_score': analysis_1[20],
            'flesch_grade': analysis_1[21],
            'fog_index': analysis_1[22],
            'smog_index': analysis_1[23],
            'ari_score': analysis_1[24]
        },
        'speech_2': {
            'speech_2_text': result[3],
            'sentiment': analysis_2[0],
            'reason_for_sentiment': analysis_2[1],
            'length': analysis_2[2],
            'num_tokens': analysis_2[3],
            'personal_count': analysis_2[4],
            'audience_count': analysis_2[5],
            'personal_ratio': analysis_2[6],
            'audience_ratio': analysis_2[7],
            'male_features_count': analysis_2[8],
            'female_features_count': analysis_2[9],
            'male_features_ratio': analysis_2[10],
            'female_features_ratio': analysis_2[11],
            'topic': analysis_2[12],
            'emotional_tone': analysis_2[13],
            'emotion_scores': analysis_2[14],
            'ttr': analysis_2[15],
            'lexical_density': analysis_2[16],
            'entropy': analysis_2[17],
            'formality': analysis_2[18],
            'reason': analysis_2[19],
            'flesch_score': analysis_2[20],
            'flesch_grade': analysis_2[21],
            'fog_index': analysis_2[22],
            'smog_index': analysis_2[23],
            'ari_score': analysis_2[24]
        },
        'winner_prompt': prompts[i] if winner == 'Speech 1' else 'draw' if winner == 'draw' else prompts[j],
        'reason': result[5]
        }


        temp_league_dict[league_no][battle_no] = battle_dict

    end_time = time.time()
    print(f"Time taken to complete the league: {end_time - start_time} seconds")
    
    # Store the prompts used in the human_prompts_used_dict if human prompts are used
    if human == True and human_file != None:
        human_prompts_used_dict[league_no] = prompts
        

    temp_elo_dict = {}
    for i in range(len(prompts)):
        temp_elo_dict[prompts[i]]= elo_ratings_dict[prompts[i]]
    
    sorted_prompts = sorted(temp_elo_dict.items(), key=lambda x: x[1], reverse=True)

    top_k_prompts = sorted_prompts[:k]
    bottom_k_prompts = sorted_prompts[-k:]

    # Check if the JSON file exists
    file_path = 'league_results.json'
    if os.path.exists(file_path):
        # If the file exists, load the existing data
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
    else:
        # If the file does not exist, start with an empty dictionary
        data = {}

    # Update the data with the current league_dict
    data.update(temp_league_dict)

    # Write the updated data back to the JSON file
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    
    # Save the top and bottom k prompts in the dictionary for the league and also rank them by 1,2,3..,k
    top_bottom_prompts_dict[league_no] = {
        'top_k_prompts': {i+1: {'prompt': prompt, 'elo': elo} for i, (prompt, elo) in enumerate(top_k_prompts)},
        'bottom_k_prompts': {i+1: {'prompt': prompt, 'elo': elo} for i, (prompt, elo) in enumerate(bottom_k_prompts)}
    }

    # Dump the top and bottom k prompts in a json file 
    with open('top_bottom_prompts_dict.json', 'w') as json_file:
        json.dump(top_bottom_prompts_dict, json_file, indent=4)
    

    # # Load the dictionary from the pickle file
    # with open('league_results.pkl', 'rb') as pickle_file:
    #     league_dict = pickle.load(pickle_file)

    # return top and bottom k prompts
    return top_k_prompts, bottom_k_prompts

# Play the final league between the league_no top k prompts and league_no -1 top k prompts to get the final top k prompts for each role
def final_league(prompts, league_no, k, temp, max_tokens):

    global top_bottom_prompts_dict_across_league
    global origin_league_prompts_dict

    final_elo_dict = {}

    #Check if the final_league_results{current_league}.json file exists
    if os.path.exists(f'final_league_results{league_no}.json'):
        print(f"Final League {league_no} is already over!")
        # Get the top and bottom k prompts along with their final elo scores from the final_league_results{league_no}.json file for that league_no
        with open(f'final_league_results{league_no}.json', 'r') as json_file:
            data = json.load(json_file)
        # Collect all prompts with their final ELO scores
        all_prompts = {}
        for battle_no, battle_dict in data[str(league_no)].items():
            prompt_1 = battle_dict['prompt_1']['prompt']
            prompt_2 = battle_dict['prompt_2']['prompt']
            prompt_1_final_elo = battle_dict['prompt_1']['final_elo']
            prompt_2_final_elo = battle_dict['prompt_2']['final_elo']
            all_prompts[prompt_1] = prompt_1_final_elo
            all_prompts[prompt_2] = prompt_2_final_elo
        
        for prompt, elo in all_prompts.items():
            final_elo_dict[prompt] = elo

        # Sort the prompts based on their ELO scores
        sorted_prompts = sorted(all_prompts.items(), key=lambda x: x[1], reverse=True)

        # Get the top k and bottom k prompts
        top_k_prompts = sorted_prompts[:k]
        bottom_k_prompts = sorted_prompts[-k:]

        # Store the top_k_prompts pro, con and bottom_k_prompts pro, con in the top_bottom_prompts_dict_con_across_league and top_bottom_prompts_dict_pro_across_league
        top_bottom_prompts_dict_across_league[league_no] = {
            'top_k_prompts': {i+1: {'prompt': prompt, 'elo': elo, 'origin': origin_league_prompts_dict[prompt]} for i, (prompt, elo) in enumerate(top_k_prompts)},
            'bottom_k_prompts': {i+1: {'prompt': prompt, 'elo': elo, 'origin': origin_league_prompts_dict[prompt]} for i, (prompt, elo) in enumerate(bottom_k_prompts)}
        }

        # Dump the top and bottom k prompts in a json file
        with open('top_bottom_prompts_dict_across_league.json', 'w') as json_file:
            json.dump(top_bottom_prompts_dict_across_league, json_file, indent=4)
    
        return top_k_prompts, bottom_k_prompts, final_elo_dict
    
    # Initialise the elo ratings of each prompt to 1200
    for prompt in prompts:
        final_elo_dict[prompt] = 1200
    
    temp_league_dict = {}
    temp_league_dict[league_no] = {}

    battle_no = 0

    round_robin_list = [(i, j) for i in range(len(prompts)) for j in range(len(prompts)) if i < j]
    random.shuffle(round_robin_list)

    # To store the results for ELO updates after parallel processing in the correct order
    results_list = [None] * len(round_robin_list)  # Preallocate list to ensure correct order
    
    start_time = time.time()

    # Use ThreadPoolExecutor to run games in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=7) as executor:
        future_to_battle = {
            executor.submit(game, prompts[i], prompts[j], temp, max_tokens): (index, i, j)
            for index, (i, j) in enumerate(round_robin_list)
        }

        for future in concurrent.futures.as_completed(future_to_battle):
            index, i, j = future_to_battle[future]
            try:
                result = future.result()
                results_list[index] = (i, j, result)  # Store results in the correct order
            except Exception as exc:
                print(f'Game {i} vs {j} generated an exception: {exc}')

    # Process results in the order of the original round robin list
    for (i, j, result) in results_list:

        # Get initial ELO ratings
        initial_elo_rating_1 = final_elo_dict[prompts[i]]
        initial_elo_rating_2 = final_elo_dict[prompts[j]]
        
        # Update ELO ratings using elo_rating function
        result, winner = elo_rating(prompts[i], prompts[j], temp, max_tokens, game_over=True, results=result, custom_dict= final_elo_dict)
        
        # Get final ELO ratings
        final_elo_rating1 = final_elo_dict[prompts[i]]
        final_elo_rating2 = final_elo_dict[prompts[j]]

        # Get speech1 and speech2
        speech_1 = result[2]
        speech_2 = result[3]

        # Check if speech 1 is not none then get the analysis of speech 1 using analysis function
        if speech_1 != "":
            analysis_1 = analysis(speech_1)
        else:
            analysis_1 = [None]*22

        # Check if speech 2 is not none then get the analysis of speech 2 using analysis function
        if speech_2 != "":
            analysis_2 = analysis(speech_2)
        else:
            analysis_2 = [None]*22

        # Increment battle number and prepare battle dictionary
        battle_no += 1

        battle_dict = {
        'prompt_1': {
            'prompt': prompts[i],
            'initial_elo': initial_elo_rating_1,
            'final_elo': final_elo_rating1
        },
        'prompt_2': {
            'prompt': prompts[j],
            'initial_elo': initial_elo_rating_2,
            'final_elo': final_elo_rating2
        },
        'speech_1': {
            'speech_1_text': result[2],
            'sentiment': analysis_1[0],
            'reason_for_sentiment': analysis_1[1],
            'length': analysis_1[2],
            'num_tokens': analysis_1[3],
            'personal_count': analysis_1[4],
            'audience_count': analysis_1[5],
            'personal_ratio': analysis_1[6],
            'audience_ratio': analysis_1[7],
            'male_features_count': analysis_1[8],
            'female_features_count': analysis_1[9],
            'male_features_ratio': analysis_1[10],
            'female_features_ratio': analysis_1[11],
            'topic': analysis_1[12],
            'emotional_tone': analysis_1[13],
            'emotion_scores': analysis_1[14],
            'ttr': analysis_1[15],
            'lexical_density': analysis_1[16],
            'entropy': analysis_1[17],
            'formality': analysis_1[18],
            'reason': analysis_1[19],
            'flesch_score': analysis_1[20],
            'flesch_grade': analysis_1[21],
            'fog_index': analysis_1[22],
            'smog_index': analysis_1[23],
            'ari_score': analysis_1[24]
        },
        'speech_2': {
            'speech_2_text': result[3],
            'sentiment': analysis_2[0],
            'reason_for_sentiment': analysis_2[1],
            'length': analysis_2[2],
            'num_tokens': analysis_2[3],
            'personal_count': analysis_2[4],
            'audience_count': analysis_2[5],
            'personal_ratio': analysis_2[6],
            'audience_ratio': analysis_2[7],
            'male_features_count': analysis_2[8],
            'female_features_count': analysis_2[9],
            'male_features_ratio': analysis_2[10],
            'female_features_ratio': analysis_2[11],
            'topic': analysis_2[12],
            'emotional_tone': analysis_2[13],
            'emotion_scores': analysis_2[14],
            'ttr': analysis_2[15],
            'lexical_density': analysis_2[16],
            'entropy': analysis_2[17],
            'formality': analysis_2[18],
            'reason': analysis_2[19],
            'flesch_score': analysis_2[20],
            'flesch_grade': analysis_2[21],
            'fog_index': analysis_2[22],
            'smog_index': analysis_2[23],
            'ari_score': analysis_2[24]
        },
        'winner_prompt': prompts[i] if winner == 'Speech 1' else 'draw' if winner == 'draw' else prompts[j],
        'reason': result[5]
        }

        temp_league_dict[league_no][battle_no] = battle_dict

    end_time = time.time()
    print(f"Time taken to complete the league: {end_time - start_time} seconds")
    
    temp_elo_dict = {}

    for i in range(len(prompts)):
        temp_elo_dict[prompts[i]]= final_elo_dict[prompts[i]]
    

    sorted_prompts = sorted(temp_elo_dict.items(), key=lambda x: x[1], reverse=True)

    # Get the top k and bottom k prompts
    top_k_prompts = sorted_prompts[:k]
    bottom_k_prompts = sorted_prompts[-k:]

    # Check if the JSON file exists
    file_path = f'final_league_results{league_no}.json'
    
    data = {}

    # Update the data with the current league_dict
    data.update(temp_league_dict)

    # Write the updated data back to the JSON file
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)


    # Store the top_k_prompts pro, con and bottom_k_prompts pro, con in the top_bottom_prompts_dict_con_across_league and top_bottom_prompts_dict_pro_across_league
    top_bottom_prompts_dict_across_league[league_no] = {
            'top_k_prompts': {i+1: {'prompt': prompt, 'elo': elo, 'origin': origin_league_prompts_dict[prompt]} for i, (prompt, elo) in enumerate(top_k_prompts)},
            'bottom_k_prompts': {i+1: {'prompt': prompt, 'elo': elo, 'origin': origin_league_prompts_dict[prompt]} for i, (prompt, elo) in enumerate(bottom_k_prompts)}
    }

    # Dump the top and bottom k prompts
    with open('top_bottom_prompts_dict_across_league.json', 'w') as json_file:
        json.dump(top_bottom_prompts_dict_across_league, json_file, indent=4)
    
    return top_k_prompts, bottom_k_prompts, final_elo_dict


def compare_elo_average(top_prompts_league, final_elo_dict, league_no, k, consecutive_fails):

    elo_average_league = 0
    
    for prompt in top_prompts_league:
        elo_average_league += final_elo_dict[prompt]
    elo_average_league = elo_average_league/len(top_prompts_league)


    # Get the top k pro prompts and top k con prompts using top_bottom_prompts_dict_across_league.json of league number = league_no -1
    with open('top_bottom_prompts_dict_across_league.json', 'r') as json_file:
        top_bottom_prompts_dict_across_final = json.load(json_file)
    
    top_prompts_final = top_bottom_prompts_dict_across_final[str(league_no-1)]['top_k_prompts']
    # Extract the prompt from the dictionary and create a list of all the prompts
    top_prompts_final = [top_prompts_final[str(i)]['prompt'] for i in range(1, k+1)]
    
    elo_average_final = 0

    for prompt in top_prompts_final:
        elo_average_final += final_elo_dict[prompt]
    elo_average_final = elo_average_final/len(top_prompts_final)

    if (elo_average_final > elo_average_league or elo_average_final > elo_average_league):
        consecutive_fails+=1
        print("consecutive_fails: ", consecutive_fails)
        
    else:
        consecutive_fails = 0
        print("consecutive_fails: ", consecutive_fails)
    
    return consecutive_fails

def proof_of_concept(league1, league2, k , temp, max_tokens):

    global top_bottom_prompts_dict_across_league
    global top_bottom_prompts_dict
    global top_bottom_prompts_dict_poc
    global origin_league_prompts_dict

    league_no = 1
    
    poc_elo_dict = {}

    if league1 !=1:
        # Get prompts1 by getting top k prompts from the league1 using top_bottom_prompts_dict_across_league

        top_prompts_league1 = top_bottom_prompts_dict_across_league[league1]['top_k_prompts']
        prompts1 = [top_prompts_league1[i]['prompt'] for i in range(1, k+1)]

        # Get pro_prompts2 by getting top k prompts from the league2 using top_bottom_prompts_dict_con_across_league
        top_prompts_league2 = top_bottom_prompts_dict_across_league[league2]['top_k_prompts']
        prompts2 = [top_prompts_league2[i]['prompt'] for i in range(1, k+1)]
    
    if league1==1:

        # get the pro_prompts1 and con_prompts1 from the top_bottom_prompts_dict_pro and top_bottom_prompts_dict_con for league1
        prompts1 = [top_bottom_prompts_dict[league1]['top_k_prompts'][i]['prompt'] for i in range(1, k+1)]

        # get the pro_prompts2 and con_prompts2
        prompts2 = [top_bottom_prompts_dict_across_league[league2]['top_k_prompts'][i]['prompt'] for i in range(1, k+1)]

    # Merge the pro_prompts1 and pro_prompts2 to get the pro_prompts
    prompts = prompts1 + prompts2

    #Check if the file exists
    if os.path.exists(f'proof_of_concept_{league1}_{league2}.json'):
        print(f"Proof of Concept {league1} vs {league2} is already over!")
        # Get the top and bottom k prompts along with their final elo scores from the proof_of_concept_{league1}_{league2}.json' file for that league_no
        with open(f'proof_of_concept_{league1}_{league2}.json', 'r') as json_file:
            data = json.load(json_file)
        # Collect all prompts with their final ELO scores
        all_prompts = {}
        for battle_no, battle_dict in data[str(league_no)].items():
            prompt_1 = battle_dict['prompt_1']['prompt']
            prompt_2 = battle_dict['prompt_2']['prompt']
            prompt_1_final_elo = battle_dict['prompt_1']['final_elo']
            prompt_2_final_elo = battle_dict['prompt_2']['final_elo']
            all_prompts[prompt_1] = prompt_1_final_elo
            all_prompts[prompt_2] = prompt_2_final_elo
        
        for prompt, elo in all_prompts.items():
            poc_elo_dict[prompt] = elo

        # Sort the prompts based on their ELO scores
        sorted_prompts = sorted(all_prompts.items(), key=lambda x: x[1], reverse=True)

        # Get the top k and bottom k prompts for the Pro side debater
        top_k_prompts = sorted_prompts[:k]
        bottom_k_prompts = sorted_prompts[-k:]

        # Store the top_k_prompts pro, con and bottom_k_prompts pro, con in the top_bottom_prompts_dict_con_poc and top_bottom_prompts_dict_pro_poc (Include origin as well)
        top_bottom_prompts_dict_poc[f"{league1} vs {league2}"] = {
            'top_k_prompts': {i+1: {'prompt': prompt, 'elo': elo, 'origin': origin_league_prompts_dict[prompt]} for i, (prompt, elo) in enumerate(top_k_prompts)},
            'bottom_k_prompts': {i+1: {'prompt': prompt, 'elo': elo, 'origin': origin_league_prompts_dict[prompt]} for i, (prompt, elo) in enumerate(bottom_k_prompts)}
        }

        # Dump the top and bottom k prompts in a json file
        with open('top_bottom_prompts_dict_poc.json', 'w') as json_file:
            json.dump(top_bottom_prompts_dict_poc, json_file, indent=4)
        
        compare_poc_elo_average(poc_elo_dict, prompts1, prompts2, league1, league2)

        return

    
    # Initialise the elo ratings of each prompt to 1200
    for prompt in prompts:
        poc_elo_dict[prompt] = 1200
    
    temp_league_dict = {}
    temp_league_dict[league_no] = {}

    battle_no = 0

    round_robin_list = [(i, j) for i in range(len(prompts)) for j in range(len(prompts)) if i < j]
    random.shuffle(round_robin_list)

    # To store the results for ELO updates after parallel processing in the correct order
    results_list = [None] * len(round_robin_list)  # Preallocate list to ensure correct order
    
    start_time = time.time()

    # Use ThreadPoolExecutor to run games in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=7) as executor:
        future_to_battle = {
            executor.submit(game, prompts[i], prompts[j], temp, max_tokens): (index, i, j)
            for index, (i, j) in enumerate(round_robin_list)
        }

        for future in concurrent.futures.as_completed(future_to_battle):
            index, i, j = future_to_battle[future]
            try:
                result = future.result()
                results_list[index] = (i, j, result)  # Store results in the correct order
            except Exception as exc:
                print(f'Game {i} vs {j} generated an exception: {exc}')

    # Process results in the order of the original round robin list
    for (i, j, result) in results_list:

        # Get initial ELO ratings
        initial_elo_rating_1 = poc_elo_dict[prompts[i]]
        initial_elo_rating_2 = poc_elo_dict[prompts[j]]
        
        # Update ELO ratings using elo_rating function
        result, winner = elo_rating(prompts[i], prompts[j], temp, max_tokens, game_over=True, results=result, custom_dict= poc_elo_dict)
        
        # Get final ELO ratings
        final_elo_rating1 = poc_elo_dict[prompts[i]]
        final_elo_rating2 = poc_elo_dict[prompts[j]]

        # Get speech1 and speech2
        speech_1 = result[2]
        speech_2 = result[3]

        # Check if speech 1 is not none then get the analysis of speech 1 using analysis function
        if speech_1 != "":
            analysis_1 = analysis(speech_1)
        else:
            analysis_1 = [None]*22
        
        # Check if speech 2 is not none then get the analysis of speech 2 using analysis function
        if speech_2 != "":
            analysis_2 = analysis(speech_2)
        else:
            analysis_2 = [None]*22

        # Increment battle number and prepare battle dictionary
        battle_no += 1
        battle_dict = {
        'prompt_1': {
            'prompt': prompts[i],
            'initial_elo': initial_elo_rating_1,
            'final_elo': final_elo_rating1
        },
        'prompt_2': {
            'prompt': prompts[j],
            'initial_elo': initial_elo_rating_2,
            'final_elo': final_elo_rating2
        },
        'speech_1': {
            'speech_1_text': result[2],
            'sentiment': analysis_1[0],
            'reason_for_sentiment': analysis_1[1],
            'length': analysis_1[2],
            'num_tokens': analysis_1[3],
            'personal_count': analysis_1[4],
            'audience_count': analysis_1[5],
            'personal_ratio': analysis_1[6],
            'audience_ratio': analysis_1[7],
            'male_features_count': analysis_1[8],
            'female_features_count': analysis_1[9],
            'male_features_ratio': analysis_1[10],
            'female_features_ratio': analysis_1[11],
            'topic': analysis_1[12],
            'emotional_tone': analysis_1[13],
            'emotion_scores': analysis_1[14],
            'ttr': analysis_1[15],
            'lexical_density': analysis_1[16],
            'entropy': analysis_1[17],
            'formality': analysis_1[18],
            'reason': analysis_1[19],
            'flesch_score': analysis_1[20],
            'flesch_grade': analysis_1[21],
            'fog_index': analysis_1[22],
            'smog_index': analysis_1[23],
            'ari_score': analysis_1[24]
        },
        'speech_2': {
            'speech_2_text': result[3],
            'sentiment': analysis_2[0],
            'reason_for_sentiment': analysis_2[1],
            'length': analysis_2[2],
            'num_tokens': analysis_2[3],
            'personal_count': analysis_2[4],
            'audience_count': analysis_2[5],
            'personal_ratio': analysis_2[6],
            'audience_ratio': analysis_2[7],
            'male_features_count': analysis_2[8],
            'female_features_count': analysis_2[9],
            'male_features_ratio': analysis_2[10],
            'female_features_ratio': analysis_2[11],
            'topic': analysis_2[12],
            'emotional_tone': analysis_2[13],
            'emotion_scores': analysis_2[14],
            'ttr': analysis_2[15],
            'lexical_density': analysis_2[16],
            'entropy': analysis_2[17],
            'formality': analysis_2[18],
            'reason': analysis_2[19],
            'flesch_score': analysis_2[20],
            'flesch_grade': analysis_2[21],
            'fog_index': analysis_2[22],
            'smog_index': analysis_2[23],
            'ari_score': analysis_2[24]
        },
        'winner_prompt': prompts[i] if winner == 'Speech 1' else 'draw' if winner == 'draw' else prompts[j],
        'reason': result[5]
        }

        temp_league_dict[league_no][battle_no] = battle_dict

    end_time = time.time()
    print(f"Time taken to complete the league: {end_time - start_time} seconds")
    
    temp_elo_dict = {}

    for i in range(len(prompts)):
        temp_elo_dict[prompts[i]] = poc_elo_dict[prompts[i]]
    
    sorted_prompts = sorted(temp_elo_dict.items(), key=lambda x: x[1], reverse=True)

    # Get the top k and bottom k prompts
    top_k_prompts = sorted_prompts[:k]
    bottom_k_prompts = sorted_prompts[-k:]

    # Check if the JSON file exists
    file_path = f'proof_of_concept_{league1}_{league2}.json'
    
    data = {}

    # Update the data with the current league_dict
    data.update(temp_league_dict)

    # Write the updated data back to the JSON file
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    compare_poc_elo_average(poc_elo_dict, prompts1, prompts2, league1, league2)

    top_bottom_prompts_dict_poc[f"{league1} vs {league2}"] = {
        'top_k_prompts': {i+1: {'prompt': prompt, 'elo': elo, 'origin': origin_league_prompts_dict[prompt]} for i, (prompt, elo) in enumerate(top_k_prompts)},
        'bottom_k_prompts': {i+1: {'prompt': prompt, 'elo': elo, 'origin': origin_league_prompts_dict[prompt]} for i, (prompt, elo) in enumerate(bottom_k_prompts)}
    }

    # Dump the top and bottom k prompts in a json file
    with open('top_bottom_prompts_dict_poc.json', 'w') as json_file:
        json.dump(top_bottom_prompts_dict_poc, json_file, indent=4)

    return


def compare_poc_elo_average(poc_elo_dict, prompts1, prompts2, league1, league2):
    # Get the average of pro_prompts1, con_prompts1, pro_prompts2, con_prompts2 using the poc_elo_dict
    elo_average1 = 0
    elo_average2 = 0
    for prompt in prompts1:
        elo_average1 += poc_elo_dict[prompt]
    elo_average1 = elo_average1/len(prompts1)
    print("elo_average1: ", elo_average1)
    

    elo_average2 = 0
    for prompt in prompts2:
        elo_average2 += poc_elo_dict[prompt]
    elo_average2 = elo_average2/len(prompts2)
    print("elo_average2: ", elo_average2)

    # Compare the average of pro_prompts1, con_prompts1, pro_prompts2, con_prompts2
    if (elo_average1 > elo_average2):
        print(f"Proof of Concept {league1} vs {league2} Failed!")
    
    else:
        print(f"Proof of Concept {league1} vs {league2} Passed!")


# Function to play the tournament. In this, first we play the initial league on the prompts generated by the get_prompts() function and then we get the top and bottom k prompts. Then we generate new prompts using the get_new_prompts() function and play the league on these new prompts. We repeat this process for n-1 number of leagues. We give the top and bottom k prompts of current league as input to the get_new_prompts() function to generate new prompts for the next league. Then we play the n'th league which is the final league which takes the top k prompts and bottom k from each of the n-1 leagues and conducts a final league on these prompts. The top k prompts from this final league are the top k prompts of the tournament.
def tournament(no_of_prompts_start, no_of_prompts_between, k, n, temp, max_tokens, current_directory):


        global top_bottom_prompts_dict
        global elo_ratings_dict
        global current_league

        file_path = os.path.join(current_directory, 'humans.csv')

        consecutive_fails = 0
        # Play all the leagues
        while(consecutive_fails<3):
            print(f"League {current_league}:\n")
            # Play the first league
            if (current_league ==1):

                prompts = get_prompts(no_of_prompts_start, current_league, temp, max_tokens, None, None, False, None)

                if prompts is None:
                    print("Invalid Input")
                    return
                
                top_prompts_league_1, bottom_prompts_league_1 = league(prompts, k, current_league, temp, max_tokens, False, None)
                # Get only the prompts from the top_pro_prompts_league1, bottom_pro_prompts_league1, top_con_prompts_league1, bottom_con_prompts_league1
                top_prompts_league_1 = [prompt for prompt, elo in top_prompts_league_1]
                bottom_prompts_league_1 = [prompt for prompt, elo in bottom_prompts_league_1]
                
                current_league+=1

            elif (current_league !=1):

                if(current_league==6):
                    break

                if (current_league ==2):
                # Generate new prompts for the next league using the top k and bottom k prompts from the previous league
                    prompts = get_prompts(no_of_prompts_between, current_league, temp, max_tokens, top_prompts_league_1, bottom_prompts_league_1, False, None)
                    if prompts is None:
                        print("Invalid Input")
                        return

                # Play the league on the new prompts
                    top_prompts_league, bottom_prompts_league = league(prompts, k, current_league, temp, max_tokens, False, None)
                    
                    top_prompts_league = [prompt for prompt, elo in top_prompts_league]
                    bottom_prompts_league = [prompt for prompt, elo in bottom_prompts_league]
                    
                    # Merge the top prompts of league 1 and 2 into a new list
                    merged_top_prompts = top_prompts_league_1 + top_prompts_league
                    merged_bottom_prompts = bottom_prompts_league_1 + bottom_prompts_league

                # Play the final league using the merged top prompts from league 1 and 2
                    top_prompts_final, bottom_prompts_final, final_elo_dict = final_league(merged_top_prompts, current_league, k, temp, max_tokens)

                    # Get only the prompts from the top_pro_prompts_final, bottom_pro_prompts_final, top_con_prompts_final, bottom_con_prompts_final
                    top_prompts_final = [prompt for prompt, elo in top_prompts_final]
                    bottom_prompts_final = [prompt for prompt, elo in bottom_prompts_final]

                    current_league+=1
                
                elif(current_league>2):

                    # Merge the top prompts of final
                    prompts = get_prompts(no_of_prompts_between, current_league, temp, max_tokens, top_prompts_final, bottom_prompts_league, False, None)

                    if prompts is None:
                        print("Invalid Input")
                        return
                    
                    top_prompts_league, bottom_prompts_league = league(prompts, k, current_league, temp, max_tokens, False, None)

                    top_prompts_league = [prompt for prompt, elo in top_prompts_league]
                    bottom_prompts_league = [prompt for prompt, elo in bottom_prompts_league]

                    # Merge the top prompts of league and final
                    merged_top_prompts = top_prompts_final + top_prompts_league
                    merged_bottom_prompts = bottom_prompts_final + bottom_prompts_league

                    # Play the final league using the merged top pro and con prompts from league and final
                    top_prompts_final, bottom_prompts_final, final_elo_dict = final_league(merged_top_prompts, current_league, k, temp, max_tokens)

                    top_prompts_final = [prompt for prompt, elo in top_prompts_final]
                    bottom_prompts_final = [prompt for prompt, elo in bottom_prompts_final]

                    consecutive_fails = compare_elo_average(top_prompts_league, final_elo_dict, current_league, k, consecutive_fails)

                    if (consecutive_fails>=3):
                        break
                
                    current_league+=1
            
        
        # Proof of concept runs
        last_league_no = 2
        # Find last league number by checking if the "prompts_{last_league_no}.csv" file exists in this current directory
        while (f'prompts_{last_league_no}.csv') in os.listdir():
            # Print the directory
            # print("Directory: ", os.listdir())
            last_league_no+=1
            # print("Last_league_no: ", last_league_no)
        last_league_no-=1
        print("Last_league_no: ", last_league_no)
        
        proof_of_concept(1, last_league_no, k , temp, max_tokens)
        proof_of_concept(2, last_league_no, k , temp, max_tokens)
        proof_of_concept(3, last_league_no, k , temp, max_tokens)
        proof_of_concept(2, last_league_no-1, k , temp, max_tokens)

# # Function to find rogue score
# def rogue_score_matrix():

#     # Initialise the rogue_score_matrix_dict_pro
#     rogue_score_matrix_dict_pro = {}

#     # Initialise the rogue_score_matrix_dict_con
#     rogue_score_matrix_dict_con = {}

#     # Read the top_bottom_prompts_dict_pro and top_bottom_prompts_dict_con json files
#     with open('top_bottom_prompts_dict_pro.json', 'r') as json_file:
#         top_bottom_prompts_dict_pro = json.load(json_file)
    
#     # Read the top_bottom_prompts_dict_con json file
#     with open('top_bottom_prompts_dict_con.json', 'r') as json_file:
#         top_bottom_prompts_dict_con = json.load(json_file)
    
#     # Read tjhe top_bottom_prompts_dict_pro_across_league json file
#     with open('top_bottom_prompts_dict_pro_across_league.json', 'r') as json_file:
#         top_bottom_prompts_dict_pro_across_league = json.load(json_file)
    
#     # Read the top_bottom_prompts_dict_con_across_league json file
#     with open('top_bottom_prompts_dict_con_across_league.json', 'r') as json_file:
#         top_bottom_prompts_dict_con_across_league = json.load(json_file)

#     # Iterate through the league number of the top_bottom_prompts_dict_pro
#     for league_no, league_dict in top_bottom_prompts_dict_pro.items():
#         # Get the top k prompts from the league_dict
#         top_k_prompts_pro = [item["prompt"] for item in league_dict["top_k_prompts"].values()]

#         # Iterature through the league number from league_no+1 onwards in the top_bottom_prompts_dict_pro_across_league
#         for league_no_across, league_dict_across in top_bottom_prompts_dict_pro_across_league.items():
#             # Check if the json file exists
#             if os.path.exists("rogue_score_matrix_pro.json"):
#                 with open("rogue_score_matrix_pro.json", 'r') as json_file:
#                     rogue_score_matrix_dict_pro = json.load(json_file)
#                 # Check if the rogue score is already calculated for this league_no vs league_no_across
#                 if f"{league_no} vs 1-{league_no_across}" in rogue_score_matrix_dict_pro:
#                     print("Already calculated")
#                     continue

#             top_k_prompts_pro_across = [item["prompt"] for item in league_dict_across["top_k_prompts"].values()]
#             rogue_L_average = 0

#             # Get the rogue score of each prompt of top_k_prompts_pro with each prompt of top_k_prompts_pro_across using the rogue_score function
#             for prompt in top_k_prompts_pro:
#                 for prompt_across in top_k_prompts_pro_across:
#                     results = rogue.compute(predictions=[prompt], references=[prompt_across])
#                     # Get the rogue L score from the results
#                     rogue_L_score = results["rougeL"]
#                     print("Prompt: ", prompt)
#                     print("Prompt Across: ", prompt_across)
#                     print("Rogue L Score: ", rogue_L_score)
#                     rogue_L_average += rogue_L_score
                
            
#             # Get the average rogue L score of the top_k_prompts_pro with the top_k_prompts_pro_across
#             rogue_L_average = rogue_L_average/(len(top_k_prompts_pro)*len(top_k_prompts_pro_across))

#             # Store the rogue L average in a json file named rogue_score_matrix_pro.json
#             rogue_score_matrix_dict_pro[f"{league_no} vs 1-{league_no_across}"] = rogue_L_average

#             # Dump the dict in the json file
#             with open("rogue_score_matrix_pro.json", 'w') as json_file:
#                 json.dump(rogue_score_matrix_dict_pro, json_file, indent=4)

#     # Iterature through the league number of the top_bottom_prompts_dict_con
#     for league_no, league_dict in top_bottom_prompts_dict_con.items():
#         # Get the top k prompts from the league_dict
#         top_k_prompts_con = [item["prompt"] for item in league_dict["top_k_prompts"].values()]


#         # Iterature through the league number from league_no+1 onwards in the top_bottom_prompts_dict_con_across_league
#         for league_no_across, league_dict_across in top_bottom_prompts_dict_con_across_league.items():
#             # Check if the json file exists
#             if os.path.exists("rogue_score_matrix_con.json"):
#                 with open("rogue_score_matrix_con.json", 'r') as json_file:
#                     rogue_score_matrix_dict_con = json.load(json_file)
#                 # Check if the rogue score is already calculated for this league_no vs league_no_across
#                 if f"{league_no} vs 1-{league_no_across}" in rogue_score_matrix_dict_con:
#                     print("Already calculated")
#                     continue

#             top_k_prompts_con_across = [item["prompt"] for item in league_dict_across["top_k_prompts"].values()]

#             rogue_L_average = 0

#             # Get the rogue score of each prompt of top_k_prompts_con with each prompt of top_k_prompts_con_across using the rogue_score function
#             for prompt in top_k_prompts_con:
#                 for prompt_across in top_k_prompts_con_across:
#                     results = rogue.compute(predictions=[prompt], references=[prompt_across])
#                     # Get the rogue L score from the results
#                     rogue_L_score = results["rougeL"]
#                     rogue_L_average += rogue_L_score
                
            
#             # Get the average rogue L score of the top_k_prompts_con with the top_k_prompts_con_across
#             rogue_L_average = rogue_L_average/(len(top_k_prompts_con)*len(top_k_prompts_con_across))

#             # Store the rogue L average in a json file named rogue_score_matrix_con.json
#             rogue_score_matrix_dict_con[f"{league_no} vs 1-{league_no_across}"] = rogue_L_average

#             # Dump the dict in the json file
#             with open("rogue_score_matrix_con.json", 'w') as json_file:
#                 json.dump(rogue_score_matrix_dict_con, json_file, indent=4)

# # Function to get the sentiment of the speech using VADER
# def get_sentiment(speech):
#     scores = sia.polarity_scores(speech)
#     compound_score = scores['compound']
#     if compound_score >= 0.05:
#         # print("Positive")
#         return "Positive", compound_score
#     elif compound_score <= -0.05:
#         # print("Negative")
#         return "Negative", compound_score
#     else:
#         # print("Neutral")
#         return "Neutral", compound_score


# Function to check if the response is in correct format, if yes then extract the answer and reason from the response for sentiment else return none
def extract_sentiment(response):

    if response == None:
        return None, None

    # Check if the response has the required keys
    required_keys = ["Answer", "Reason for the answer"]
    if not all(key in response for key in required_keys):
        print("Response does not have required keys")
        return None, None

    # Check if the answer key in response dictionary has value as positive/negative/neutral only
    if response["Answer"].strip().strip('"').strip("'").strip().strip("**").strip("*").strip().strip('"').strip() not in ["Positive", "Negative", "Neutral"]:
        print("Invalid Answer")
        return None, None
    # Check if the reason key in response dictionary has a valid reason
    if not response["Reason for the answer"].strip():
        print("Invalid Reason")
        return None, None
    
    # Get the answer and reason from the response dictionary
    answer = response["Answer"].strip().strip('"').strip("'").strip().strip("**").strip("*").strip().strip('"').strip()
    reason = response["Reason for the answer"].strip().strip('"').strip("'").strip().strip("**").strip("*").strip().strip('"').strip()

    return answer, reason


# Function to check if the speech given is of which sentiment by prompting gpt 4o mini
def get_sentiment(speech):

    # Prompt the gpt to check if the speech is formal or informal and ask it to generate the respone in json format with Answer: [Formal/Informat] and Reason: [Reason]
    instruction = (
                "You are given a speech. Determine if the speech has a positive or negative or neutral sentiment along with the reason for your answer.\n"
                "The speech is as follows:\n"
                f"{speech}\n\n"
                "Format your response in JSON with the following fields:\n"
                "{\n"
                '  "Answer": "[Positive/Negative/Neutral]",\n'
                '  "Reason for the answer": "[Explain why this speech is Positive/Negative/Neutral]",\n'
                "}\n\n"
                "Ensure all fields are filled with meaningful content."
            )

    times_runned = 0

    while times_runned < 6:
        response = api_call_openai_json("gpt-4o-mini-2024-07-18", instruction, 0.5, 500)
        # Extract the answer and reason from the response
        answer, reason = extract_sentiment(response)

        # If answer and reason is none print response
        if answer is None and reason is None:
            print("Response not in correct format\n")
            print(response)

        # Check if the answer and reason are not None
        if answer is not None and reason is not None:
            # print("Speech: ", speech)
            # print("Answer: ", answer)
            # print("Reason: ", reason)
            return answer, reason
        
        times_runned += 1

        if times_runned == 6:
            print("Failed to get the response")
            return None, None


# Function to get the sentimental analysis using the BERT
def get_sentiment_bert(speech):

    result = sentiment_pipeline(speech)

    print("Sentiment: ", result[0]['label'])
    print("Score: ", result[0]['score'])

    return result[0]['label'], result[0]['score']


# Function to get the length of the speech in words
def get_length(speech):
    words = speech.split()
    # Load the tokenizer associated with "o200k_base"
    # Encode the text and count tokens
    tokens = tokenizer.encode(speech)
    num_tokens = len(tokens)
    # Get the length of the speech
    # print("Length of the speech: ", len(words))

    # # Get the words and number of tokens
    # print("Number of tokens: ", num_tokens)

    return len(words), num_tokens

# Function to get the personal and audience pronouns in the speech
def get_pronouns(speech):

    # Define personal and audience pronouns
    personal_pronouns = {"i", "my", "me", "myself"}
    audience_pronouns = {"you", "yours", "yourself", "your"}

    # Define male features, female features, other features
    # Female features
    female_features = {
        "she", "her", "hers", "herself", 
        "woman", "women", "female", "females", 
        "girl", "girls"
    }

    # Male features
    male_features = {
        "he", "him", "his", "himself", 
        "man", "men", "male", "males", 
        "boy", "boys"
    }

    # Features that are not gendered
    other_features = {
        "it", "its", "itself", "me", "myself", 
        "our", "ourselves", "their", "theirs", 
        "them", "themselves", "they", "us", 
        "we", "you", "yourself", "yourselves"
    }


    tokens = word_tokenize(speech)
    tagged_tokens = pos_tag(tokens)
   
    # Corrected pronoun filters
    personal = [word for word, tag in tagged_tokens if (tag == "PRP" or tag == "PRP$") and word.lower() in personal_pronouns]
    audience = [word for word, tag in tagged_tokens if (tag == "PRP" or tag == "PRP$") and word.lower() in audience_pronouns]

    # print("Personal Pronouns:", personal)
    # print("Audience Pronouns:", audience)

    # Count pronouns
    personal_count = len(personal)
    audience_count = len(audience)

    # Calculate ratios (if desired)
    total_pronouns_personal_audience = personal_count + audience_count
    personal_ratio = personal_count / total_pronouns_personal_audience if total_pronouns_personal_audience > 0 else 0
    audience_ratio = audience_count / total_pronouns_personal_audience if total_pronouns_personal_audience > 0 else 0

    # print(f"Personal Pronouns: {personal_count}, Audience Pronouns: {audience_count}")
    # print(f"Personal Pronoun Ratio: {personal_ratio:.2f}, Audience Pronoun Ratio: {audience_ratio:.2f}")

    # Get the male, female and other_features
    male = [word for word, tag in tagged_tokens if word.lower() in male_features]
    female = [word for word, tag in tagged_tokens if word.lower() in female_features]
    other = [word for word, tag in tagged_tokens if word.lower() in other_features]

    # print("Male Features:", male)
    # print("Female Features:", female)
    # print("Other Features:", other)

    # Count the features of male, female, others
    male_features_count = len(male)
    female_features_count = len(female)
    other_features_count = len(other)

    # Get the ratios (if desired)
    total_features_count = male_features_count + female_features_count + other_features_count
    male_features_ratio = male_features_count / total_features_count if total_features_count > 0 else 0
    female_features_ratio = female_features_count / total_features_count if total_features_count > 0 else 0

    # print(f"Male Features: {male_features_count}, Female Features: {female_features_count}")
    # print(f"Male Feature Ratio: {male_features_ratio:.2f}, Female Feature Ratio: {female_features_ratio:.2f}")

    return personal_count, audience_count, personal_ratio, audience_ratio, male_features_count, female_features_count, male_features_ratio, female_features_ratio
    

def get_topic(speech):
    
    # Step 1: Split the speech into sentences
    sentences = sent_tokenize(speech)

    # Step 2: Initialize BERTopic
    topic_model = BERTopic()

    # Step 3: Fit BERTopic to the sentences
    topics, probs = topic_model.fit_transform(sentences)

    # Step 4: Aggregate topics
    topic_counts = Counter(topics)  # Count frequency of each topic
    dominant_topic_id = topic_counts.most_common(1)[0][0]  # Most common topic ID

    # Step 5: Get words for the dominant topic
    dominant_topic = topic_model.get_topic(dominant_topic_id)

    # Step 6: Extract keywords and scores with the maximum score
    if dominant_topic:
        max_score = max(word_score[1] for word_score in dominant_topic)  # Find the max score
        keywords_with_max_score = [(word, score) for word, score in dominant_topic if score == max_score]

        # Separate the keywords and their scores into two lists
        keywords = [word for word, score in keywords_with_max_score]
        scores = [score for word, score in keywords_with_max_score]
    else:
        keywords = []
        scores = []

    # print(f"Dominant Topic ID: {dominant_topic_id}")
    # print(f"Dominant Topic Keywords and Scores: {list(zip(keywords, scores))}")
    return keywords, scores


def get_topic_lda(speech):

    # Step 1: Preprocess the speech
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(speech.lower())

    filtered_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word.isalnum() and word not in stop_words
    ]

    # Step 2: Create a dictionary and BoW representation
    dictionary = corpora.Dictionary([filtered_tokens])
    bow_corpus = [dictionary.doc2bow(filtered_tokens)]

    # Step 3: Apply LDA to extract topics
    lda_model = LdaModel(corpus=bow_corpus, num_topics=5, id2word=dictionary, passes=10)

    # Step 4: Get topics with keyword scores
    topics = []
    for topic_id, topic_info in lda_model.show_topics(formatted=False, num_words=5):
        topics.append(
            {
                'topic_id': topic_id,
                'keywords': [(word, float(weight)) for word, weight in topic_info],  # Keyword and score
            }
        )

    return get_topic_number_with_extraction(topics, speech)


def get_topic_number_with_extraction(topics, speech):
    # Step 1: Generate the prompt
    prompt = (
        "Here is a speech:\n"
        f"\"{speech}\"\n\n"
        "Based on the following extracted topics and their associated keyword scores, identify the dominant theme of the speech. "
        "Each topic includes its keywords and the weight of each keyword in the topic:\n\n"
    )
    for topic in topics:
        keywords_with_scores = ", ".join(f"{word} ({score})" for word, score in topic['keywords'])
        prompt += f"Topic {topic['topic_id'] + 1}: {keywords_with_scores}\n"

    prompt += (
        "\nWhich topic best represents the main theme of the speech? "
        "Format your response as follows:\n"
        "Topic Number: [Enter the topic number that best represents the main theme of the speech]"
    )

    times_runned = 0
    while times_runned < 6:
        # Step 2: Use GPT to get the topic
        response = api_call_openai("gpt-4o-mini-2024-07-18",prompt, 0.5, 500).strip()
        if response.startswith("Topic Number:") or response.startswith("***Topic Number:***") or response.startswith("**Topic Number:**"):
            # Check if topic number is integer and within the range of 1-5
            try:
                topic_number = int(response.split(":")[1].strip())
                if 1 <= topic_number <= 5:
                    # print("Topic Number:", topic_number)
                    return topics[topic_number - 1]
                else:
                    print("Invalid Topic Number")
                    print(response)
            except ValueError:
                print("Invalid Topic Number")
                print(response)
                times_runned += 1
        
    print("Failed to get the topic number")
    return topics[random.randint(0, 4)]

# Function to extract emotional tone using NCRLex
def get_emotional_tone(speech):
    # Analyze emotions
    emotion = NRCLex(speech)

    # # Display emotion scores
    # print("Emotion Scores:", emotion.raw_emotion_scores)

    # # Display dominant emotions
    # print("Dominant Emotions:", emotion.top_emotions)

    return emotion.raw_emotion_scores, emotion.top_emotions

# Function to get the emotional tone using Bert 
def get_emotional_tone_bert(speech):

    result = emotion_pipeline(speech)

    # print("Emotion: ", result[0]['label'])

    return result[0]['label'], result[0]['score']


# Function to get the diversity using TTR, Lexical Diversity, Shannon's Entropy
def get_diversity(speech):
    # Preprocessing
    tokens = word_tokenize(speech.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]

    # Type-Token Ratio (TTR)
    unique_words = set(filtered_tokens)
    ttr = len(unique_words) / len(filtered_tokens)

    # Lexical Density
    pos_tags = pos_tag(filtered_tokens)
    content_words = [word for word, pos in pos_tags if pos.startswith(('N', 'V', 'J', 'R'))]
    lexical_density = len(content_words) / len(filtered_tokens)

    # Shannon's Entropy
    word_freq = Counter(filtered_tokens)
    entropy = -sum((freq / len(filtered_tokens)) * math.log2(freq / len(filtered_tokens)) for freq in word_freq.values())

        # Print results
    # print(f"Type-Token Ratio (TTR): {ttr:.2f}")
    # print(f"Lexical Density: {lexical_density:.2f}")
    # print(f"Shannon's Entropy: {entropy:.2f}")

    return ttr, lexical_density, entropy

# Function to check if the response is in correct format, if yes then extract the answer and reason from the response for formal/informal else return none
def extract_formality(response):

    if response == None:
        return None, None

    required_keys = ["Answer", "Reason for the answer"]
    # Check if the response has the required keys
    if not all(key in response for key in required_keys):
        print("Response not in correct format")
        return None, None

    # Check if the answer key in response dictionary has value as formal/informal only
    if response["Answer"].strip().strip('"').strip("'").strip().strip("**").strip("*").strip().strip('"').strip() not in ["Formal", "Informal"]:
        print("Invalid Answer")
        return None, None
    # Check if the reason key in response dictionary has a valid reason
    if not response["Reason for the answer"].strip():
        print("Invalid Reason")
        return None, None
    
    # Get the answer and reason from the response dictionary
    answer = response["Answer"].strip().strip('"').strip("'").strip().strip("**").strip("*").strip().strip('"').strip()
    reason = response["Reason for the answer"].strip().strip('"').strip("'").strip().strip("**").strip("*").strip().strip('"').strip()

    return answer, reason


# Function to check if the speech given is formal or informal by prompting gpt 4o mini
def get_formality(speech):

    # Prompt the gpt to check if the speech is formal or informal and ask it to generate the respone in json format with Answer: [Formal/Informat] and Reason: [Reason]
    instruction = (
                "You are given a speech. Determine if the speech is formal or informal along with the reason for your answer.\n"
                "The speech is as follows:\n"
                f"{speech}\n\n"
                "Format your response in JSON with the following fields:\n"
                "{\n"
                '  "Answer": "[Formal/Informal]",\n'
                '  "Reason for the answer": "[Explain why this speech is formal/informal]",\n'
                "}\n\n"
                "Ensure all fields are filled with meaningful content."
            )

    times_runned = 0

    while times_runned < 6:
        response = api_call_openai_json("gpt-4o-mini-2024-07-18", instruction, 0.5, 500)
        # Extract the answer and reason from the response
        answer, reason = extract_formality(response)

        # If answer and reason is none print response
        if answer is None and reason is None:
            print("Response not in correct format\n")
            print(response)

        # Check if the answer and reason are not None
        if answer is not None and reason is not None:
            # print("Speech: ", speech)
            # print("Answer: ", answer)
            # print("Reason: ", reason)
            return answer, reason
        
        times_runned += 1

        if times_runned == 6:
            print("Failed to get the response")
            return None, None


# Function to get the readibility
def get_readability(speech):

    # Flesch Reading Ease Score
    flesch_score = textstat.flesch_reading_ease(speech)
    # print(f"Flesch Reading Ease Score: {flesch_score}")

    # Flesch-Kincaid Grade Level
    flesch_grade = textstat.flesch_kincaid_grade(speech)
    # print(f"Flesch-Kincaid Grade Level: {flesch_grade}")

    # Gunning Fog Index
    fog_index = textstat.gunning_fog(speech)
    # print(f"Gunning Fog Index: {fog_index}")

    # SMOG Index
    smog_index = textstat.smog_index(speech)
    # print(f"SMOG Index: {smog_index}")

    # Automated Readability Index
    ari_score = textstat.automated_readability_index(speech)
    # print(f"Automated Readability Index (ARI): {ari_score}")

    return flesch_score, flesch_grade, fog_index, smog_index, ari_score

# Function to call all the above functions and returns all the analysis in a list
def analysis(speech):
    
        # Get the sentiment of the speech
        sentiment, reason = get_sentiment(speech)
    
        # Get the length of the speech
        length, num_tokens = get_length(speech)
    
        # Get the personal and audience pronouns in the speech
        personal_count, audience_count, personal_ratio, audience_ratio, male_features_count, female_features_count, male_features_ratio, female_features_ratio = get_pronouns(speech)
    
        # Get the topic of the speech using LDA
        topic = get_topic_lda(speech)
    
        # Get the emotional tone of the speech using bert
        emotional_tone, emotion_scores = get_emotional_tone_bert(speech)
    
        # Get the diversity of the speech
        ttr, lexical_density, entropy = get_diversity(speech)
    
        # Get the formality of the speech
        formality, reason = get_formality(speech)
    
        # Get the readability of the speech
        flesch_score, flesch_grade, fog_index, smog_index, ari_score = get_readability(speech)
    
        return [sentiment, reason, length, num_tokens, personal_count, audience_count, personal_ratio, audience_ratio, male_features_count, female_features_count, male_features_ratio, female_features_ratio, topic, emotional_tone, emotion_scores, ttr, lexical_density, entropy, formality, reason, flesch_score, flesch_grade, fog_index, smog_index, ari_score]

def main():

    # get_sentiment_bert("I am happy")
    # get_length("I am happy")
    # get_pronouns("We need to tackle her well or he is happy")
    # get_topic("We need to tackle climate change immediately by using renewable energy. Economic reforms are vital for job creation and growth. Healthcare should be affordable and accessible to everyone in the country. Education is the foundation for a brighter future. We must address climate change and ensure sustainability for future generations.")
    # get_topic_lda("We need to tackle climate change immediately by using renewable energy. Economic reforms are vital for job creation and growth. Healthcare should be affordable and accessible to everyone in the country. Education is the foundation for a brighter future. We must address climate change and ensure sustainability for future generations.")
    # get_emotional_tone("We need to tackle climate change immediately by using renewable energy. Economic reforms are vital for job creation and growth. Healthcare should be affordable and accessible to everyone in the country. Education is the foundation for a brighter future. We must address climate change and ensure sustainability for future generations.")
    # get_emotional_tone_bert("We need to tackle climate change immediately by using renewable energy. Economic reforms are vital for job creation and growth. Healthcare should be affordable and accessible to everyone in the country. Education is the foundation for a brighter future. We must address climate change and ensure sustainability for future generations.")
    # get_diversity("We need to tackle climate change immediately by using renewable energy. Economic reforms are vital for job creation and growth. Healthcare should be affordable and accessible to everyone in the country. Education is the foundation for a brighter future. We must address climate change and ensure sustainability for future generations.")
    # get_formality("We need to tackle climate change immediately by using renewable energy. Economic reforms are vital for job creation and growth. Healthcare should be affordable and accessible to everyone in the country. Education is the foundation for a brighter future. We must address climate change and ensure sustainability for future generations.")
    # get_readability("We need to tackle climate change immediately by using renewable energy. Economic reforms are vital for job creation and growth. Healthcare should be affordable and accessible to everyone in the country. Education is the foundation for a brighter future. We must address climate change and ensure sustainability for future generations.")


    global elo_ratings_dict
    global league_dict
    global top_bottom_prompts_dict
    global human_prompts_used_dict
    global top_bottom_prompts_dict_across_league
    global current_league
    global origin_league_prompts_dict
    global top_bottom_prompts_dict_poc

    current_directory = os.getcwd()
    temp = [0.5]
    max_tokens =[400]
    n = 4

    no_of_runs = 0
    start_main_time = time.time()

    for no_of_prompts_start in range (10,11,2):
        for no_of_prompts_between in range (8,11,2):
            for k in range (3,math.floor(no_of_prompts_between/2),1):
                for temperature in temp:
                    for max_token in max_tokens:
                        if (no_of_runs==1):
                            break
                        print("no_of_prompts_start: ", no_of_prompts_start)
                        print("no_of_prompts_between: ", no_of_prompts_between)
                        print("k: ", k)
                        print("temperature: ", temperature)
                        print("max_token: ", max_token)
                        folder_name = "no_of_prompts_start_"+str(no_of_prompts_start)+"_no_of_prompts_between_"+str(no_of_prompts_between)+"_k_"+str(k)+"_temperature_"+str(temperature)+"_max_token_"+str(max_token)
                        new_directory = os.path.join(current_directory, folder_name)

                        # Check if the directory is already present then don't make the directory again
                        if os.path.exists(new_directory):
                            print(f"Directory {new_directory} already exists!")
                            
                        else:
                            os.makedirs(new_directory, exist_ok=True)

                        os.chdir(new_directory)
                        tournament(no_of_prompts_start, no_of_prompts_between, k, n, temperature, max_token, current_directory)
                        # Dictionary to store the ELO ratings of the prompts
                        elo_ratings_dict = {}
                        # Dictionary to store the history of the games in the leagues
                        league_dict = {}
                        #league number
                        current_league = 1
                        # Dictionary to store the top and bottom k prompts from each league
                        top_bottom_prompts_dict = {}
                        # Dictionary to store the human prompts which have been used in a league already
                        human_prompts_used_dict = {}
                        top_bottom_prompts_dict_across_league = {}
                        origin_league_prompts_dict = {}
                        top_bottom_prompts_dict_poc = {}
                        no_of_runs += 1
                            
    end_main_time = time.time()
    print("Total time taken for all the runs: ", end_main_time - start_main_time)
    
    # rogue_score_matrix()

    # prompts = set()
    # role = 'Pro'
    # with open(csv_path, mode='r') as file:  
    #     csv_reader = csv.reader(file)
    #     # Iterate through the rows and add them to the list except for the header for the specific role
    #     for row in csv_reader:
    #         if row[0] != "Prompt ID" and row[2] == role:
    #             prompts.add(row[1])
    # prompts = list(prompts)
    
    # no_of_prompts = 5
    # # Now select only those prompts which have not already played in any league before
    # if role == 'Pro':
    #     prompts = [prompt for prompt in prompts if prompt not in human_prompts_used_dict_pro.values()]
    # elif role == 'Con':
    #     prompts = [prompt for prompt in prompts if prompt not in human_prompts_used_dict_con.values()]

    # # Randomly sample no_of_prompts from the prompts list
    # prompts = sample(prompts, no_of_prompts)
    # print(prompts)
    # print(len(prompts))

    # # current_directory = os.getcwd()
    # directory_name = "new_folder"
    # new_directory = os.path.join(current_directory, directory_name)
    # os.makedirs(new_directory, exist_ok=True)
    # csv_file_name = "example.csv"
    # csv_file_path = os.path.join(new_directory, csv_file_name)
    # # Write data to the CSV file
    # data = [
    # ["Name", "Age", "City"],
    # ["Alice", 30, "New York"],
    # ["Bob", 25, "Los Angeles"],
    # ["Charlie", 35, "Chicago"]
    # ]
    # with open(csv_file_path, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerows(data)
    # # print(new_directory)


main()