import sys
import os

# Add the parent directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Now you can import everything from imports.py
from imports import *


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
top_bottom_prompts_dict_pro = {}
top_bottom_prompts_dict_con = {}
# Dictionary to store the human prompts which have been used in a league already
human_prompts_used_dict_pro = {}
human_prompts_used_dict_con = {}
# Dictonary to store top k prompts across leagues for each role by using the final league
top_bottom_prompts_dict_con_across_league = {}
top_bottom_prompts_dict_pro_across_league = {}
origin_league_prompts_dict_pro = {}
origin_league_prompts_dict_con = {}
top_bottom_prompts_dict_pro_poc = {}
top_bottom_prompts_dict_con_poc = {}

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

    return json.loads(response.choices[0].message.content.strip())


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

def get_debate_topic(no_of_topics, temp, max_tokens, file_path):
    """
    Parameters
    ----------
    no_of_topics: int
        Number of topics required for the Debate Battle
    
    Returns
    --------
    topics: list
        List of topics generated for the Debate Battle
    """

    topics = []
    topic_set = set()
    last_topic_id = get_last_topic_id(file_path)

    instruction_for_topics = """Please generate a unique debate topic which is debatable and interesting.
    
    Make sure that the response starts with 'Topic: ' followed by the topic text"""
    
    while not check_topics(no_of_topics, file_path):

        response_text = api_call_openai("gpt-4o-mini-2024-07-18", instruction_for_topics, temp, max_tokens)

        # Store all the prompts in a set to remove duplicates and store them in a csv file
        if response_text.startswith("Topic:"):
            topic = response_text[len("Topic:"):].strip().strip('"').strip('"')
            if topic not in topic_set:
                topic_set.add(topic)
                with open(file_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    if f.tell() == 0:
                        writer.writerow(["Topic ID", "Topic"])
                    last_topic_id += 1
                    writer.writerow([last_topic_id, topic])

        else:
            print(response_text)
            print("Response did not start with 'Topic:', retrying...")
        
    topics = pd.read_csv(file_path)["Topic"].to_list()
    # print(topics[:5])

    # print(topics)
    return topics

def check_topics(no_of_topics, file_path):
    """
    Parameters
    ----------
    no_of_topics: int
        Number of topics required for the Debate Battle
    file_name: str
        Name of the file to store the topics
        
    Returns
    --------
    bool
        True if the number of topics required have been generated, False otherwise
    """

    if not os.path.exists(file_path):
        return False
    
    df = pd.read_csv(file_path)

    df.drop_duplicates("Topic")

    df["Topic ID"] = range(0, len(df))

    df.to_csv(file_path, index=False)

    if len(df)>= no_of_topics+1:

        return True

    else:

        return False

def get_last_topic_id(file_path):
    """
    Parameters
    ----------
    file_name: str
        Name of the file to read the last topic ID from
        
    Returns
    --------
    last_topic_id: int
        The last used topic ID
    """

    if not os.path.exists(file_path):
        return 0
    
    with open(file_path, "r") as f:
        reader = csv.reader(f)
        data = list(reader)
        if len(data) > 1:
            return int(data[-1][0])
        else:
            return 0

def get_prompts(no_of_prompts, role, league_no, temp, max_tokens, top_k_prompts=None, bottom_k_prompts=None, human=False, human_file=None):
    """
    Parameters
    ----------
    no_of_prompts: int
        Number of prompts required for the Debate Battle for the specific role
    role: str
        Role of the debater ('Pro' or 'Con')
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

    global human_prompts_used_dict_pro
    global human_prompts_used_dict_con
    
    if human and not human_file:
        print("Error in input! Please provide the human_file for the human prompts to be read.")
        return None
    
    existing_prompts = set()
    if os.path.exists(f"prompts_{league_no}.csv"):
        # Read existing prompts from the CSV file
        with open(f"prompts_{league_no}.csv", mode='r') as file:  
            csv_reader = csv.reader(file)
            for row in csv_reader:
                if row[0] != "Prompt ID" and row[2] == role:
                    existing_prompts.add(row[1])
    
    if human:
        if (check_league(1)):
            # Get the specified role prompts from the league_results.json file
            with open('league_results.json', 'r') as json_file:
                data = json.load(json_file)

            prompts = set()

            for battle_no, battle_dict in data[str(league_no)].items():
                if role == 'Pro':
                    pro_prompt = battle_dict['pro-prompt']['prompt']
                    prompts.add(pro_prompt)
                elif role == 'Con':
                    con_prompt = battle_dict['con-prompt']['prompt']
                    prompts.add(con_prompt)

            prompts = list(prompts)

            return prompts


        else: 
            prompts = set()
            with open(human_file, mode='r') as file:  
                csv_reader = csv.reader(file)
                for row in csv_reader:
                    if row[0] != "Prompt ID" and row[2] == role:
                        prompts.add(row[1])
            prompts = list(prompts)

            if role == 'Pro':
                prompts = [prompt for prompt in prompts if prompt not in existing_prompts and prompt not in human_prompts_used_dict_pro.values()]
            elif role == 'Con':
                prompts = [prompt for prompt in prompts if prompt not in existing_prompts and prompt not in human_prompts_used_dict_con.values()]

            prompts = sample(prompts, no_of_prompts)
            return prompts

    last_prompt_id = get_last_prompt_id(f"prompts_{league_no}.csv", role)

    # if top_k_prompts is None and bottom_k_prompts is None:
    #     if role == 'Pro' and len(existing_prompts)==0:
    #         instruction_for_prompts = """I am a Large Language Model playing a Debate Battle. Help me make a prompt to win the game for my role as the Pro side debater.

    #         Make a creative, optimal prompt to help a large language model be good in their role as the Pro side debater.

    #         Note that the prompt you are generating is independent of the debate topic. You are only generating a prompt for the Pro side debater.

    #         Make sure that the response starts with 'Prompt: ' followed by the prompt text."""
        
    #     elif role == 'Pro' and len(existing_prompts)!=0:
    #         instruction_for_prompts = f"""I am a Large Language Model playing a Debate Battle. Help me make a prompt to win the game for my role as the Pro side debater.

    #         Make a creative, optimal prompt to help a large language model be good in their role as the Pro side debater.

    #         Additionally, I already have a set of prompts that have been generated so far:

    #         Existing prompts:
    #         {existing_prompts}

    #         Do not combine, replicate, or produce anything similar to these existing prompts. Your goal is to create a completely new and unique prompt.

    #         Note that the prompt you are generating is independent of the debate topic. You are only generating a prompt for the Pro side debater.

    #         Make sure that the response starts with 'Prompt: ' followed by the prompt text."""


    #     elif role == 'Con' and len(existing_prompts)==0:
    #         instruction_for_prompts = """I am a Large Language Model playing a Debate Battle. Help me make a prompt to win the game for my role as the Con side debater.

    #         Make a creative, optimal prompt to help a large language model be good in their role as the Con side debater.

    #         Note that the prompt you are generating is independent of the debate topic. You are only generating a prompt for the Con side debater.

    #         Make sure that the response starts with 'Prompt: ' followed by the prompt text."""

    #     elif role == 'Con' and len(existing_prompts)!=0:
    #         instruction_for_prompts = f"""I am a Large Language Model playing a Debate Battle. Help me make a prompt to win the game for my role as the Con side debater.

    #         Make a creative, optimal prompt to help a large language model be good in their role as the Con side debater.

    #         Additionally, I already have a set of prompts that have been generated so far:

    #         Existing prompts:
    #         {existing_prompts}

    #         Do not combine, replicate, or produce anything similar to these existing prompts. Your goal is to create a completely new and unique prompt.

    #         Note that the prompt you are generating is independent of the debate topic. You are only generating a prompt for the Con side debater.

    #         Make sure that the response starts with 'Prompt: ' followed by the prompt text.""" 

    # else:
    #     top_prompts_text = "\n".join([f'{i+1}. "Prompt: {prompt}"' for i, prompt in enumerate(top_k_prompts)])
    #     bottom_prompts_text = "\n".join([f'{i+1}. "Prompt: {prompt}"' for i, prompt in enumerate(bottom_k_prompts)])
        
    #     if role == 'Pro' and len(existing_prompts) != 0:
    #         instruction_for_prompts = f"""I am a Large Language Model playing a Debate Battle. Help me create a prompt to excel in my role as the Pro side debater.

    #         To assist you in generating better prompts, here are some examples of the best and worst prompts from previous games:

    #         Best prompts:
    #         {top_prompts_text}

    #         Worst prompts:
    #         {bottom_prompts_text}

    #         Additionally, I already have a set of prompts that have been generated so far:

    #         Existing prompts:
    #         {existing_prompts}

    #         Use the best and worst prompts to understand which strategies have worked well or poorly in the past. Your goal is not to combine, replicate, or produce anything similar to the existing or best prompts. Instead, use the insights from the best prompts to create a completely new and unique prompt that can outperform the previous best examples.

    #         Note that the prompt you are generating is independent of the debate topic. You are only generating a prompt for the Pro side debater.

    #         Make sure that the response starts with 'Prompt: ' followed by the prompt text."""

    #     elif role == 'Pro' and len(existing_prompts) == 0:
    #         instruction_for_prompts = f"""I am a Large Language Model playing a Debate Battle. Help me create a prompt to excel in my role as the Pro side debater.

    #     To assist you in generating better prompts, here are some examples of the best and worst prompts from previous games:

    #     Best prompts:
    #     {top_prompts_text}

    #     Worst prompts:
    #     {bottom_prompts_text}

    #     Use the best and worst prompts to understand which strategies have worked well or poorly in the past. Your goal is not to combine, replicate, or produce anything similar to the best prompts. Instead, use the insights from the best prompts to create a completely new and unique prompt that can outperform the previous best examples.

    #     Note that the prompt you are generating is independent of the debate topic. You are only generating a prompt for the Pro side debater.

    #     Make sure that the response starts with 'Prompt: ' followed by the prompt text."""

    #     elif role == 'Con' and len(existing_prompts) != 0:
    #         instruction_for_prompts = f"""I am a Large Language Model playing a Debate Battle. Help me create a prompt to excel in my role as the Con side debater.

    #     To assist you in generating better prompts, here are some examples of the best and worst prompts from previous games:

    #     Best prompts:
    #     {top_prompts_text}

    #     Worst prompts:
    #     {bottom_prompts_text}

    #     Additionally, I already have a set of prompts that have been generated so far:

    #     Existing prompts:
    #     {existing_prompts}

    #     Use the best and worst prompts to understand which strategies have worked well or poorly in the past. Your goal is not to combine, replicate, or produce anything similar to the existing or best prompts. Instead, use the insights from the best prompts to create a completely new and unique prompt that can outperform the previous best examples.

    #     Note that the prompt you are generating is independent of the debate topic. You are only generating a prompt for the Con side debater.

    #     Make sure that the response starts with 'Prompt: ' followed by the prompt text."""
            
    #     elif role == 'Con' and len(existing_prompts) == 0:
    #         instruction_for_prompts = f"""I am a Large Language Model playing a Debate Battle. Help me create a prompt to excel in my role as the Con side debater.

    #     To assist you in generating better prompts, here are some examples of the best and worst prompts from previous games:

    #     Best prompts:
    #     {top_prompts_text}

    #     Worst prompts:
    #     {bottom_prompts_text}

    #     Use the best and worst prompts to understand which strategies have worked well or poorly in the past. Your goal is not to combine, replicate, or produce anything similar to the best prompts. Instead, use the insights from the best prompts to create a completely new and unique prompt that can outperform the previous best examples.

    #     Note that the prompt you are generating is independent of the debate topic. You are only generating a prompt for the Con side debater.

    #     Make sure that the response starts with 'Prompt: ' followed by the prompt text."""
            
    prompts = []
    prompt_set = existing_prompts.copy()
    
    
    while not check_prompts(no_of_prompts, f"prompts_{league_no}.csv", role):
        times_runned = 0
        candidate_prompts_set = set()
        while(len(candidate_prompts_set)<4):
            if(times_runned>10):
                print("Unable to generate prompts! Limit exceeded")
                break

            json_response = api_call_openai_json("gpt-4o-mini-2024-07-18", instruction_for_prompts(existing_prompts, role, candidate_prompts_set, top_k_prompts, bottom_k_prompts), temp, max_tokens)
            prompt = extract_prompt(json_response)
            if prompt is None:
                print(json_response)
                times_runned += 1
                print("Response did not contain the required keys, retrying...")
                continue
            if prompt not in prompt_set:
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
                print("Prompt already exists in the prompt set, retrying...")
                times_runned += 1

        best_prompt, reason = finding_best_prompt_from_candidates(existing_prompts, list(candidate_prompts_set), role, temp, max_tokens, top_k_prompts, bottom_k_prompts)
        if best_prompt not in prompt_set:
            prompt_set.add(best_prompt)
            existing_prompts.add(best_prompt)
            last_prompt_id += 1
            with open(f"prompts_{league_no}.csv", "a", newline="") as f:
                writer = csv.writer(f)
                if f.tell() == 0:
                    writer.writerow(["Prompt ID", "Prompt", "Role"])
                writer.writerow([last_prompt_id, best_prompt, role])
        else:
            print("Best prompt already exists in the prompt set, retrying...")
            times_runned += 1
        
    # print("Prompts generated successfully!")
    with open(f"prompts_{league_no}.csv", mode='r') as file:  
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if row[0] != "Prompt ID" and row[2] == role:
                prompts.append(row[1])
    # print(prompts)
    # bert_score()
    rogue_score()
    return prompts

def bert_score():
    """
    Calculate the BERT score for the prompts generated using the prompts csv files and store it in a json file.
    The index of the prompt is used as the key in the JSON file.
    """
    # Calculate the BERT score for the prompts_{current_league}.csv file with each other 
    df_pro = pd.read_csv(f"prompts_{current_league}.csv")
    df_pro = df_pro[df_pro["Role"] == "Pro"]
    df_con = pd.read_csv(f"prompts_{current_league}.csv")
    df_con = df_con[df_con["Role"] == "Con"]
    
    pro_prompts = df_pro["Prompt"].to_list()
    con_prompts = df_con["Prompt"].to_list()

    # Function to check if a key exists in JSON file
    def key_exists_in_json(filename, key):
        if os.path.exists(filename):
            with open(filename, "r") as f:
                data = json.load(f)
            return str(key) in data
        return False

    # Process Pro Prompts
    for i in range(len(pro_prompts) - 1):
        references = pro_prompts[i + 1:]
        predictions = [pro_prompts[i]] * len(references)
        
        # Check if the key for this prompt already exists in the JSON file
        if key_exists_in_json(f"rogue_score_{current_league}_pro.json", i):
            print(f"Skipping calculation for pro prompt {i}, already exists in the file.")
            continue
        
        print("going to calculate results")
        # Compute the BERT score
        results = bertscore.compute(predictions=predictions, references=references, lang="en", model_type="distilbert-base-uncased")
        print("results computed")
        print(i)

        # Check if the JSON file exists and update it
        if os.path.exists(f"rogue_score_{current_league}_pro.json"):
            with open(f"rogue_score_{current_league}_pro.json", "r") as f:
                data = json.load(f)
            data[str(i)] = results  # Store the results under the index `i`
        else:
            data = {str(i): results}

        # Write back to the JSON file
        with open(f"rogue_score_{current_league}_pro.json", "w") as f:
            json.dump(data, f)
    
    # Process Con Prompts
    for i in range(len(con_prompts) - 1):
        references = con_prompts[i + 1:]
        predictions = [con_prompts[i]] * len(references)

        # Check if the key for this prompt already exists in the JSON file
        if key_exists_in_json(f"bert_score_{current_league}_con.json", i):
            print(f"Skipping calculation for con prompt {i}, already exists in the file.")
            continue
        
        # Compute the BERT score
        results = bertscore.compute(predictions=predictions, references=references, lang="en", model_type="distilbert-base-uncased")

        # Check if the JSON file exists and update it
        if os.path.exists(f"bert_score_{current_league}_con.json"):
            with open(f"bert_score_{current_league}_con.json", "r") as f:
                data = json.load(f)
            data[str(i)] = results  # Store the results under the index `i`
        else:
            data = {str(i): results}

        # Write back to the JSON file
        with open(f"bert_score_{current_league}_con.json", "w") as f:
            json.dump(data, f)

    # Process cross-league prompts
    for i in range(1, current_league):
        if os.path.exists(f"prompts_{current_league-i}.csv"):
            df_pro_prev = pd.read_csv(f"prompts_{current_league-i}.csv")
            df_pro_prev = df_pro_prev[df_pro_prev["Role"] == "Pro"]
            df_con_prev = pd.read_csv(f"prompts_{current_league-i}.csv")
            df_con_prev = df_con_prev[df_con_prev["Role"] == "Con"]
            
            pro_prompts_prev = df_pro_prev["Prompt"].to_list()
            con_prompts_prev = df_con_prev["Prompt"].to_list()

            # Process cross-league pro prompts
            for j in range(len(pro_prompts)):
                if key_exists_in_json(f"rogue_score_{current_league}_pro_{current_league-i}.json", j):
                    print(f"Skipping calculation for cross-league pro prompt {j}, already exists in the file.")
                    continue
                
                references = pro_prompts_prev
                predictions = [pro_prompts[j]] * len(references)
                
                results = bertscore.compute(predictions=predictions, references=references, lang="en", model_type="distilbert-base-uncased")
                
                # Check if the JSON file exists and update it
                if os.path.exists(f"rogue_score_{current_league}_pro_{current_league-i}.json"):
                    with open(f"rogue_score_{current_league}_pro_{current_league-i}.json", "r") as f:
                        data = json.load(f)
                    data[str(j)] = results  # Store the results under the index `j`
                else:
                    data = {str(j): results}

                # Write back to the JSON file
                with open(f"rogue_score_{current_league}_pro_{current_league-i}.json", "w") as f:
                    json.dump(data, f)
            
            # Process cross-league con prompts
            for j in range(len(con_prompts)):
                if key_exists_in_json(f"bert_score_{current_league}_con_{current_league-i}.json", j):
                    print(f"Skipping calculation for cross-league con prompt {j}, already exists in the file.")
                    continue
                
                references = con_prompts_prev
                predictions = [con_prompts[j]] * len(references)
                
                results = bertscore.compute(predictions=predictions, references=references, lang="en", model_type="distilbert-base-uncased")
                
                # Check if the JSON file exists and update it
                if os.path.exists(f"bert_score_{current_league}_con_{current_league-i}.json"):
                    with open(f"bert_score_{current_league}_con_{current_league-i}.json", "r") as f:
                        data = json.load(f)
                    data[str(j)] = results  # Store the results under the index `j`
                else:
                    data = {str(j): results}

                # Write back to the JSON file
                with open(f"bert_score_{current_league}_con_{current_league-i}.json", "w") as f:
                    json.dump(data, f)
def rogue_score():
    """
    Calculate the ROUGE score for the prompts generated using the prompts csv files and store it in a json file.
    The index of the prompt is used as the key in the JSON file.
    Separate calculations for 'Pro' and 'Con' roles.
    """
    # Load the current league prompts for "Pro" and "Con"
    df_pro = pd.read_csv(f"prompts_{current_league}.csv")
    df_pro = df_pro[df_pro["Role"] == "Pro"]
    df_con = pd.read_csv(f"prompts_{current_league}.csv")
    df_con = df_con[df_con["Role"] == "Con"]
    
    pro_prompts = df_pro["Prompt"].to_list()
    con_prompts = df_con["Prompt"].to_list()

    # Function to check if a key exists in JSON file
    def key_exists_in_json(filename, key):
        if os.path.exists(filename):
            with open(filename, "r") as f:
                data = json.load(f)
            return str(key) in data
        return False

    # Process Pro Prompts: Compare each "Pro" prompt with subsequent "Pro" prompts
    for i in range(len(pro_prompts) - 1):
        for j in range(i + 1, len(pro_prompts)):  # Compare pro_prompts[i] with pro_prompts[j] individually
            key = f"{i}_{j}"

            # Check if the key for this prompt pair already exists in the JSON file
            if key_exists_in_json(f"rogue_score_{current_league}_pro.json", key):
                print(f"Skipping calculation for pro prompt pair {i}-{j}, already exists in the file.")
                continue
            
            # Compute the ROUGE score between pro_prompts[i] and pro_prompts[j]
            results = rogue.compute(predictions=[pro_prompts[i]], references=[pro_prompts[j]])

            # Check if the JSON file exists and update it
            if os.path.exists(f"rogue_score_{current_league}_pro.json"):
                with open(f"rogue_score_{current_league}_pro.json", "r") as f:
                    data = json.load(f)
                data[key] = results  # Store the results under the key `i_j`
            else:
                data = {key: results}

            # Write back to the JSON file
            with open(f"rogue_score_{current_league}_pro.json", "w") as f:
                json.dump(data, f)

    # Process Con Prompts: Compare each "Con" prompt with subsequent "Con" prompts
    for i in range(len(con_prompts) - 1):
        for j in range(i + 1, len(con_prompts)):  # Compare con_prompts[i] with con_prompts[j] individually
            key = f"{i}_{j}"

            # Check if the key for this prompt pair already exists in the JSON file
            if key_exists_in_json(f"rogue_score_{current_league}_con.json", key):
                print(f"Skipping calculation for con prompt pair {i}-{j}, already exists in the file.")
                continue
            
            # Compute the ROUGE score between con_prompts[i] and con_prompts[j]
            results = rogue.compute(predictions=[con_prompts[i]], references=[con_prompts[j]])

            # Check if the JSON file exists and update it
            if os.path.exists(f"rogue_score_{current_league}_con.json"):
                with open(f"rogue_score_{current_league}_con.json", "r") as f:
                    data = json.load(f)
                data[key] = results  # Store the results under the key `i_j`
            else:
                data = {key: results}

            # Write back to the JSON file
            with open(f"rogue_score_{current_league}_con.json", "w") as f:
                json.dump(data, f)

    # Process cross-league "Pro" prompts
    for i in range(1, current_league):
        if os.path.exists(f"prompts_{current_league-i}.csv"):
            df_pro_prev = pd.read_csv(f"prompts_{current_league-i}.csv")
            df_pro_prev = df_pro_prev[df_pro_prev["Role"] == "Pro"]
            pro_prompts_prev = df_pro_prev["Prompt"].to_list()

            # Compare each current "Pro" prompt with previous league's "Pro" prompts
            for j in range(len(pro_prompts)):
                for k in range(len(pro_prompts_prev)):
                    key = f"{j}_{k}"
                    
                    if key_exists_in_json(f"rogue_score_{current_league}_pro_{current_league-i}.json", key):
                        print(f"Skipping calculation for cross-league pro prompt pair {j}-{k}, already exists in the file.")
                        continue

                    results = rogue.compute(predictions=[pro_prompts[j]], references=[pro_prompts_prev[k]])

                    # Check if the JSON file exists and update it
                    if os.path.exists(f"rogue_score_{current_league}_pro_{current_league-i}.json"):
                        with open(f"rogue_score_{current_league}_pro_{current_league-i}.json", "r") as f:
                            data = json.load(f)
                        data[key] = results
                    else:
                        data = {key: results}

                    # Write back to the JSON file
                    with open(f"rogue_score_{current_league}_pro_{current_league-i}.json", "w") as f:
                        json.dump(data, f)

    # Process cross-league "Con" prompts
    for i in range(1, current_league):
        if os.path.exists(f"prompts_{current_league-i}.csv"):
            df_con_prev = pd.read_csv(f"prompts_{current_league-i}.csv")
            df_con_prev = df_con_prev[df_con_prev["Role"] == "Con"]
            con_prompts_prev = df_con_prev["Prompt"].to_list()

            # Compare each current "Con" prompt with previous league's "Con" prompts
            for j in range(len(con_prompts)):
                for k in range(len(con_prompts_prev)):
                    key = f"{j}_{k}"

                    if key_exists_in_json(f"rogue_score_{current_league}_con_{current_league-i}.json", key):
                        print(f"Skipping calculation for cross-league con prompt pair {j}-{k}, already exists in the file.")
                        continue

                    results = rogue.compute(predictions=[con_prompts[j]], references=[con_prompts_prev[k]])

                    # Check if the JSON file exists and update it
                    if os.path.exists(f"rogue_score_{current_league}_con_{current_league-i}.json"):
                        with open(f"rogue_score_{current_league}_con_{current_league-i}.json", "r") as f:
                            data = json.load(f)
                        data[key] = results
                    else:
                        data = {key: results}

                    # Write back to the JSON file
                    with open(f"rogue_score_{current_league}_con_{current_league-i}.json", "w") as f:
                        json.dump(data, f)

# Function to extract prompt text from JSON response
def extract_prompt(json_response):
    """
    Extracts the prompt text from the JSON response if it meets all criteria.
    """
    # Define the required keys
    required_keys = ["Reason for diversity", "Is this diverse", "Reason for best prompt", "Is this the best prompt", "Prompt Text"]

    # Check if all required keys are present and their values are non-zero or non-empty
    if all(key in json_response for key in required_keys) and all(json_response[key] for key in required_keys):
        # Extract the prompt text
        prompt_text = json_response["Prompt Text"].strip().strip('"').strip("'").strip().strip("**").strip("*").strip().strip('"').strip()
        return prompt_text
    else:
        # If any key is missing or has an empty/zero value, return None
        return None
def instruction_for_prompts(existing_prompts, role, candidates, top_k_prompts, bottom_k_prompts):
    # Prepare text for existing prompts
    # add existing prompts, candidate prompts, top and bottom prompts
 
    existing_and_candidate_prompts = existing_prompts.union(candidates)
    existing_and_candidate_prompts_text = ""
    if len(existing_and_candidate_prompts) != 0:
        existing_and_candidate_prompts_text = "\n".join([f'{i+1}. "Prompt: {prompt}"' for i, prompt in enumerate(existing_and_candidate_prompts)])
    if top_k_prompts is not None:
        top_prompts_text = "\n".join([f'{i+1}. "Prompt: {prompt}"' for i, prompt in enumerate(top_k_prompts)])
    if bottom_k_prompts is not None:
        bottom_prompts_text = "\n".join([f'{i+1}. "Prompt: {prompt}"' for i, prompt in enumerate(bottom_k_prompts)])

    if role == 'Pro' and top_k_prompts is None and bottom_k_prompts is None:
        if len(existing_and_candidate_prompts) == 0:
            instructions = (
                "Help me create a prompt which I can provide to an AI model to engage effectively in a debate as the Pro side debater.\n\n"
                "While generating prompts, keep the following things in mind:\n"
                "-- Your goal is to generate a completely new and unique prompt that offers a fresh, innovative perspective.\n"
                "-- The prompt should demonstrate superior effectiveness and have the potential to outperform all other prompts possible.\n"
                "-- The prompt you are generating is independent of the debate topic. Focus on crafting a unique prompt for the Pro side debater.\n\n"
                "Make sure that the response is in JSON format with the following fields:\n"
                "{\n"
                '  "Reason for diversity": "[explain why this prompt is diverse]",\n'
                '  "Is this diverse": "[true/false]",\n'
                '  "Reason for best prompt": "[explain why this prompt is the best and hence can outperform all other prompts possbile]",\n'
                '  " Is this the best prompt": "[true/false]",\n'
                '  "Prompt Text": "[write only your prompt here and dont start by Prompt:]"\n'
                "}\n\n"
                "Ensure all fields are provided and have non-zero or non-empty values."
            )
        elif len(existing_and_candidate_prompts)!=0:

            instructions = (
                "Help me create a prompt which I can provide to an AI model to engage effectively in a debate as the Pro side debater.\n\n"
                "Here is a list of existing prompts that have already been generated:\n"
                f"{existing_and_candidate_prompts_text}\n\n"
                "While generating prompts, keep the following things in mind:\n"
                "-- Do not combine, replicate, or produce anything similar to the above prompts.\n"
                "-- To ensure diversity in the prompts you generate, aim for lexical variety by minimizing the repetition of common words from above prompts.\n"
                "-- Your goal is to generate a completely new and unique prompt that is distinct from all of these and offers a fresh, innovative perspective.\n"
                "-- The prompt should demonstrate superior effectiveness and have the potential to outperform all the above prompts.\n"
                "-- The prompt you are generating is independent of the debate topic. Focus on crafting a unique prompt for the Pro side debater.\n\n"
                "Make sure that the response is in JSON format with the following fields:\n"
                "{\n"
                '  "Reason for diversity": "[explain why this prompt is diverse]",\n'
                '  "Is this diverse": "[true/false]",\n'
                '  "Reason for best prompt": "[explain why this prompt is the best and hence can outperform all above prompts]",\n'
                '  " Is this the best prompt": "[true/false]",\n'
                '  "Prompt Text": "[write only your prompt here and dont start by Prompt:]"\n'
                "}\n\n"
                "Ensure all fields are provided and have non-zero or non-empty values."
            )
    if role == 'Pro' and top_k_prompts is not None and bottom_k_prompts is not None:
        if len(existing_and_candidate_prompts) == 0:
            instructions = (
                "Help me create a prompt which I can provide to an AI model to engage effectively in a debate as the Pro side debater.\n\n"
                "Here is a list of best and worst prompts:\n"
                f"Best prompts:\n{top_prompts_text}\n\nWorst prompts:\n{bottom_prompts_text}\n\n"
                "While generating prompts, keep the following things in mind:\n"
                "-- Do not combine, replicate, or produce anything similar to the above prompts.\n"
                "-- To ensure diversity in the prompts you generate, aim for lexical variety by minimizing the repetition of common words from above prompts.\n"
                "-- Your goal is to generate a completely new and unique prompt that offers a fresh, innovative perspective.\n"
                "-- The prompt should demonstrate superior effectiveness and have the potential to outperform all the above prompts.\n"
                "-- The prompt you are generating is independent of the debate topic. Focus on crafting a unique prompt for the Pro side debater.\n\n"
                "Make sure that the response is in JSON format with the following fields:\n"
                "{\n"
                '  "Reason for diversity": "[explain why this prompt is diverse]",\n'
                '  "Is this diverse": "[true/false]",\n'
                '  "Reason for best prompt": "[explain why this prompt is the best and hence can outperform all the above prompts]",\n'
                '  " Is this the best prompt": "[true/false]",\n'
                '  "Prompt Text": "[write only your prompt here and dont start by Prompt:]"\n'
                "Ensure all fields are provided and have non-zero or non-empty values."
            )
        elif len(existing_and_candidate_prompts)!=0:
            instructions = (
                "Help me create a prompt which I can provide to an AI model to engage effectively in a debate as the Pro side debater.\n\n"
                "Here is a list of best and worst prompts:\n"
                f"Best prompts:\n{top_prompts_text}\n\nWorst prompts:\n{bottom_prompts_text}\n\n"
                "Here is a list of existing prompts that have already been generated:\n"
                f"{existing_and_candidate_prompts_text}\n\n"
                "While generating prompts, keep the following things in mind:\n"
                "-- Do not combine, replicate, or produce anything similar to the above prompts.\n"
                "-- To ensure diversity in the prompts you generate, aim for lexical variety by minimizing the repetition of common words from above prompts.\n"
                "-- Your goal is to generate a completely new and unique prompt that is distinct from all of these and offers a fresh, innovative perspective.\n"
                "-- The prompt should demonstrate superior effectiveness and have the potential to outperform all the above prompts.\n"
                "-- The prompt you are generating is independent of the debate topic. Focus on crafting a unique prompt for the Pro side debater.\n\n"
                "Make sure that the response is in JSON format with the following fields:\n"
                "{\n"
                '  "Reason for diversity": "[explain why this prompt is diverse]",\n'
                '  "Is this diverse": "[true/false]",\n'
                '  "Reason for best prompt": "[explain why this prompt is the best and hence can outperform all the above prompts]",\n'
                '  " Is this the best prompt": "[true/false]",\n'
                '  "Prompt Text": "[write only your prompt here and dont start by Prompt:]"\n'
                "}\n\n"
                "Ensure all fields are provided and have non-zero or non-empty values."
            )

    if role == 'Con' and top_k_prompts is None and bottom_k_prompts is None:
        if len(existing_and_candidate_prompts) == 0:
            instructions = (
                "Help me create a prompt which I can provide to an AI model to engage effectively in a debate as the Con side debater.\n\n"
                "While generating prompts, keep the following things in mind:\n"
                "-- Your goal is to generate a completely new and unique prompt that offers a fresh, innovative perspective.\n"
                " -- The prompt should demonstrate superior effectiveness and have the potential to outperform any other prompts possible.\n"
                "-- The prompt you are generating is independent of the debate topic. Focus on crafting a unique prompt for the Con side debater.\n\n"
                "Make sure that the response is in JSON format with the following fields:\n"
                "{\n"
                '  "Reason for diversity": "[explain why this prompt is diverse]",\n'
                '  "Is this diverse": "[true/false]",\n'
                '  "Reason for best prompt": "[explain why this prompt is the best and hence can outperform any other prompts possible]",\n'
                '  " Is this the best prompt": "[true/false]",\n'
                '  "Prompt Text": "[write only your prompt here and dont start by Prompt:]"\n'
                "}\n\n"
                "Ensure all fields are provided and have non-zero or non-empty values."
            )
        elif len(existing_and_candidate_prompts)!=0:

            instructions = (
                "Help me create a prompt which I can provide to an AI model to engage effectively in a debate as the Con side debater.\n\n"
                "Here is a list of existing prompts that have already been generated:\n"
                f"{existing_and_candidate_prompts_text}\n\n"
                "While generating prompts, keep the following things in mind:\n"
                "-- Do not combine, replicate, or produce anything similar to the above prompts.\n"
                "-- To ensure diversity in the prompts you generate, aim for lexical variety by minimizing the repetition of common words from above prompts.\n"
                "-- Your goal is to generate a completely new and unique prompt that is distinct from all of these and offers a fresh, innovative perspective.\n"
                "-- The prompt should demonstrate superior effectiveness and have the potential to outperform all the above prompts.\n"
                "-- The prompt you are generating is independent of the debate topic. Focus on crafting a unique prompt for the Con side debater.\n\n"
                "Make sure that the response is in JSON format with the following fields:\n"
                "{\n"
                '  "Reason for diversity": "[explain why this prompt is diverse]",\n'
                '  "Is this diverse": "[true/false]",\n'
                '  "Reason for best prompt": "[explain why this prompt is the best and hence can outperform all above prommpts]",\n'
                '  " Is this the best prompt": "[true/false]",\n'
                '  "Prompt Text": "[write only your prompt here and dont start by Prompt:]"\n'
                "}\n\n"
                "Ensure all fields are provided and have non-zero or non-empty values."
            )
    if role == 'Con' and top_k_prompts is not None and bottom_k_prompts is not None:

        if len(existing_and_candidate_prompts) == 0:

            instructions = (
                "Help me create a prompt which I can provide to an AI model to engage effectively in a debate as the Con side debater.\n\n"
                "Here is a list of best and worst prompts:\n"
                f"Best prompts:\n{top_prompts_text}\n\nWorst prompts:\n{bottom_prompts_text}\n\n"
                "While generating prompts, keep the following things in mind:\n"
                "-- Do not combine, replicate, or produce anything similar to the above prompts.\n"
                "-- To ensure diversity in the prompts you generate, aim for lexical variety by minimizing the repetition of common words from above prompts.\n"
                "-- Your goal is to generate a completely new and unique prompt that is distinct from all of these and offers a fresh, innovative perspective.\n"
                "-- Your prompt should demonstrate superior effectiveness and have the potential to outperform all the above prompts.\n"
                "-- The prompt you are generating is independent of the debate topic. Focus on crafting a unique prompt for the Con side debater.\n\n"
                "Make sure that the response is in JSON format with the following fields:\n"
                "{\n"
                '  "Reason for diversity": "[explain why this prompt is diverse]",\n'
                '  "Is this diverse": "[true/false]",\n'
                '  "Reason for best prompt": "[explain why this prompt is the best and hence can outperform all above prompts]",\n'
                '  " Is this the best prompt": "[true/false]",\n'
                '  "Prompt Text": "[write only your prompt here and dont start by Prompt:]"\n'
                "}\n\n"
                "Ensure all fields are provided and have non-zero or non-empty values."
            )
        
        elif len(existing_and_candidate_prompts)!=0:
            instructions = (
                "Help me create a prompt which I can provide to an AI model to engage effectively in a debate as the Con side debater.\n\n"
                "Here is a list of best and worst prompts:\n"
                f"Best prompts:\n{top_prompts_text}\n\nWorst prompts:\n{bottom_prompts_text}\n\n"
                "Here is a list of existing prompts that have already been generated:\n"
                f"{existing_and_candidate_prompts_text}\n\n"
                "While generating prompts, keep the following things in mind:\n"
                "-- Do not combine, replicate, or produce anything similar to the above prompts.\n"
                "-- To ensure diversity in the prompts you generate, aim for lexical variety by minimizing the repetition of common words from above prompts.\n"
                "-- Your goal is to generate a completely new and unique prompt that is distinct from all of these and offers a fresh, innovative perspective.\n"
                "-- The prompt should demonstrate superior effectiveness and have the potential to outperform all the above prompts.\n"
                "-- The prompt you are generating is independent of the debate topic. Focus on crafting a unique prompt for the Con side debater.\n\n"
                "Make sure that the response is in JSON format with the following fields:\n"
                "{\n"
                '  "Reason for diversity": "[explain why this prompt is diverse]",\n'
                '  "Is this diverse": "[true/false]",\n'
                '  "Reason for best prompt": "[explain why this prompt is the best and hence can outperform all above promtps]",\n'
                '  " Is this the best prompt": "[true/false]",\n'
                '  "Prompt Text": "[write only your prompt here and dont start by Prompt:]"\n'
                "}\n\n"
                "Ensure all fields are provided and have non-zero or non-empty values."
            )

    return instructions

def check_prompts(no_of_prompts, file_name, role):
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
            if row[0] != "Prompt ID" and row[2] == role:
                count += 1
        if count >= no_of_prompts:
            return True
        else:
            return False

def get_last_prompt_id(file_name, role):
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
            if row[0] != "Prompt ID" and row[2] == role:
                last_id = int(row[0])
        return last_id
    
def extract_best_prompt(json_response):
    """
    Extracts the best prompt and reason from the JSON response if it meets all criteria.
    """
    # Define the required keys
    required_keys = ["Reason for the prompt being the best", "Prompt Text"]

    # Check if all required keys are present and their values are non-zero or non-empty
    if all(key in json_response for key in required_keys) and all(json_response[key] for key in required_keys):
        # Extract the prompt text and reason
        reason = json_response["Reason for the prompt being the best"].strip().strip('"').strip("'").strip().strip("**").strip("*").strip().strip('"').strip()
        prompt_text = json_response["Prompt Text"].strip().strip('"').strip("'").strip().strip("**").strip("*").strip().strip('"').strip()
        return prompt_text, reason
    else:
        # If any key is missing or has an empty/zero value, return None
        return None, None

def finding_best_prompt_from_candidates(existing_prompts, candidates, role, temp, max_tokens, top_k_prompts=None, bottom_k_prompts=None):

    if len(existing_prompts) != 0:
        existing_prompts_text = "\n".join([f'{i+1}. "Prompt: {prompt}"' for i, prompt in enumerate(existing_prompts)])

    if top_k_prompts is None and bottom_k_prompts is None:
        if len(existing_prompts) != 0:
            instruction_to_select_best_prompt = (
                "Help me select a prompt from the 5 candidate prompts that I can give to an AI model to engage "
                f"effectively in a debate as the {role} side debater.\n\n"
                "Here are the 3 candidate prompts from which you need to select the best one:\n"
                f"1. \"{candidates[0]}\"\n"
                f"2. \"{candidates[1]}\"\n"
                f"3. \"{candidates[2]}\"\n\n"
                "The existing prompts are:\n"
                f"{existing_prompts_text}\n\n"
                "Consider the following criteria while evaluating each response:\n"
                "- **Quality**: The prompt should demonstrate superior effectiveness and have the potential to outperform "
                "any of the other candidate prompts.\n\n"
                "Make sure that the response is in JSON format with the following fields:\n"
                "{\n"
                '  "Reason for the prompt being the best": "[explain why this prompt is best among the candidates]",\n'
                '  "Prompt Text": "[write only your prompt here and dont start by Prompt:]"\n'
                "}\n\n"
                "Ensure all fields are provided and have non-zero or non-empty values."
                
            )

        elif len(existing_prompts) == 0:
            instruction_to_select_best_prompt = (
                "Help me select a prompt from the 5 candidate prompts that I can give to an AI model to engage "
                f"effectively in a debate as the {role} side debater.\n\n"
                "Here are the 3 candidate prompts from which you need to select the best one:\n"
                f"1. \"{candidates[0]}\"\n"
                f"2. \"{candidates[1]}\"\n"
                f"3. \"{candidates[2]}\"\n\n"
                "Consider the following criteria while evaluating each response:\n"
                "- **Quality**: The prompt should demonstrate superior effectiveness and have the potential to outperform "
                "any of the other candidate prompts.\n\n"
                "Make sure that the response is in JSON format with the following fields:\n"
                "{\n"
                '  "Reason for the prompt being the best": "[explain why this prompt is best among the candidates]",\n'
                '  "Prompt Text": "[write only your prompt here and dont start by Prompt:]"\n'
                "}\n\n"
                "Ensure all fields are provided and have non-zero or non-empty values."
            )

    if top_k_prompts is not None and bottom_k_prompts is not None:

        top_prompts_text = "\n".join([f'{i+1}. "Prompt: {prompt}"' for i, prompt in enumerate(top_k_prompts)])
        bottom_prompts_text = "\n".join([f'{i+1}. "Prompt: {prompt}"' for i, prompt in enumerate(bottom_k_prompts)])

        if len(existing_prompts) != 0:
            instruction_to_select_best_prompt = (
                "Help me select a prompt from the 5 candidate prompts that I can give to an AI model to engage "
                f"effectively in a debate as the {role} side debater.\n\n"
                "Here are the 3 candidate prompts from which you need to select the best one:\n"
                f"1. \"{candidates[0]}\"\n"
                f"2. \"{candidates[1]}\"\n"
                f"3. \"{candidates[2]}\"\n\n"
                "The existing prompts are:\n"
                f"{existing_prompts_text}\n\n"
                "Additionally, here are some examples of the best and worst prompts:\n\n"
                "Best prompts:\n"
                f"{top_prompts_text}\n\n"
                "Worst prompts:\n"
                f"{bottom_prompts_text}\n\n"
                "Evaluate each response based on the following criteria:\n\n"
                "- **Quality**: The prompt should demonstrate superior effectiveness and have the potential to outperform "
                "all other candidate prompts as well as the best and worst prompts.\n\n"
                "Make sure that the response is in JSON format with the following fields:\n"
                "{\n"
                '  "Reason for the prompt being the best": "[explain why this prompt is best among the candidates]",\n'
                '  "Prompt Text": "[write only your prompt here and dont start by Prompt:]"\n'
                "}\n\n"
                "Ensure all fields are provided and have non-zero or non-empty values."
            )

        elif len(existing_prompts) == 0:
            instruction_to_select_best_prompt = (
                "Help me select a prompt from the 5 candidate prompts that I can give to an AI model to engage "
                f"effectively in a debate as the {role} side debater.\n\n"
                "Here are the 3 candidate prompts:\n"
                f"1. \"{candidates[0]}\"\n"
                f"2. \"{candidates[1]}\"\n"
                f"3. \"{candidates[2]}\"\n\n"
                "Additionally, here are some examples of the best and worst prompts:\n\n"
                "Best prompts:\n"
                f"{top_prompts_text}\n\n"
                "Worst prompts:\n"
                f"{bottom_prompts_text}\n\n"
                "Evaluate each response based on the following criteria:\n"
                "- **Quality**: The prompt should demonstrate superior effectiveness and have the potential to outperform "
                "all other candidate prompts as well as the best and worst prompts.\n\n"
                "Make sure that the response is in JSON format with the following fields:\n"
                "{\n"
                '  "Reason for the prompt being the best": "[explain why this prompt is best among the candidates]",\n'
                '  "Prompt Text": "[write only your prompt here and dont start by Prompt:]"\n'
                "}\n\n"
                "Ensure all fields are provided and have non-zero or non-empty values."
            )

    candidate_prompts = [candidates[0], candidates[1], candidates[2]]

    times_runned =0
    while(times_runned<10):
        json_response = api_call_openai_json("gpt-4o-mini-2024-07-18", instruction_to_select_best_prompt, temp, max_tokens)
        best_prompt, reason = extract_best_prompt(json_response)
        if best_prompt is None or reason is None:
            print(json_response)
            print("Response did not contain the required keys, retrying...")
            times_runned += 1
            continue
        
         # Make a dictionary which has key as the prompt and value as a dictionary which has the reason and candidate prompts
        best_prompt_dict = {best_prompt: {"reason": reason, "candidates": candidates}}

        # Store this dict in a json file
        # Check if the json file already has data then update the data

        if os.path.exists(f"selected_prompt_dict_{role}.json"):
            with open(f"selected_prompt_dict_{role}.json", "r") as f:
                data = json.load(f)
                data.update(best_prompt_dict)
                with open(f"selected_prompt_dict_{role}.json", "w") as f:
                    json.dump(data, f)
        else:
            with open(f"selected_prompt_dict_{role}.json", "w") as f:
                json.dump(best_prompt_dict, f)

        return best_prompt, reason


# Function to play the Negotiation game
def game(pro_prompt, con_prompt, topic, total_rounds, temp, max_tokens):
    
        """
        Parameters
        ----------
        pro_prompt: str
            The prompt for the Pro side debater
        con_prompt: str
            The prompt for the Con side debater
        topic: str
            The topic for the Debate Battle
        total_rounds: int
            The total number of rounds in the Debate Battle
    
        Returns
        --------
        winner: str
            The winner of the Debate Battle
        """

        criteria = """Reasoning and Evidence: Evaluate each debater's logical consistency, strength of arguments, and use of relevant evidence. Strong arguments should be well-supported by facts or examples and free of logical fallacies.

Listening and Response: Assess how effectively each debater addresses and rebuts the opponent’s arguments. Consider their ability to identify and counter key points made by the opposition.

Organization and Prioritization: Look for a clear structure and logical flow in each debater’s presentation. The debater should prioritize major points and present them in a coherent sequence.

Clarity of Expression: Focus on the debater's ability to convey their arguments clearly and effectively. Well-articulated points are more persuasive and easier to follow.

Originality and Creativity: Consider the uniqueness of the arguments. Innovative or fresh perspectives on the topic can enhance the depth and impact of the debate.

Overall Impact: Reflect on the overall persuasiveness and memorability of each debater's case. Evaluate which debater presented a more convincing and compelling argument overall."""
        times_runned = 0
        # Initialize the arguments for both the Pro and Con side debaters
        pro_arguments = []
        con_arguments = []

        start_time = time.time()
        # Play the Debate Battle game for the total number of rounds
        for round_number in range(1, total_rounds + 1):
            while True:
                if(times_runned>10):
                    print("Failed to generate all sarguement")
                    break

                pro_argument_text = api_call_openai("gpt-4o-mini-2024-07-18", prompt_to_pro(pro_arguments, con_arguments, topic, round_number, total_rounds, pro_prompt), temp, max_tokens).replace('\n', '').replace('\r', '')

                if pro_argument_text.startswith("Argument:") or pro_argument_text.startswith("***Argument:***") or pro_argument_text.startswith("**Argument:**"):
                    argument = pro_argument_text.strip().split("Argument:")[-1].strip().strip('"').strip('*').replace('\n\n', ' ').replace('  ', ' ').strip()
                    pro_arguments.append(argument)
                    break

                else:
                    print(pro_argument_text)
                    print("Argument did not start with 'Argument:', retrying...")
                    times_runned+=1

            while True:
                if(times_runned>10):
                    print("Failed to generate all sarguement")
                    break

                con_argument_text = api_call_openai("gpt-4o-mini-2024-07-18", prompt_to_con(pro_arguments, con_arguments, topic, round_number, total_rounds, con_prompt), temp, max_tokens).replace('\n', '').replace('\r', '')


                if con_argument_text.startswith("Argument:") or con_argument_text.startswith("***Argument:***") or con_argument_text.startswith("**Argument:**"):
                    argument = con_argument_text.strip().split("Argument:")[-1].strip().strip('"').strip('*').replace('\n\n', ' ').replace('  ', ' ').strip()
                    con_arguments.append(argument)
                    break
                else:
                    print(con_argument_text)
                    print("Argument did not start with 'Argument:', retrying...")
                    times_runned+=1

        # print("Pro Arguments: ", pro_arguments)
        # print("Con Arguments: ", con_arguments)

        times_runned = 0
        while True:
            if(times_runned>10):
                print("Failed to generate winner")
                return None

            # Judge the Debate Battle by getting the json response from the judge
            judge_response = api_call_openai_json("gpt-4o-mini-2024-07-18", prompt_to_judge(topic, pro_arguments, con_arguments, criteria), temp, max_tokens)
            winner, reason = extract_winner(judge_response)
            # Check if both reason and winner are not none
            if winner is not None and reason is not None:
                break
            else:
                print(judge_response)
                print("Response did not contain the required keys, retrying...")
                times_runned+=1

        end_time = time.time()
        print(f"Time taken to complete the game: {end_time - start_time} seconds")
        results = [pro_prompt, con_prompt, topic, total_rounds, pro_arguments, con_arguments, winner, reason]
        return results

# Function to extract the winner and reason from the JSON response
def extract_winner(json_response):
    """
    Extracts the winner and reason from the JSON response if it meets all criteria.
    """
    # Define the required keys
    required_keys = ["winner", "reason"]

    # Check if all required keys are present and their values are non-zero or non-empty
    if all(key in json_response for key in required_keys) and all(json_response[key] for key in required_keys):
        # Extract the winner and reason
        winner = json_response["winner"].lower().strip().strip('"').strip("'").strip().strip("**").strip("*").strip().strip('"').strip()
        reason = json_response["reason"].strip().strip('"').strip("'").strip().strip("**").strip("*").strip().strip('"').strip()
        return winner, reason
    else:
        # If any key is missing or has an empty/zero value, return None
        return None, None

# Function for prompting the Pro side Debater
def prompt_to_pro(pro_arguments, con_arguments, topic, round_number, total_rounds, pro_prompt):
    """
    Parameters
    ----------
    pro_arguments: list
        List of arguments presented by the Pro side debater
    con_arguments: list
        List of arguments presented by the Con side debater
    topic: str
        The topic for the Debate Battle
    round_number: int
        The current round number
    total_rounds: int
        The total number of rounds in the Debate Battle
    pro_prompt: str
        The prompt for the Pro side debater
    con_prompt: str
        The prompt for the Con side debater
    
    Returns
    --------
    pro_argument: str
        The argument presented by the Pro side debater
    """

    # Determine the round type
    if round_number == 1:
        round_type = "Opening"
    elif round_number == total_rounds:
        round_type = "Closing"
    else:
        round_type = "Rebuttal"

    # Segregate arguments into "your arguments" and "opponent's arguments"
    your_arguments = ' '.join([f"({round_type}) {arg}" for arg in pro_arguments])
    opponent_arguments = ' '.join([f"({round_type}) {arg}" for arg in con_arguments])

    # Prompt the Pro side debater

    prompt_pro = f"{pro_prompt}"

    prompt_pro += f"\n\nYou are debating on the topic: '{topic}' and you are on the Pro side."

    prompt_pro += f"\n\nThis is the {round_type} round."
    
    prompt_pro+= f"""\n\nYou are expected to present your {round_type} statement.
    
    Your arguments so far are: \"{your_arguments}.\" Your opponent's arguments so far are: \"{opponent_arguments}.\" It's your turn to present your argument.
    
    Make sure that the response starts with 'Argument: ' followed by your argument."""

    return prompt_pro

# Function for prompting the Con side Debater
def prompt_to_con(pro_arguments, con_arguments, topic, round_number, total_rounds, con_prompt):
    """
    Parameters
    ----------
    pro_arguments: list
        List of arguments presented by the Pro side debater
    con_arguments: list
        List of arguments presented by the Con side debater
    topic: str
        The topic for the Debate Battle
    round_number: int
        The current round number
    total_rounds: int
        The total number of rounds in the Debate Battle
    con_prompt: str
        The prompt for the Con side debater
    
    Returns
    --------
    con_argument: str
        The argument presented by the Con side debater
    """ 
    
    # Determine the round type
    if round_number == 1:
        round_type = "Opening"
    elif round_number == total_rounds:
        round_type = "Closing"
    else:
        round_type = "Rebuttal"

    # Segregate arguments into "your arguments" and "opponent's arguments"
    your_arguments = ' '.join([f"({round_type}) {arg}" for arg in con_arguments])
    opponent_arguments = ' '.join([f"({round_type}) {arg}" for arg in pro_arguments])

    # Prompt the Con side debater
    prompt_con = f"{con_prompt}"

    prompt_con += f"\n\nYou are debating on the topic: '{topic}' and you are on the Con side."

    prompt_con += f"\n\nThis is the {round_type} round."

    prompt_con += f"""\n\nYou are expected to present your {round_type} statement.

    Your arguments so far are: \"{your_arguments}.\" Your opponent's arguments so far are: \"{opponent_arguments}.\" It's your turn to present your argument.

    Make sure that the response starts with 'Argument: ' followed by your argument."""

    return prompt_con


def prompt_to_judge(topic, pro_arguments, con_arguments, criteria=None):
    """
    Parameters
    ----------
    topic: str
        The topic for the Debate Battle
    pro_arguments: list
        List of arguments presented by the Pro side debater
    con_arguments: list
        List of arguments presented by the Con side debater 
    criteria: str
        The criteria based on which the judge has to decide the winner
    
    Returns
    --------
    prompt_judge: str
    """
    
    # Interleave Pro and Con arguments
    rounds = ""
    for round_no, (pro_arg, con_arg) in enumerate(zip(pro_arguments, con_arguments), start=1):
        rounds += (
            f"Round {round_no}:\n"
            f"Pro: {pro_arg}\n"
            f"Con: {con_arg}\n\n"
        )

    # Prompt the judge with criteria
    if criteria is not None:
        prompt_judge = (
            f"You are a judge in a debate on the topic: '{topic}' between the Pro and Con sides.\n\n"
            f"{rounds}"
            f"Based on the arguments presented and the following criteria: {criteria}, please decide the winner of the debate.\n\n"
            "Note that there cannot be a draw, you have to decide a winner which can be either the Pro side debater or the Con side debater, not both.\n\n"
            "Here is the format you need to respond in (in JSON format):\n\n"
            "{\n"
            '    "reason": "[your reason here]",\n'
            '    "winner": "[pro/con]"\n'
            "}\n\n"
            "Ensure that all the fields are provided and have non-zero or non-empty values."
        )

    # Prompt the judge without criteria
    else:
        prompt_judge = (
            f"You are a judge in a debate on the topic: '{topic}' between the Pro and Con sides.\n\n"
            f"{rounds}"
            "Based on the arguments presented, please decide the winner of the debate.\n\n"
            "Note that there cannot be a draw, you have to decide a winner which can be either the Pro side debater or the Con side debater, not both.\n\n"
            "Here is the format you need to respond in (in JSON format):\n\n"
            "{\n"
            '    "reason": "[your reason here]",\n'
            '    "winner": "[Pro/Con]"\n'
            "}\n\n"
            "Ensure that all the fields are provided and have non-zero or non-empty values."
        )

    return prompt_judge



# finding the ELO rating of the prompts after the battle
def elo_rating(pro_prompt, con_prompt, topic, total_rounds, temp, max_tokens, all_topics, game_over = False, results = None, custom_dict = None):
    
    """
    Parameters
    ----------
    
    pro_prompt: str
        The prompt for the Pro side debater
    con_prompt: str
        The prompt for the Con side debater
    topic: str
        The topic for the Debate Battle
    total_rounds: int
        The total number of rounds in the Debate Battle

    Returns
    --------
   
    """
    
    global elo_ratings_dict
    
    # Get the results of the game
    if game_over == False:
        results = game(pro_prompt, con_prompt, topic, total_rounds, temp, max_tokens)
        if results == None:
            print("Failed to generate results")
            winner = "draw"
            results = [pro_prompt, con_prompt, topic, total_rounds, [], [], winner, None]
            return results, winner

    if game_over == True:
        if results == None:
            print("Failed to generate results")
            # Don't update the ELO ratings if the results are not generated
            winner = "draw"
            results = [pro_prompt, con_prompt, topic, total_rounds, [], [], winner, None]
            return results, winner
                
    # Get the winner of the game
    winner = results[6]
    if (custom_dict == None):
        #If it is a draw, then calculate the ELO ratings of the prompts and print the updated ELO ratings
        if winner == "draw":
            # Find the ELO scores of the winner and loser
            winner_rating = elo_ratings_dict[pro_prompt]
            loser_rating = elo_ratings_dict[con_prompt]
            # Calculate the expected score of the winner and loser
            expected_winner = 1 / (1 + 10 ** ((loser_rating - winner_rating) / 400))
            expected_loser = 1 / (1 + 10 ** ((winner_rating - loser_rating) / 400))
            # Update the ELO ratings of the winner and loser
            winner_rating = winner_rating + 32 * (0.5 - expected_winner)
            loser_rating = loser_rating + 32 * (0.5 - expected_loser)
            # Update the ELO ratings in the dictionary
            elo_ratings_dict[pro_prompt] = winner_rating
            elo_ratings_dict[con_prompt] = loser_rating
            # Print the updated ELO ratings
            # print("Updated ELO Ratings:")
            # print(f"{pro_prompt}: {winner_rating}")
            # print(f"{con_prompt}: {loser_rating}")

        #If the winner is the Pro side debater, then calculate the ELO ratings of the prompts and print the updated ELO ratings

        elif winner == "pro":
            # Find the ELO scores of the winner and loser
            winner_rating = elo_ratings_dict[pro_prompt]
            loser_rating = elo_ratings_dict[con_prompt]
            # Calculate the expected score of the winner and loser
            expected_winner = 1 / (1 + 10 ** ((loser_rating - winner_rating) / 400))
            expected_loser = 1 / (1 + 10 ** ((winner_rating - loser_rating) / 400))
            # Update the ELO ratings of the winner and loser
            winner_rating = winner_rating + 32 * (1 - expected_winner)
            loser_rating = loser_rating + 32 * (0 - expected_loser)
            # Update the ELO ratings in the dictionary
            elo_ratings_dict[pro_prompt] = winner_rating
            elo_ratings_dict[con_prompt] = loser_rating
            # # Print the updated ELO ratings
            # print("Updated ELO Ratings:")
            # print(f"{pro_prompt}: {winner_rating}")
            # print(f"{con_prompt}: {loser_rating}")

        #If the winner is the Con side debater, then calculate the ELO ratings of the prompts and print the updated ELO ratings

        elif winner == "con":
            # Find the ELO scores of the winner and loser
            winner_rating = elo_ratings_dict[con_prompt]
            loser_rating = elo_ratings_dict[pro_prompt]
            # Calculate the expected score of the winner and loser
            expected_winner = 1 / (1 + 10 ** ((loser_rating - winner_rating) / 400))
            expected_loser = 1 / (1 + 10 ** ((winner_rating - loser_rating) / 400))
            # Update the ELO ratings of the winner and loser
            winner_rating = winner_rating + 32 * (1 - expected_winner)
            loser_rating = loser_rating + 32 * (0 - expected_loser)
            # Update the ELO ratings in the dictionary
            elo_ratings_dict[con_prompt] = winner_rating
            elo_ratings_dict[pro_prompt] = loser_rating
            # # Print the updated ELO ratings
            # print("Updated ELO Ratings:")
            # print(f"{con_prompt}: {winner_rating}")
            # print(f"{pro_prompt}: {loser_rating}")
    
    elif(custom_dict!=None):
        #If it is a draw, then calculate the ELO ratings of the prompts and print the updated ELO ratings
        if winner == "draw":
            # Find the ELO scores of the winner and loser
            winner_rating = custom_dict[pro_prompt]
            loser_rating = custom_dict[con_prompt]
            # Calculate the expected score of the winner and loser
            expected_winner = 1 / (1 + 10 ** ((loser_rating - winner_rating) / 400))
            expected_loser = 1 / (1 + 10 ** ((winner_rating - loser_rating) / 400))
            # Update the ELO ratings of the winner and loser
            winner_rating = winner_rating + 32 * (0.5 - expected_winner)
            loser_rating = loser_rating + 32 * (0.5 - expected_loser)
            # Update the ELO ratings in the dictionary
            custom_dict[pro_prompt] = winner_rating
            custom_dict[con_prompt] = loser_rating
            # Print the updated ELO ratings
            # print("Updated ELO Ratings:")
            # print(f"{pro_prompt}: {winner_rating}")
            # print(f"{con_prompt}: {loser_rating}")

        #If the winner is the Pro side debater, then calculate the ELO ratings of the prompts and print the updated ELO ratings

        elif winner == "pro":
            # Find the ELO scores of the winner and loser
            winner_rating = custom_dict[pro_prompt]
            loser_rating = custom_dict[con_prompt]
            # Calculate the expected score of the winner and loser
            expected_winner = 1 / (1 + 10 ** ((loser_rating - winner_rating) / 400))
            expected_loser = 1 / (1 + 10 ** ((winner_rating - loser_rating) / 400))
            # Update the ELO ratings of the winner and loser
            winner_rating = winner_rating + 32 * (1 - expected_winner)
            loser_rating = loser_rating + 32 * (0 - expected_loser)
            # Update the ELO ratings in the dictionary
            custom_dict[pro_prompt] = winner_rating
            custom_dict[con_prompt] = loser_rating
            # # Print the updated ELO ratings
            # print("Updated ELO Ratings:")
            # print(f"{pro_prompt}: {winner_rating}")
            # print(f"{con_prompt}: {loser_rating}")

        #If the winner is the Con side debater, then calculate the ELO ratings of the prompts and print the updated ELO ratings

        elif winner == "con":
            # Find the ELO scores of the winner and loser
            winner_rating = custom_dict[con_prompt]
            loser_rating = custom_dict[pro_prompt]
            # Calculate the expected score of the winner and loser
            expected_winner = 1 / (1 + 10 ** ((loser_rating - winner_rating) / 400))
            expected_loser = 1 / (1 + 10 ** ((winner_rating - loser_rating) / 400))
            # Update the ELO ratings of the winner and loser
            winner_rating = winner_rating + 32 * (1 - expected_winner)
            loser_rating = loser_rating + 32 * (0 - expected_loser)
            # Update the ELO ratings in the dictionary
            custom_dict[con_prompt] = winner_rating
            custom_dict[pro_prompt] = loser_rating
            # Print the updated ELO ratings
            # print("Updated ELO Ratings:")
            # print(f"{con_prompt}: {winner_rating}")
            # print(f"{pro_prompt}: {loser_rating}")
    
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
def league(pro_prompts, con_prompts, topics, total_rounds, k, league_no, temp, max_tokens, human = False, human_file = None):

    """
    Parameters
    ----------
    pro_prompts: list
        List of prompts for the Pro side debater
    con_prompts: list
        List of prompts for the Con side debater
    topics: list
        List of topics for the Debate Battle
    total_rounds: int
        The total number of rounds in the Debate Battle
    k: int
        The number of top and bottom prompts to be returned after the league
    league_no: int
        The league number for which the league is to be played
    human: bool
        True if the prompts are generated by humans, False if the prompts are not generated by LLM
    human_file: str
        The name of the file where the human prompts are stored
    final: bool
        True if it is the final league, False otherwise

    Returns
    --------
    top_k_prompts: list
        List of top k prompts after the league
    bottom_k_prompts: list
        List of bottom k prompts after the league
    """

    global elo_ratings_dict
    global league_dict
    global top_bottom_prompts_dict_pro
    global top_bottom_prompts_dict_con
    global human_prompts_used_dict_pro
    global human_prompts_used_dict_con
    global origin_league_prompts_dict_pro
    global origin_league_prompts_dict_con


    for pro_prompt in pro_prompts:
        origin_league_prompts_dict_pro[pro_prompt] = league_no

    for con_prompt in con_prompts:
        origin_league_prompts_dict_con[con_prompt] = league_no

    #Check if the league is already palyed or not
    if check_league(league_no):

        print(f"League {league_no} is already over!")
        # Get the top and bottom k prompts for pro and con side along with their final elo scores from the league_results.json file for that league_no and also put it in the top_bottom_prompts_dict_pro and top_bottom_prompts_dict_con
        with open('league_results.json', 'r') as json_file:
            data = json.load(json_file)
        
        # Collect all prompts with their final ELO scores
        all_pro_prompts = {}
        all_con_prompts = {}
        for battle_no, battle_dict in data[str(league_no)].items():
            pro_prompt = battle_dict['pro-prompt']['prompt']
            con_prompt = battle_dict['con-prompt']['prompt']
            pro_final_elo = battle_dict['pro-prompt']['final_elo']
            con_final_elo = battle_dict['con-prompt']['final_elo']
            all_pro_prompts[pro_prompt] = pro_final_elo
            all_con_prompts[con_prompt] = con_final_elo

        # Store all the prompts in the human_prompts_used_dict_pro and con if human prompts are used
        if human == True and human_file != None:
            human_prompts_used_dict_pro[league_no] = list(all_pro_prompts.keys())
            human_prompts_used_dict_con[league_no] = list(all_con_prompts.keys())

        # Store all the elo scores in the elo_ratings_dict with the key has prompt and value as elo score
        for prompt, elo in all_pro_prompts.items():
            elo_ratings_dict[prompt] = elo

        for prompt, elo in all_con_prompts.items():
            elo_ratings_dict[prompt] = elo

        # Sort the prompts based on their ELO scores
        sorted_prompts_pro = sorted(all_pro_prompts.items(), key=lambda x: x[1], reverse=True)
        sorted_prompts_con = sorted(all_con_prompts.items(), key=lambda x: x[1], reverse=True)

        # Get the top k and bottom k prompts for the Pro side debater
        top_k_prompts_pro = sorted_prompts_pro[:k]
        bottom_k_prompts_pro = sorted_prompts_pro[-k:]

        # Get the top k and bottom k prompts for the Con side debater
        top_k_prompts_con = sorted_prompts_con[:k]
        bottom_k_prompts_con = sorted_prompts_con[-k:]

        # Store the top and bottom k prompts for the Pro side debater in the dictionary for the league and also rank them by 1,2,3..,k
        top_bottom_prompts_dict_pro[league_no] = {
            'top_k_prompts': {i+1: {'prompt': prompt, 'elo': elo} for i, (prompt, elo) in enumerate(top_k_prompts_pro)},
            'bottom_k_prompts': {i+1: {'prompt': prompt, 'elo': elo} for i, (prompt, elo) in enumerate(bottom_k_prompts_pro)}
        }

        # Store the top and bottom k prompts for the Con side debater in the dictionary for the league and also rank them by 1,2,3..,k
        top_bottom_prompts_dict_con[league_no] = {
            'top_k_prompts': {i+1: {'prompt': prompt, 'elo': elo} for i, (prompt, elo) in enumerate(top_k_prompts_con)},
            'bottom_k_prompts': {i+1: {'prompt': prompt, 'elo': elo} for i, (prompt, elo) in enumerate(bottom_k_prompts_con)}
        }

        # Dump the top and bottom k prompts for the Pro side and Con side debators in a json file 
        with open('top_bottom_prompts_dict_pro.json', 'w') as json_file:
            json.dump(top_bottom_prompts_dict_pro, json_file, indent=4)
    
        with open('top_bottom_prompts_dict_con.json', 'w') as json_file:
            json.dump(top_bottom_prompts_dict_con, json_file, indent=4)

        return top_k_prompts_pro, bottom_k_prompts_pro, top_k_prompts_con, bottom_k_prompts_con
    
    temp_league_dict = {}
    temp_league_dict[league_no] = {}

    for prompt in pro_prompts:
        elo_ratings_dict[prompt] = 1200

    for prompt in con_prompts:
        elo_ratings_dict[prompt] = 1200

    battle_no = 0

    round_robin_list = [(i, j) for i in range(len(pro_prompts)) for j in range(len(con_prompts))]
    random.shuffle(round_robin_list)

    # To store the results for ELO updates after parallel processing in the correct order
    results_list = [None] * len(round_robin_list)  # Preallocate list to ensure correct order
    
    start_time = time.time()

    # Use ThreadPoolExecutor to run games in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=7) as executor:
        future_to_battle = {
            executor.submit(game, pro_prompts[i], con_prompts[j], random.choice(topics), total_rounds, temp, max_tokens): (index, i, j)
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

        # Use the topic from the result (ensures correct topic is used)
        topic = result[2]
        # Get initial ELO ratings
        initial_elo_rating_1 = elo_ratings_dict[pro_prompts[i]]
        initial_elo_rating_2 = elo_ratings_dict[con_prompts[j]]
        
        # Update ELO ratings using elo_rating function
        result, winner = elo_rating(pro_prompts[i], con_prompts[j], topic, total_rounds, temp, max_tokens, topics, game_over=True, results=result, custom_dict=None)
        
        # Get final ELO ratings
        final_elo_rating1 = elo_ratings_dict[pro_prompts[i]]
        final_elo_rating2 = elo_ratings_dict[con_prompts[j]]

        # Increment battle number and prepare battle dictionary
        battle_no += 1
        battle_dict = {
            'topic': topic,
            'total_rounds': total_rounds,
            'pro-prompt': {
                'prompt': pro_prompts[i],
                'initial_elo': initial_elo_rating_1,
                'final_elo': final_elo_rating1
            },
            'con-prompt': {
                'prompt': con_prompts[j],
                'initial_elo': initial_elo_rating_2,
                'final_elo': final_elo_rating2
            },
            'pro_arguments': result[4],
            'con_arguments': result[5],
            'winner_role': winner,
            'winner_prompt': pro_prompts[i] if winner == 'pro' else 'draw' if winner == 'draw' else con_prompts[j],
            'reason': result[7]
        }

        temp_league_dict[league_no][battle_no] = battle_dict

    end_time = time.time()
    print(f"Time taken to complete the league: {end_time - start_time} seconds")
    
    # Store the prompts used in the human_prompts_used_dict if human prompts are used
    if human == True and human_file != None:
        human_prompts_used_dict_pro[league_no] = pro_prompts
        human_prompts_used_dict_con[league_no] = con_prompts

    temp_elo_dict_pro={}
    for i in range(len(pro_prompts)):
        temp_elo_dict_pro[pro_prompts[i]]= elo_ratings_dict[pro_prompts[i]]
    

    sorted_prompts_pro = sorted(temp_elo_dict_pro.items(), key=lambda x: x[1], reverse=True)

    # Get the top k and bottom k prompts for the Pro side debater
    top_k_prompts_pro = sorted_prompts_pro[:k]
    bottom_k_prompts_pro = sorted_prompts_pro[-k:]

    temp_elo_dict_con={}
    for i in range(len(con_prompts)):
        temp_elo_dict_con[con_prompts[i]]= elo_ratings_dict[con_prompts[i]]

    sorted_prompts_con = sorted(temp_elo_dict_con.items(), key=lambda x: x[1], reverse=True)

    # Get the top k and bottom k prompts for the Con side debater
    top_k_prompts_con = sorted_prompts_con[:k]
    bottom_k_prompts_con = sorted_prompts_con[-k:]

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
    

    # Save the top and bottom k prompts for the Pro side debater in the dictionary for the league and also rank them by 1,2,3..,k
    top_bottom_prompts_dict_pro[league_no] = {
        'top_k_prompts': {i+1: {'prompt': prompt, 'elo': elo} for i, (prompt, elo) in enumerate(top_k_prompts_pro)},
        'bottom_k_prompts': {i+1: {'prompt': prompt, 'elo': elo} for i, (prompt, elo) in enumerate(bottom_k_prompts_pro)}
    }

    # Save the top and bottom k prompts for the Con side debater in the dictionary for the league and also rank them by 1,2,3..,k
    top_bottom_prompts_dict_con[league_no] = {
        'top_k_prompts': {i+1: {'prompt': prompt, 'elo': elo} for i, (prompt, elo) in enumerate(top_k_prompts_con)},
        'bottom_k_prompts': {i+1: {'prompt': prompt, 'elo': elo} for i, (prompt, elo) in enumerate(bottom_k_prompts_con)}
    }

    # Dump the top and bottom k prompts for the Pro side and Con side debators in a json file 
    with open('top_bottom_prompts_dict_pro.json', 'w') as json_file:
        json.dump(top_bottom_prompts_dict_pro, json_file, indent=4)
    
    with open('top_bottom_prompts_dict_con.json', 'w') as json_file:
        json.dump(top_bottom_prompts_dict_con, json_file, indent=4)

    # # Load the dictionary from the pickle file
    # with open('league_results.pkl', 'rb') as pickle_file:
    #     league_dict = pickle.load(pickle_file)

    # return top and bottom k prompts
    return top_k_prompts_pro, bottom_k_prompts_pro, top_k_prompts_con, bottom_k_prompts_con

# Play the final league between the league_no top k prompts and league_no -1 top k prompts to get the final top k prompts for each role
def final_league(pro_prompts, con_prompts, topics, total_rounds, league_no, k, temp, max_tokens):

    global top_bottom_prompts_dict_con_across_league
    global top_bottom_prompts_dict_pro_across_league
    global origin_league_prompts_dict_pro
    global origin_league_prompts_dict_con

    final_elo_dict = {}

    #Check if the final_league_results{current_league}.json file exists
    if os.path.exists(f'final_league_results{league_no}.json'):
        print(f"Final League {league_no} is already over!")
        # Get the top and bottom k prompts for pro and con side along with their final elo scores from the final_league_results{league_no}.json file for that league_no
        with open(f'final_league_results{league_no}.json', 'r') as json_file:
            data = json.load(json_file)
        # Collect all prompts with their final ELO scores
        all_pro_prompts = {}
        all_con_prompts = {}
        for battle_no, battle_dict in data[str(league_no)].items():
            pro_prompt = battle_dict['pro-prompt']['prompt']
            con_prompt = battle_dict['con-prompt']['prompt']
            pro_final_elo = battle_dict['pro-prompt']['final_elo']
            con_final_elo = battle_dict['con-prompt']['final_elo']
            all_pro_prompts[pro_prompt] = pro_final_elo
            all_con_prompts[con_prompt] = con_final_elo
        
        for prompt, elo in all_pro_prompts.items():
            final_elo_dict[prompt] = elo

        for prompt, elo in all_con_prompts.items():
            final_elo_dict[prompt] = elo

        # Sort the prompts based on their ELO scores
        sorted_prompts_pro = sorted(all_pro_prompts.items(), key=lambda x: x[1], reverse=True)
        sorted_prompts_con = sorted(all_con_prompts.items(), key=lambda x: x[1], reverse=True)

        # Get the top k and bottom k prompts for the Pro side debater
        top_k_prompts_pro = sorted_prompts_pro[:k]
        bottom_k_prompts_pro = sorted_prompts_pro[-k:]

        # Get the top k and bottom k prompts for the Con side debater
        top_k_prompts_con = sorted_prompts_con[:k]
        bottom_k_prompts_con = sorted_prompts_con[-k:]


        # Store the top_k_prompts pro, con and bottom_k_prompts pro, con in the top_bottom_prompts_dict_con_across_league and top_bottom_prompts_dict_pro_across_league
        top_bottom_prompts_dict_pro_across_league[league_no] = {
            'top_k_prompts': {i+1: {'prompt': prompt, 'elo': elo, 'origin': origin_league_prompts_dict_pro[prompt]} for i, (prompt, elo) in enumerate(top_k_prompts_pro)},
            'bottom_k_prompts': {i+1: {'prompt': prompt, 'elo': elo, 'origin': origin_league_prompts_dict_pro[prompt]} for i, (prompt, elo) in enumerate(bottom_k_prompts_pro)}
        }

        top_bottom_prompts_dict_con_across_league[league_no] = {
            'top_k_prompts': {i+1: {'prompt': prompt, 'elo': elo, 'origin': origin_league_prompts_dict_con[prompt]} for i, (prompt, elo) in enumerate(top_k_prompts_con)},
            'bottom_k_prompts': {i+1: {'prompt': prompt, 'elo': elo, 'origin': origin_league_prompts_dict_con[prompt]} for i, (prompt, elo) in enumerate(bottom_k_prompts_con)}
        }

        # Dump the top and bottom k prompts for the Pro side and Con side debators in a json file
        with open('top_bottom_prompts_dict_pro_across_league.json', 'w') as json_file:
            json.dump(top_bottom_prompts_dict_pro_across_league, json_file, indent=4)
    
        with open('top_bottom_prompts_dict_con_across_league.json', 'w') as json_file:
            json.dump(top_bottom_prompts_dict_con_across_league, json_file, indent=4)

        return top_k_prompts_pro, bottom_k_prompts_pro, top_k_prompts_con, bottom_k_prompts_con, final_elo_dict
    
    # Initialise the elo ratings of each prompt to 1200
    for prompt in pro_prompts:
        final_elo_dict[prompt] = 1200
    
    for prompt in con_prompts:
        final_elo_dict[prompt] = 1200

    temp_league_dict = {}
    temp_league_dict[league_no] = {}

    battle_no = 0

    round_robin_list = [(i, j) for i in range(len(pro_prompts)) for j in range(len(con_prompts))]
    random.shuffle(round_robin_list)

    # To store the results for ELO updates after parallel processing in the correct order
    results_list = [None] * len(round_robin_list)  # Preallocate list to ensure correct order
    
    start_time = time.time()

    # Use ThreadPoolExecutor to run games in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=7) as executor:
        future_to_battle = {
            executor.submit(game, pro_prompts[i], con_prompts[j], random.choice(topics), total_rounds, temp, max_tokens): (index, i, j)
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

        # Use the topic from the result (ensures correct topic is used)
        topic = result[2]
        # Get initial ELO ratings
        initial_elo_rating_1 = final_elo_dict[pro_prompts[i]]
        initial_elo_rating_2 = final_elo_dict[con_prompts[j]]
        
        # Update ELO ratings using elo_rating function
        result, winner = elo_rating(pro_prompts[i], con_prompts[j], topic, total_rounds, temp, max_tokens, topics, game_over=True, results=result, custom_dict= final_elo_dict)
        
        # Get final ELO ratings
        final_elo_rating1 = final_elo_dict[pro_prompts[i]]
        final_elo_rating2 = final_elo_dict[con_prompts[j]]

        # Increment battle number and prepare battle dictionary
        battle_no += 1
        battle_dict = {
            'topic': topic,
            'total_rounds': total_rounds,
            'pro-prompt': {
                'prompt': pro_prompts[i],
                'initial_elo': initial_elo_rating_1,
                'final_elo': final_elo_rating1
            },
            'con-prompt': {
                'prompt': con_prompts[j],
                'initial_elo': initial_elo_rating_2,
                'final_elo': final_elo_rating2
            },
            'pro_arguments': result[4],
            'con_arguments': result[5],
            'winner_role': winner,
            'winner_prompt': pro_prompts[i] if winner == 'pro' else 'draw' if winner == 'draw' else con_prompts[j],
            'reason': result[7]
        }

        temp_league_dict[league_no][battle_no] = battle_dict

    end_time = time.time()
    print(f"Time taken to complete the league: {end_time - start_time} seconds")
    
    temp_elo_dict_pro={}

    for i in range(len(pro_prompts)):
        temp_elo_dict_pro[pro_prompts[i]]= final_elo_dict[pro_prompts[i]]
    

    sorted_prompts_pro = sorted(temp_elo_dict_pro.items(), key=lambda x: x[1], reverse=True)

    # Get the top k and bottom k prompts for the Pro side debater
    top_k_prompts_pro = sorted_prompts_pro[:k]
    bottom_k_prompts_pro = sorted_prompts_pro[-k:]

    temp_elo_dict_con={}
    for i in range(len(con_prompts)):
        temp_elo_dict_con[con_prompts[i]]= final_elo_dict[con_prompts[i]]

    sorted_prompts_con = sorted(temp_elo_dict_con.items(), key=lambda x: x[1], reverse=True)

    # Get the top k and bottom k prompts for the Con side debater
    top_k_prompts_con = sorted_prompts_con[:k]
    bottom_k_prompts_con = sorted_prompts_con[-k:]

    # Check if the JSON file exists
    file_path = f'final_league_results{league_no}.json'
    
    data = {}

    # Update the data with the current league_dict
    data.update(temp_league_dict)

    # Write the updated data back to the JSON file
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)


    # Store the top_k_prompts pro, con and bottom_k_prompts pro, con in the top_bottom_prompts_dict_con_across_league and top_bottom_prompts_dict_pro_across_league
    top_bottom_prompts_dict_pro_across_league[league_no] = {
            'top_k_prompts': {i+1: {'prompt': prompt, 'elo': elo, 'origin': origin_league_prompts_dict_pro[prompt]} for i, (prompt, elo) in enumerate(top_k_prompts_pro)},
            'bottom_k_prompts': {i+1: {'prompt': prompt, 'elo': elo, 'origin': origin_league_prompts_dict_pro[prompt]} for i, (prompt, elo) in enumerate(bottom_k_prompts_pro)}
    }

    top_bottom_prompts_dict_con_across_league[league_no] = {
            'top_k_prompts': {i+1: {'prompt': prompt, 'elo': elo, 'origin': origin_league_prompts_dict_con[prompt]} for i, (prompt, elo) in enumerate(top_k_prompts_con)},
            'bottom_k_prompts': {i+1: {'prompt': prompt, 'elo': elo, 'origin': origin_league_prompts_dict_con[prompt]} for i, (prompt, elo) in enumerate(bottom_k_prompts_con)}
    }

    # Dump the top and bottom k prompts for the Pro side and Con side debators in a json file
    with open('top_bottom_prompts_dict_pro_across_league.json', 'w') as json_file:
        json.dump(top_bottom_prompts_dict_pro_across_league, json_file, indent=4)
    
    with open('top_bottom_prompts_dict_con_across_league.json', 'w') as json_file:
        json.dump(top_bottom_prompts_dict_con_across_league, json_file, indent=4)

    return top_k_prompts_pro, bottom_k_prompts_pro, top_k_prompts_con, bottom_k_prompts_con, final_elo_dict


def compare_elo_average(top_pro_prompts_league, top_con_prompts_league , final_elo_dict, league_no, k, consecutive_fails):

    # Get the elo ratings average of top_pro_prompts_league and top_con_prompts_league using the final_elo_ratings_dict
    pro_elo_average_league = 0
    con_elo_average_league = 0

    for prompt in top_pro_prompts_league:
        pro_elo_average_league += final_elo_dict[prompt]
    pro_elo_average_league = pro_elo_average_league/len(top_pro_prompts_league)

    for prompt in top_con_prompts_league:
        con_elo_average_league += final_elo_dict[prompt]
    con_elo_average_league = con_elo_average_league/len(top_con_prompts_league)

    # Get the top k pro prompts and top k con prompts using top_bottom_prompts_dict_con_across_league.json and top_bottom_prompts_dict_pro_across_league.json of league number = league_no -1
    with open('top_bottom_prompts_dict_pro_across_league.json', 'r') as json_file:
        top_bottom_prompts_dict_pro_across_final = json.load(json_file)
    
    with open('top_bottom_prompts_dict_con_across_league.json', 'r') as json_file:
        top_bottom_prompts_dict_con_across_final = json.load(json_file)
    
    top_pro_prompts_final = top_bottom_prompts_dict_pro_across_final[str(league_no-1)]['top_k_prompts']
    # Extract the prompt from the dictionary and create a list of all the prompts
    top_pro_prompts_final = [top_pro_prompts_final[str(i)]['prompt'] for i in range(1, k+1)]
    
    top_con_prompts_final = top_bottom_prompts_dict_con_across_final[str(league_no-1)]['top_k_prompts']
    # Extract the prompt from the dictionary and create a list of all the prompts
    top_con_prompts_final = [top_con_prompts_final[str(i)]['prompt'] for i in range(1, k+1)]

    pro_elo_average_final = 0
    con_elo_average_final = 0

    for prompt in top_pro_prompts_final:
        pro_elo_average_final += final_elo_dict[prompt]
    pro_elo_average_final = pro_elo_average_final/len(top_pro_prompts_final)

    for prompt in top_con_prompts_final:
        con_elo_average_final += final_elo_dict[prompt]
    con_elo_average_final = con_elo_average_final/len(top_con_prompts_final)

    if (pro_elo_average_final > pro_elo_average_league or con_elo_average_final > con_elo_average_league):
        consecutive_fails+=1
        print("consecutive_fails: ", consecutive_fails)
        
    else:
        consecutive_fails = 0
        print("consecutive_fails: ", consecutive_fails)
    
    return consecutive_fails

def proof_of_concept(league1, league2, topics, total_rounds, k , temp, max_tokens):

    global top_bottom_prompts_dict_pro_across_league
    global top_bottom_prompts_dict_con_across_league
    global top_bottom_prompts_dict_pro
    global top_bottom_prompts_dict_con
    global top_bottom_prompts_dict_pro_poc
    global top_bottom_prompts_dict_con_poc
    global origin_league_prompts_dict_pro
    global origin_league_prompts_dict_con 


    league_no = 1
    
    # Select 10 random topics from the topics list
    topics = random.sample(topics, 100)

    poc_elo_dict = {}

    if league1 !=1:
        # Get pro_prompts1 by getting top k prompts from the league1 using top_bottom_prompts_dict_con_across_league

        top_pro_prompts_league1 = top_bottom_prompts_dict_pro_across_league[league1]['top_k_prompts']
        pro_prompts1 = [top_pro_prompts_league1[i]['prompt'] for i in range(1, k+1)]

        # Get con_prompts1 by getting top k prompts from the league1 using top_bottom_prompts_dict_con_across_league
        top_con_prompts_league1 = top_bottom_prompts_dict_con_across_league[league1]['top_k_prompts']
        con_prompts1 = [top_con_prompts_league1[i]['prompt'] for i in range(1, k+1)]

        # Get pro_prompts2 by getting top k prompts from the league2 using top_bottom_prompts_dict_con_across_league
        top_pro_prompts_league2 = top_bottom_prompts_dict_pro_across_league[league2]['top_k_prompts']
        pro_prompts2 = [top_pro_prompts_league2[i]['prompt'] for i in range(1, k+1)]

        # Get con_prompts2 by getting top k prompts from the league2 using top_bottom_prompts_dict_con_across_league
        top_con_prompts_league2 = top_bottom_prompts_dict_con_across_league[league2]['top_k_prompts']
        con_prompts2 = [top_con_prompts_league2[i]['prompt'] for i in range(1, k+1)]
    
    if league1==1:

        # get the pro_prompts1 and con_prompts1 from the top_bottom_prompts_dict_pro and top_bottom_prompts_dict_con for league1
        pro_prompts1 = [top_bottom_prompts_dict_pro[league1]['top_k_prompts'][i]['prompt'] for i in range(1, k+1)]
        con_prompts1 = [top_bottom_prompts_dict_con[league1]['top_k_prompts'][i]['prompt'] for i in range(1, k+1)]

        # get the pro_prompts2 and con_prompts2
        pro_prompts2 = [top_bottom_prompts_dict_pro_across_league[league2]['top_k_prompts'][i]['prompt'] for i in range(1, k+1)]
        con_prompts2 = [top_bottom_prompts_dict_con_across_league[league2]['top_k_prompts'][i]['prompt'] for i in range(1, k+1)]

    # Merge the pro_prompts1 and pro_prompts2 to get the pro_prompts
    pro_prompts = pro_prompts1 + pro_prompts2

    # Merge the con_prompts1 and con_prompts2 to get the con_prompts
    con_prompts = con_prompts1 + con_prompts2


    #Check if the file exists
    if os.path.exists(f'proof_of_concept_{league1}_{league2}.json'):
        print(f"Proof of Concept {league1} vs {league2} is already over!")
        # Get the top and bottom k prompts for pro and con side along with their final elo scores from the proof_of_concept_{league1}_{league2}.json' file for that league_no
        with open(f'proof_of_concept_{league1}_{league2}.json', 'r') as json_file:
            data = json.load(json_file)
        # Collect all prompts with their final ELO scores
        all_pro_prompts = {}
        all_con_prompts = {}
        for battle_no, battle_dict in data[str(league_no)].items():
            pro_prompt = battle_dict['pro-prompt']['prompt']
            con_prompt = battle_dict['con-prompt']['prompt']
            pro_final_elo = battle_dict['pro-prompt']['final_elo']
            con_final_elo = battle_dict['con-prompt']['final_elo']
            all_pro_prompts[pro_prompt] = pro_final_elo
            all_con_prompts[con_prompt] = con_final_elo
        
        for prompt, elo in all_pro_prompts.items():
            poc_elo_dict[prompt] = elo

        for prompt, elo in all_con_prompts.items():
            poc_elo_dict[prompt] = elo

        # Sort the prompts based on their ELO scores
        sorted_prompts_pro = sorted(all_pro_prompts.items(), key=lambda x: x[1], reverse=True)
        sorted_prompts_con = sorted(all_con_prompts.items(), key=lambda x: x[1], reverse=True)

        # Get the top k and bottom k prompts for the Pro side debater
        top_k_prompts_pro = sorted_prompts_pro[:k]
        bottom_k_prompts_pro = sorted_prompts_pro[-k:]

        # Get the top k and bottom k prompts for the Con side debater
        top_k_prompts_con = sorted_prompts_con[:k]
        bottom_k_prompts_con = sorted_prompts_con[-k:]

        # Store the top_k_prompts pro, con and bottom_k_prompts pro, con in the top_bottom_prompts_dict_con_poc and top_bottom_prompts_dict_pro_poc (Include origin as well)
        top_bottom_prompts_dict_pro_poc[f"{league1} vs {league2}"] = {
            'top_k_prompts': {i+1: {'prompt': prompt, 'elo': elo, 'origin': origin_league_prompts_dict_pro[prompt]} for i, (prompt, elo) in enumerate(top_k_prompts_pro)},
            'bottom_k_prompts': {i+1: {'prompt': prompt, 'elo': elo, 'origin': origin_league_prompts_dict_pro[prompt]} for i, (prompt, elo) in enumerate(bottom_k_prompts_pro)}
        }

        top_bottom_prompts_dict_con_poc[f"{league1} vs {league2}"] = {
            'top_k_prompts': {i+1: {'prompt': prompt, 'elo': elo, 'origin': origin_league_prompts_dict_con[prompt]} for i, (prompt, elo) in enumerate(top_k_prompts_con)},
            'bottom_k_prompts': {i+1: {'prompt': prompt, 'elo': elo, 'origin': origin_league_prompts_dict_con[prompt]} for i, (prompt, elo) in enumerate(bottom_k_prompts_con)}
        }

        # Dump the top and bottom k prompts for the Pro side and Con side debators in a json file
        with open('top_bottom_prompts_dict_pro_poc.json', 'w') as json_file:
            json.dump(top_bottom_prompts_dict_pro_poc, json_file, indent=4)
        
        with open('top_bottom_prompts_dict_con_poc.json', 'w') as json_file:
            json.dump(top_bottom_prompts_dict_con_poc, json_file, indent=4)


        compare_poc_elo_average(poc_elo_dict, pro_prompts1, con_prompts1, pro_prompts2, con_prompts2, league1, league2)

        return

    
    # Initialise the elo ratings of each prompt to 1200
    for prompt in pro_prompts:
        poc_elo_dict[prompt] = 1200
    
    for prompt in con_prompts:
        poc_elo_dict[prompt] = 1200

    temp_league_dict = {}
    temp_league_dict[league_no] = {}

    battle_no = 0

    round_robin_list = [(i, j) for i in range(len(pro_prompts)) for j in range(len(con_prompts))]
    random.shuffle(round_robin_list)

    # To store the results for ELO updates after parallel processing in the correct order
    results_list = [None] * len(round_robin_list)  # Preallocate list to ensure correct order
    
    start_time = time.time()

    # Use ThreadPoolExecutor to run games in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=7) as executor:
        future_to_battle = {
            executor.submit(game, pro_prompts[i], con_prompts[j], random.choice(topics), total_rounds, temp, max_tokens): (index, i, j)
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

        # Use the topic from the result (ensures correct topic is used)
        topic = result[2]
        # Get initial ELO ratings
        initial_elo_rating_1 = poc_elo_dict[pro_prompts[i]]
        initial_elo_rating_2 = poc_elo_dict[con_prompts[j]]
        
        # Update ELO ratings using elo_rating function
        result, winner = elo_rating(pro_prompts[i], con_prompts[j], topic, total_rounds, temp, max_tokens, topics, game_over=True, results=result, custom_dict= poc_elo_dict)
        
        # Get final ELO ratings
        final_elo_rating1 = poc_elo_dict[pro_prompts[i]]
        final_elo_rating2 = poc_elo_dict[con_prompts[j]]

        # Increment battle number and prepare battle dictionary
        battle_no += 1
        battle_dict = {
            'topic': topic,
            'total_rounds': total_rounds,
            'pro-prompt': {
                'prompt': pro_prompts[i],
                'initial_elo': initial_elo_rating_1,
                'final_elo': final_elo_rating1
            },
            'con-prompt': {
                'prompt': con_prompts[j],
                'initial_elo': initial_elo_rating_2,
                'final_elo': final_elo_rating2
            },
            'pro_arguments': result[4],
            'con_arguments': result[5],
            'winner_role': winner,
            'winner_prompt': pro_prompts[i] if winner == 'pro' else 'draw' if winner == 'draw' else con_prompts[j],
            'reason': result[7]
        }

        temp_league_dict[league_no][battle_no] = battle_dict

    end_time = time.time()
    print(f"Time taken to complete the league: {end_time - start_time} seconds")
    
    temp_elo_dict_pro={}

    for i in range(len(pro_prompts)):
        temp_elo_dict_pro[pro_prompts[i]]= poc_elo_dict[pro_prompts[i]]
    

    sorted_prompts_pro = sorted(temp_elo_dict_pro.items(), key=lambda x: x[1], reverse=True)

    # Get the top k and bottom k prompts for the Pro side debater
    top_k_prompts_pro = sorted_prompts_pro[:k]
    bottom_k_prompts_pro = sorted_prompts_pro[-k:]

    temp_elo_dict_con={}
    for i in range(len(con_prompts)):
        temp_elo_dict_con[con_prompts[i]]= poc_elo_dict[con_prompts[i]]

    sorted_prompts_con = sorted(temp_elo_dict_con.items(), key=lambda x: x[1], reverse=True)

    # Get the top k and bottom k prompts for the Con side debater
    top_k_prompts_con = sorted_prompts_con[:k]
    bottom_k_prompts_con = sorted_prompts_con[-k:]

    # Check if the JSON file exists
    file_path = f'proof_of_concept_{league1}_{league2}.json'
    
    data = {}

    # Update the data with the current league_dict
    data.update(temp_league_dict)

    # Write the updated data back to the JSON file
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    compare_poc_elo_average(poc_elo_dict, pro_prompts1, con_prompts1, pro_prompts2, con_prompts2, league1, league2)

    # Store the top_k_prompts pro, con and bottom_k_prompts pro, con in the top_bottom_prompts_dict_con_poc and top_bottom_prompts_dict_pro_poc (Include origin as well)
    top_bottom_prompts_dict_pro_poc[f"{league1} vs {league2}"] = {
        'top_k_prompts': {i+1: {'prompt': prompt, 'elo': elo, 'origin': origin_league_prompts_dict_pro[prompt]} for i, (prompt, elo) in enumerate(top_k_prompts_pro)},
        'bottom_k_prompts': {i+1: {'prompt': prompt, 'elo': elo, 'origin': origin_league_prompts_dict_pro[prompt]} for i, (prompt, elo) in enumerate(bottom_k_prompts_pro)}
    }

    top_bottom_prompts_dict_con_poc[f"{league1} vs {league2}"] = {
        'top_k_prompts': {i+1: {'prompt': prompt, 'elo': elo, 'origin': origin_league_prompts_dict_con[prompt]} for i, (prompt, elo) in enumerate(top_k_prompts_con)},
        'bottom_k_prompts': {i+1: {'prompt': prompt, 'elo': elo, 'origin': origin_league_prompts_dict_con[prompt]} for i, (prompt, elo) in enumerate(bottom_k_prompts_con)}
    }

    # Dump the top and bottom k prompts for the Pro side and Con side debators in a json file
    with open('top_bottom_prompts_dict_pro_poc.json', 'w') as json_file:
        json.dump(top_bottom_prompts_dict_pro_poc, json_file, indent=4)
    
    with open('top_bottom_prompts_dict_con_poc.json', 'w') as json_file:
        json.dump(top_bottom_prompts_dict_con_poc, json_file, indent=4)

    return


def compare_poc_elo_average(poc_elo_dict, pro_prompts1, con_prompts1, pro_prompts2, con_prompts2, league1, league2):
    # Get the average of pro_prompts1, con_prompts1, pro_prompts2, con_prompts2 using the poc_elo_dict
    pro_elo_average1 = 0
    con_elo_average1 = 0
    for prompt in pro_prompts1:
        pro_elo_average1 += poc_elo_dict[prompt]
    pro_elo_average1 = pro_elo_average1/len(pro_prompts1)
    print("pro_elo_average1: ", pro_elo_average1)
    for prompt in con_prompts1:
        con_elo_average1 += poc_elo_dict[prompt]
    con_elo_average1 = con_elo_average1/len(con_prompts1)
    print("con_elo_average1: ", con_elo_average1)

    pro_elo_average2 = 0
    con_elo_average2 = 0
    for prompt in pro_prompts2:
        pro_elo_average2 += poc_elo_dict[prompt]
    pro_elo_average2 = pro_elo_average2/len(pro_prompts2)
    print("pro_elo_average2: ", pro_elo_average2)

    for prompt in con_prompts2:
        con_elo_average2 += poc_elo_dict[prompt]
    con_elo_average2 = con_elo_average2/len(con_prompts2)
    print("con_elo_average2: ", con_elo_average2)

    # Compare the average of pro_prompts1, con_prompts1, pro_prompts2, con_prompts2
    if (pro_elo_average1 > pro_elo_average2 or con_elo_average1 > con_elo_average2):
        print(f"Proof of Concept {league1} vs {league2} Failed!")
    
    else:
        print(f"Proof of Concept {league1} vs {league2} Passed!")


# Function to play the tournament. In this, first we play the initial league on the prompts generated by the get_prompts() function and then we get the top and bottom k prompts. Then we generate new prompts using the get_new_prompts() function and play the league on these new prompts. We repeat this process for n-1 number of leagues. We give the top and bottom k prompts of current league as input to the get_new_prompts() function to generate new prompts for the next league. Then we play the n'th league which is the final league which takes the top k prompts and bottom k from each of the n-1 leagues and conducts a final league on these prompts. The top k prompts from this final league are the top k prompts of the tournament.
def tournament(no_of_topics, no_of_prompts_start, no_of_prompts_between, total_rounds, k, n, temp, max_tokens, current_directory):
    
        """
        Parameters
        ----------
        no_of_topics: int
            Number of topics for the Debate Battle
        no_of_prompts: int
            Number of prompts required for the Debate Battle for each role in the first league
        no_of_prompts_between: int
            Number of prompts required for the Debate Battle for each role in the subsequent leagues
        total_rounds: int
            The total number of rounds in the Debate Battle
        k: int
            Number of top and bottom prompts to be returned after each league
        n: int
            Number of leagues to be played in the tournament
        
        Returns
        --------
        None
        """


        global top_bottom_prompts_dict_pro
        global top_bottom_prompts_dict_con
        global elo_ratings_dict
        global current_league


        file_path = os.path.join(current_directory, 'humans.csv')
        file_path_2 = os.path.join(current_directory, 'human_topics.csv')
        
        topics = get_debate_topic(no_of_topics, temp, max_tokens, file_path_2)
        consecutive_fails = 0
        # Play all the leagues
        while(consecutive_fails<3):
            print(f"League {current_league}:\n")
            # Play the first league
            if (current_league ==1):

                pro_prompts = get_prompts(no_of_prompts_start, 'Pro', current_league, temp, max_tokens, None, None, False, None)
                con_prompts = get_prompts(no_of_prompts_start, 'Con', current_league, temp, max_tokens, None, None, False, None)

                if pro_prompts is None or con_prompts is None:
                    print("Invalid Input")
                    return
                
                top_pro_prompts_league_1, bottom_pro_prompts_league_1, top_con_prompts_league_1, bottom_con_prompts_league_1 = league(pro_prompts, con_prompts, topics, total_rounds, k, current_league, temp, max_tokens, False, None)
                # Get only the prompts from the top_pro_prompts_league1, bottom_pro_prompts_league1, top_con_prompts_league1, bottom_con_prompts_league1
                top_pro_prompts_league_1 = [prompt for prompt, elo in top_pro_prompts_league_1]
                bottom_pro_prompts_league_1 = [prompt for prompt, elo in bottom_pro_prompts_league_1]
                top_con_prompts_league_1 = [prompt for prompt, elo in top_con_prompts_league_1]
                bottom_con_prompts_league_1 = [prompt for prompt, elo in bottom_con_prompts_league_1]

                current_league+=1

            elif (current_league !=1):

                if(current_league==6):
                    break

                if (current_league ==2):
                # Generate new prompts for the next league using the top k and bottom k prompts from the previous league
                    pro_prompts = get_prompts(no_of_prompts_between, 'Pro', current_league, temp, max_tokens, top_pro_prompts_league_1, bottom_pro_prompts_league_1, False, None)
                    con_prompts = get_prompts(no_of_prompts_between, 'Con', current_league, temp, max_tokens, top_con_prompts_league_1, bottom_con_prompts_league_1, False, None)
                    if pro_prompts is None or con_prompts is None:
                        print("Invalid Input")
                        return

                # Play the league on the new prompts
                    top_pro_prompts_league, bottom_pro_prompts_league, top_con_prompts_league, bottom_con_prompts_league = league(pro_prompts, con_prompts, topics, total_rounds, k, current_league, temp, max_tokens, False, None)
                    
                    # Get only the prompts from the top_pro_prompts_league, bottom_pro_prompts_league, top_con_prompts_league, bottom_con_prompts_league
                    top_pro_prompts_league = [prompt for prompt, elo in top_pro_prompts_league]
                    bottom_pro_prompts_league = [prompt for prompt, elo in bottom_pro_prompts_league]
                    top_con_prompts_league = [prompt for prompt, elo in top_con_prompts_league]
                    bottom_con_prompts_league = [prompt for prompt, elo in bottom_con_prompts_league]

                    # Merge the top pro prompts of league 1 and 2 into a new list
                    merged_top_pro_prompts = top_pro_prompts_league_1 + top_pro_prompts_league
                    merged_bottom_pro_prompts = bottom_pro_prompts_league_1 + bottom_pro_prompts_league

                    # Merge the top con prompts of league 1 and 2
                    merged_top_con_prompts = top_con_prompts_league_1 + top_con_prompts_league
                    merged_bottom_con_prompts = bottom_con_prompts_league_1 + bottom_con_prompts_league

                # Play the final league using the merged top pro and con prompts from league 1 and 2
                    top_pro_prompts_final, bottom_pro_prompts_final, top_con_prompts_final, bottom_con_prompts_final, final_elo_dict = final_league(merged_top_pro_prompts, merged_top_con_prompts, topics, total_rounds, current_league, k, temp, max_tokens)

                    # Get only the prompts from the top_pro_prompts_final, bottom_pro_prompts_final, top_con_prompts_final, bottom_con_prompts_final
                    top_pro_prompts_final = [prompt for prompt, elo in top_pro_prompts_final]
                    bottom_pro_prompts_final = [prompt for prompt, elo in bottom_pro_prompts_final]
                    top_con_prompts_final = [prompt for prompt, elo in top_con_prompts_final]
                    bottom_con_prompts_final = [prompt for prompt, elo in bottom_con_prompts_final]

                    current_league+=1
                
                elif(current_league>2):

                    # Merge the top pro prompts of final
                    pro_prompts = get_prompts(no_of_prompts_between, 'Pro', current_league, temp, max_tokens, top_pro_prompts_final, bottom_pro_prompts_league, False, None)
                    con_prompts = get_prompts(no_of_prompts_between, 'Con', current_league, temp, max_tokens, top_con_prompts_final, bottom_con_prompts_league, False, None)

                    if pro_prompts is None or con_prompts is None:
                        print("Invalid Input")
                        return
                    
                    top_pro_prompts_league, bottom_pro_prompts_league, top_con_prompts_league, bottom_con_prompts_league = league(pro_prompts, con_prompts, topics, total_rounds, k, current_league, temp, max_tokens, False, None)

                    # Get only the prompts from the top_pro_prompts_league, bottom_pro_prompts_league, top_con_prompts_league, bottom_con_prompts_league
                    top_pro_prompts_league = [prompt for prompt, elo in top_pro_prompts_league]
                    bottom_pro_prompts_league = [prompt for prompt, elo in bottom_pro_prompts_league]
                    top_con_prompts_league = [prompt for prompt, elo in top_con_prompts_league]
                    bottom_con_prompts_league = [prompt for prompt, elo in bottom_con_prompts_league]

                    # Merge the top pro prompts of league and final
                    merged_top_pro_prompts = top_pro_prompts_final + top_pro_prompts_league
                    merged_bottom_pro_prompts = bottom_pro_prompts_final + bottom_pro_prompts_league

                    # Merge the top con prompts of league and final
                    merged_top_con_prompts = top_con_prompts_final + top_con_prompts_league
                    merged_bottom_con_prompts = bottom_con_prompts_final + bottom_con_prompts_league

                    # Play the final league using the merged top pro and con prompts from league and final
                    top_pro_prompts_final, bottom_pro_prompts_final, top_con_prompts_final, bottom_con_prompts_final, final_elo_dict = final_league(merged_top_pro_prompts, merged_top_con_prompts, topics, total_rounds, current_league, k, temp, max_tokens)

                    # Get only the prompts from the top_pro_prompts_final, bottom_pro_prompts_final, top_con_prompts_final, bottom_con_prompts_final
                    top_pro_prompts_final = [prompt for prompt, elo in top_pro_prompts_final]
                    bottom_pro_prompts_final = [prompt for prompt, elo in bottom_pro_prompts_final]
                    top_con_prompts_final = [prompt for prompt, elo in top_con_prompts_final]
                    bottom_con_prompts_final = [prompt for prompt, elo in bottom_con_prompts_final]

                    consecutive_fails = compare_elo_average(top_pro_prompts_league, top_con_prompts_league, final_elo_dict, current_league, k, consecutive_fails)

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
        
        proof_of_concept(1, last_league_no, topics, total_rounds, k , temp, max_tokens)
        proof_of_concept(2, last_league_no, topics, total_rounds, k , temp, max_tokens)
        proof_of_concept(3, last_league_no, topics, total_rounds, k , temp, max_tokens)
        proof_of_concept(2, last_league_no-1, topics, total_rounds, k , temp, max_tokens)

# Function to find rogue score
def rogue_score_matrix():

    # Initialise the rogue_score_matrix_dict_pro
    rogue_score_matrix_dict_pro = {}

    # Initialise the rogue_score_matrix_dict_con
    rogue_score_matrix_dict_con = {}

    # Read the top_bottom_prompts_dict_pro and top_bottom_prompts_dict_con json files
    with open('top_bottom_prompts_dict_pro.json', 'r') as json_file:
        top_bottom_prompts_dict_pro = json.load(json_file)
    
    # Read the top_bottom_prompts_dict_con json file
    with open('top_bottom_prompts_dict_con.json', 'r') as json_file:
        top_bottom_prompts_dict_con = json.load(json_file)
    
    # Read tjhe top_bottom_prompts_dict_pro_across_league json file
    with open('top_bottom_prompts_dict_pro_across_league.json', 'r') as json_file:
        top_bottom_prompts_dict_pro_across_league = json.load(json_file)
    
    # Read the top_bottom_prompts_dict_con_across_league json file
    with open('top_bottom_prompts_dict_con_across_league.json', 'r') as json_file:
        top_bottom_prompts_dict_con_across_league = json.load(json_file)

    # Iterate through the league number of the top_bottom_prompts_dict_pro
    for league_no, league_dict in top_bottom_prompts_dict_pro.items():
        # Get the top k prompts from the league_dict
        top_k_prompts_pro = [item["prompt"] for item in league_dict["top_k_prompts"].values()]

        # Iterature through the league number from league_no+1 onwards in the top_bottom_prompts_dict_pro_across_league
        for league_no_across, league_dict_across in top_bottom_prompts_dict_pro_across_league.items():
            # Check if the json file exists
            if os.path.exists("rogue_score_matrix_pro.json"):
                with open("rogue_score_matrix_pro.json", 'r') as json_file:
                    rogue_score_matrix_dict_pro = json.load(json_file)
                # Check if the rogue score is already calculated for this league_no vs league_no_across
                if f"{league_no} vs 1-{league_no_across}" in rogue_score_matrix_dict_pro:
                    print("Already calculated")
                    continue

            top_k_prompts_pro_across = [item["prompt"] for item in league_dict_across["top_k_prompts"].values()]
            rogue_L_average = 0

            # Get the rogue score of each prompt of top_k_prompts_pro with each prompt of top_k_prompts_pro_across using the rogue_score function
            for prompt in top_k_prompts_pro:
                for prompt_across in top_k_prompts_pro_across:
                    results = rogue.compute(predictions=[prompt], references=[prompt_across])
                    # Get the rogue L score from the results
                    rogue_L_score = results["rougeL"]
                    print("Prompt: ", prompt)
                    print("Prompt Across: ", prompt_across)
                    print("Rogue L Score: ", rogue_L_score)
                    rogue_L_average += rogue_L_score
                
            
            # Get the average rogue L score of the top_k_prompts_pro with the top_k_prompts_pro_across
            rogue_L_average = rogue_L_average/(len(top_k_prompts_pro)*len(top_k_prompts_pro_across))

            # Store the rogue L average in a json file named rogue_score_matrix_pro.json
            rogue_score_matrix_dict_pro[f"{league_no} vs 1-{league_no_across}"] = rogue_L_average

            # Dump the dict in the json file
            with open("rogue_score_matrix_pro.json", 'w') as json_file:
                json.dump(rogue_score_matrix_dict_pro, json_file, indent=4)

    # Iterature through the league number of the top_bottom_prompts_dict_con
    for league_no, league_dict in top_bottom_prompts_dict_con.items():
        # Get the top k prompts from the league_dict
        top_k_prompts_con = [item["prompt"] for item in league_dict["top_k_prompts"].values()]


        # Iterature through the league number from league_no+1 onwards in the top_bottom_prompts_dict_con_across_league
        for league_no_across, league_dict_across in top_bottom_prompts_dict_con_across_league.items():
            # Check if the json file exists
            if os.path.exists("rogue_score_matrix_con.json"):
                with open("rogue_score_matrix_con.json", 'r') as json_file:
                    rogue_score_matrix_dict_con = json.load(json_file)
                # Check if the rogue score is already calculated for this league_no vs league_no_across
                if f"{league_no} vs 1-{league_no_across}" in rogue_score_matrix_dict_con:
                    print("Already calculated")
                    continue

            top_k_prompts_con_across = [item["prompt"] for item in league_dict_across["top_k_prompts"].values()]

            rogue_L_average = 0

            # Get the rogue score of each prompt of top_k_prompts_con with each prompt of top_k_prompts_con_across using the rogue_score function
            for prompt in top_k_prompts_con:
                for prompt_across in top_k_prompts_con_across:
                    results = rogue.compute(predictions=[prompt], references=[prompt_across])
                    # Get the rogue L score from the results
                    rogue_L_score = results["rougeL"]
                    rogue_L_average += rogue_L_score
                
            
            # Get the average rogue L score of the top_k_prompts_con with the top_k_prompts_con_across
            rogue_L_average = rogue_L_average/(len(top_k_prompts_con)*len(top_k_prompts_con_across))

            # Store the rogue L average in a json file named rogue_score_matrix_con.json
            rogue_score_matrix_dict_con[f"{league_no} vs 1-{league_no_across}"] = rogue_L_average

            # Dump the dict in the json file
            with open("rogue_score_matrix_con.json", 'w') as json_file:
                json.dump(rogue_score_matrix_dict_con, json_file, indent=4)
            

def main():


    global elo_ratings_dict
    global league_dict
    global top_bottom_prompts_dict_pro
    global top_bottom_prompts_dict_con
    global human_prompts_used_dict_pro
    global human_prompts_used_dict_con
    global top_bottom_prompts_dict_con_across_league
    global top_bottom_prompts_dict_pro_across_league
    global current_league
    global origin_league_prompts_dict_pro
    global origin_league_prompts_dict_con
    global top_bottom_prompts_dict_pro_poc
    global top_bottom_prompts_dict_con_poc

    current_directory = os.getcwd()
    no_of_topics = 132
    temp = [0.5]
    max_tokens =[500]
    n = 4

    no_of_runs = 0
    start_main_time = time.time()

    # for no_of_prompts_start in range (8,11,2):
    #     for no_of_prompts_between in range (8,11,2):
    #         for total_rounds in range (3,6,1):
    #             for k in range (3,math.floor(no_of_prompts_between/2),1):
    #                 for temperature in temp:
    #                     for max_token in max_tokens:
    #                         if (no_of_runs==1):
    #                             break
    #                         print("no_of_prompts_start: ", no_of_prompts_start)
    #                         print("no_of_prompts_between: ", no_of_prompts_between)
    #                         print("total_rounds: ", total_rounds)
    #                         print("k: ", k)
    #                         print("temperature: ", temperature)
    #                         print("max_token: ", max_token)
    #                         folder_name = "no_of_prompts_start_"+str(no_of_prompts_start)+"_no_of_prompts_between_"+str(no_of_prompts_between)+"_total_rounds_"+str(total_rounds)+"_k_"+str(k)+"_temperature_"+str(temperature)+"_max_token_"+str(max_token)
    #                         new_directory = os.path.join(current_directory, folder_name)

    #                         # Check if the directory is already present then don't make the directory again
    #                         if os.path.exists(new_directory):
    #                             print(f"Directory {new_directory} already exists!")
                                
    #                         else:
    #                             os.makedirs(new_directory, exist_ok=True)

    #                         os.chdir(new_directory)
    #                         tournament(no_of_topics, no_of_prompts_start, no_of_prompts_between, total_rounds, k, n, temperature, max_token, current_directory)
    #                         # Dictionary to store the ELO ratings of the prompts
    #                         elo_ratings_dict = {}
    #                         # Dictionary to store the history of the games in the leagues
    #                         league_dict = {}
    #                         #league number
    #                         current_league = 1
    #                         # Dictionary to store the top and bottom k prompts from each league
    #                         top_bottom_prompts_dict_pro = {}
    #                         top_bottom_prompts_dict_con = {}
    #                         # Dictionary to store the human prompts which have been used in a league already
    #                         human_prompts_used_dict_pro = {}
    #                         human_prompts_used_dict_con = {}
    #                         top_bottom_prompts_dict_con_across_league = {}
    #                         top_bottom_prompts_dict_pro_across_league = {}
    #                         origin_league_prompts_dict_pro = {}
    #                         origin_league_prompts_dict_con = {}
    #                         top_bottom_prompts_dict_pro_poc = {}
    #                         top_bottom_prompts_dict_con_poc = {}
    #                         no_of_runs += 1
                            
    # end_main_time = time.time()
    # print("Total time taken for all the runs: ", end_main_time - start_main_time)
    
    rogue_score_matrix()

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
