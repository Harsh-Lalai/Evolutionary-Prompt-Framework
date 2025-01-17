
# Evolutionary-Prompt-Framework

## Project Description

The Evolutionary-Prompt-Framework is an automated prompt evolutionary system designed to optimize prompt generation for various games. The framework involves generating prompts using Large Language Models (LLMs) for different leagues. Each league is played for various games, and the best-performing prompts are selected to assist in generating prompts for further leagues. This iterative process helps in evolving more accurate and effective prompts over time.

## Setup Instructions

To get started with the project, follow these steps:

### 1. Clone the Repository

Clone the repository to your local machine by running:

```bash
git clone https://github.com/Harsh-Lalai/Evolutionary-Prompt-Framework.git
```

### 2. Navigate to the Project Directory

Change to the project directory:

```bash
cd Evolutionary-Prompt-Framework
```

### 3. Create a Conda Environment

Create a new Conda environment to manage dependencies:

```bash
conda create --name evolutionary-prompt-framework python=3.8
```

Activate the environment:

```bash
conda activate evolutionary-prompt-framework
```

### 4. Install the Required Packages

Use `pip` to install the necessary packages as specified in the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 5. Export OpenAI API Key

To interact with OpenAI's services, you need to export your API key:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Make sure to replace `"your-api-key-here"` with your actual OpenAI API key.

### 6. Select and Play a Game

Navigate to the folder corresponding to the game you want to play. For example:

```bash
cd Debate-Battle
```

### 7. Choose the Type of Prompt to Start With

Once inside the game folder, you will have the option to choose whether you want to start with **human-generated prompts** or **LLM-generated prompts**. Select the option that best suits your needs for the game.

- If you want to start with **human-generated prompts**, run the script with "human start" in the name.
- For **math game** and **Presidential Speech**, only the human start options are available.

### 8. Run the Framework

After selecting the type of prompts, execute the necessary Python files to start the evolutionary prompt process:

```bash
python run_game.py
```

Replace `run_game.py` with the appropriate script based on your game. For example:

```bash
python debate_battles_human_start.py
```
