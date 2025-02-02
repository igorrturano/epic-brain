# Epic Brain

**Epic Brain** is a bot developed to help GMs (Game Masters) track the interactions and actions of characters on [epic-shard.com](https://epic-shard.com). It processes logs of actions and dialogues of the characters and generates interpretative summaries, making it easier to follow the players' progress.

## Features

- **Log Summary**: The bot reads logs of character actions and dialogues and generates an interpretative summary.
- **Behavioral Interpretation**: In addition to listing actions, the bot provides an interpretation of the character's behavior.
- **Discord Integration**: The bot can be integrated with Discord, allowing game masters to request summaries directly in the chat.
- **Local Execution**: Everything is executed locally without the need for external APIs.

## Technologies Used

- **Python**: The main programming language of the project.
- **llama-cpp-python**: A library to load and interact with GGUF language models.
- **Discord.py**: A library for Discord integration.
- **GGUF Models**: Lightweight and efficient language models, such as Mistral 7B or LLaMA 2 7B.

## How to Set Up

### Prerequisites

1. **Python 3.8 or higher**.
2. **VPS or local machine** with sufficient memory to run the GGUF model.
3. **Discord Token**: Create a bot in the [Discord Developer Portal](https://discord.com/developers/applications) and obtain the token.
4. **.env File**: Create a `.env` file at the root of the project following the example model in `.env.example`.

### Setup Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/epic-brain.git
   cd epic-brain
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the GGUF model**:
   - Download a GGUF model (for example, Mistral 7B, LLaMA 2 7B, or DeepSeek R1 8B) from [Hugging Face](https://huggingface.co/TheBloke).
   - Place the model file in the `models/` folder.

4. **Configure the Discord bot**:
   - In the `.env` file, replace `'YOUR_TOKEN_HERE'` with your bot's token.

5. **Organize the logs**:
   - Place the character logs in the `logs/` folder, organized by date and character. Example:
     ```
     logs/
     ├── 2023-10-01/
     │   ├── character_X.txt
     │   ├── character_Y.txt
     ├── 2023-10-02/
     │   ├── character_X.txt
     │   ├── character_Z.txt
     ```

6. **Run the bot**:
   ```bash
   python bot.py
   ```

## How to Use

1. **On Discord**, send a message in the following format:
   ```
   !resumo character_X
   ```
   The bot will respond with a summary of the actions and an interpretation of the character's behavior.

2. **Example Response**:
   ```
   It appears that XXXXX is lost and trying to find their routine. They asked where things were and **opened a small box**, possibly in search of something. Their action of **opening the small box** can be interpreted as a gesture of desperation or a search for guidance.
   ```

## Project Structure

```
epic-brain/
├── logs/                  # Folder to store game logs
├── models/                # Folder to store the GGUF model
├── bot.py                 # Main bot script
├── summarizer.py          # Script for log summarization
├── summarizer_utils.py    # Utility functions
├── download_model.py      # Script to download the GGUF model
├── requirements.txt       # List of dependencies
└── README.md              # This file
```