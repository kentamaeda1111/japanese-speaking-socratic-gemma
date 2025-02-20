from anthropic import Anthropic
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import json
from datetime import datetime
import os
import csv
import time
from src.utils.config import get_api_keys

# Model parameter settings
# AI1 (User) parameters
AI1_MODEL_NAME = "claude-3-5-sonnet-20241022"
AI1_MAX_TOKENS = 2048

# AI2 (Assistant) parameters
AI2_MODEL_NAME = "claude-3-5-sonnet-20241022"
AI2_MAX_TOKENS = 2048

# API settings
api_keys = get_api_keys()
AI1_API_KEY = api_keys['claude_api_key_1']
AI2_API_KEY = api_keys['claude_api_key_2']

# Note: You need two different API keys for the dual AI system
# Visit https://console.anthropic.com to obtain your API keys

@dataclass
class AIConfig:
    """Data class for AI configuration"""
    api_key: str
    model: Optional[str] = None

class AutomationSettings:
    """Class for managing automation settings"""
    def __init__(self, settings: Dict[str, Any]):
        # AI1 (User) parameters
        self.AI1_MODEL_NAME = AI1_MODEL_NAME
        self.AI1_MAX_TOKENS = AI1_MAX_TOKENS
        self.AI1_TEMPERATURE = float(settings['AI1_TEMPERATURE'])
        self.AI1_API_KEY = AI1_API_KEY

        # AI2 (Assistant) parameters
        self.AI2_MODEL_NAME = AI2_MODEL_NAME
        self.AI2_MAX_TOKENS = AI2_MAX_TOKENS
        self.AI2_TEMPERATURE = float(settings['AI2_TEMPERATURE'])
        self.AI2_API_KEY = AI2_API_KEY

        # Global settings
        self.MAX_MESSAGE_PAIRS = int(settings['MAX_MESSAGE_PAIRS'])
        self.MAX_TURNS = int(settings['MAX_TURNS'])
        self.INITIAL_QUESTION_ID = int(settings['INITIAL_QUESTION_ID'])
        self.DIALOGUE_KEYWORD = settings['DIALOGUE_KEYWORD']

        # Prompt settings
        self.USER_PROMPT_ID = int(settings['USER_PROMPT_ID'])
        self.OTHERS_ID = int(settings['OTHERS_ID'])
        self.PERSONA_ID = int(settings['PERSONA_ID'])
        self.TRANSFORM_ID = int(settings['TRANSFORM_ID'])
        self.ASSISTANT_PROMPT_ID = int(settings['ASSISTANT_PROMPT_ID'])
        self.RESPONSE_ID = int(settings['RESPONSE_ID'])
        self.UPDATE_ID = int(settings['UPDATE_ID'])

class DialogueLogger:
    """Class for logging dialogue content to files"""
    def __init__(self, settings: AutomationSettings):
        if not os.path.exists('data/dialogue/raw'):
            os.makedirs('data/dialogue/raw')
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = f'data/dialogue/raw/dialogue_{settings.DIALOGUE_KEYWORD}_{timestamp}_{settings.MAX_TURNS}.txt'
        self.settings = settings
        
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(self._get_log_header())

    def _get_log_header(self) -> str:
        """Generate log file header information"""
        try:
            # Load prompts from JSON files
            user_prompt = self._load_prompt_from_json('user_system_prompt.json', self.settings.USER_PROMPT_ID)
            assistant_prompt = self._load_prompt_from_json('assistant_system_prompt.json', self.settings.ASSISTANT_PROMPT_ID)
            
            # Get attribute data
            persona_data = self._load_attribute_data('persona.json', self.settings.PERSONA_ID, 'personas')
            others_data = self._load_attribute_data('others.json', self.settings.OTHERS_ID, 'others')
            transform_data = self._load_attribute_data('transform.json', self.settings.TRANSFORM_ID, 'transform')
            response_data = self._load_attribute_data('response.json', self.settings.RESPONSE_ID, 'responses')
            update_data = self._load_attribute_data('update.json', self.settings.UPDATE_ID, 'update')
            
            header = f"""AI1 (User) Parameters:
- Model: {self.settings.AI1_MODEL_NAME}
- Max Tokens: {self.settings.AI1_MAX_TOKENS}
- Temperature: {self.settings.AI1_TEMPERATURE}

AI2 (Assistant) Parameters:
- Model: {self.settings.AI2_MODEL_NAME}
- Max Tokens: {self.settings.AI2_MAX_TOKENS}
- Temperature: {self.settings.AI2_TEMPERATURE}

Global Settings:
- Max Message Pairs: {self.settings.MAX_MESSAGE_PAIRS}
- Max Turns: {self.settings.MAX_TURNS}
- Initial Question ID: {self.settings.INITIAL_QUESTION_ID}

User Details:
- User Prompt ID: {self.settings.USER_PROMPT_ID}
- Title: {user_prompt['title']}
- Others:
  - ID: {self.settings.OTHERS_ID}
  - Label: {others_data['label']}
- Persona:
  - ID: {self.settings.PERSONA_ID}
  - Label: {persona_data['label']}
- Transform:
  - ID: {self.settings.TRANSFORM_ID}
  - Label: {transform_data['label']}

Assistant Details:
- Assistant Prompt ID: {self.settings.ASSISTANT_PROMPT_ID}
- Title: {assistant_prompt['title']}
- Response:
  - ID: {self.settings.RESPONSE_ID}
  - Label: {response_data['label']}
- Update:
  - ID: {self.settings.UPDATE_ID}
  - Label: {update_data['label']}

=== Dialogue Log ===

"""
            return header
        except Exception as e:
            print(f"Error generating log header: {str(e)}")
            return "=== Dialogue Log ===\n\n"

    def log_message(self, speaker: str, message: str):
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"{speaker}: {message}\n\n")

    def _load_prompt_from_json(self, file_name: str, prompt_id: int) -> Dict:
        """Load prompt from JSON file"""
        try:
            if file_name == 'user_system_prompt.json':
                dir_path = os.path.join('data', 'prompts', 'user_system_prompt')
            elif file_name == 'assistant_system_prompt.json':
                dir_path = os.path.join('data', 'prompts', 'assistant_system_prompt')
            else:
                dir_path = os.path.join('data', 'prompts')
            
            file_path = os.path.join(dir_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for prompt in data['prompts']:
                    if prompt['id'] == prompt_id:
                        return prompt
            raise ValueError(f"Prompt ID {prompt_id} not found in {file_name}")
        except FileNotFoundError:
            raise FileNotFoundError(f"File {file_name} not found in prompts directory")

    def _load_attribute_data(self, file_name: str, attribute_id: int, key: str) -> Dict:
        """Load attribute data from JSON file"""
        try:
            file_path = os.path.join(
                'data', 'prompts',
                'assistant_system_prompt' if file_name in ['response.json', 'update.json'] else 'user_system_prompt',
                file_name
            )
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data[key]:
                    if item['id'] == attribute_id:
                        return item
            raise ValueError(f"Attribute ID {attribute_id} not found in {file_name}")
        except Exception as e:
            print(f"Error loading attribute data: {str(e)}")
            return {"label": "Error loading data"}

class AIDialogue:
    def __init__(self, settings: AutomationSettings):
        self.settings = settings
        self.client1 = Anthropic(api_key=settings.AI1_API_KEY)
        self.client2 = Anthropic(api_key=settings.AI2_API_KEY)
        
        self.ai1_messages = []
        self.ai2_messages = []
        self.logger = DialogueLogger(settings)

        # Load system prompts
        self.ai1_system_prompt = self._load_user_system_prompt()
        self.ai2_system_prompt = self._load_assistant_system_prompt()

    def _load_user_system_prompt(self) -> str:
        """Load user system prompt and replace all placeholders"""
        try:
            # Load prompt from user_system_prompt.json
            user_prompt = self._load_prompt_from_json(
                'user_system_prompt.json', 
                self.settings.USER_PROMPT_ID
            )
            # Load initial question from questions.json
            question = self._load_prompt_from_json(
                'questions.json', 
                self.settings.INITIAL_QUESTION_ID
            )
            
            # Get content from JSON files
            persona_data = self._load_attribute_data(
                'persona.json', 
                self.settings.PERSONA_ID, 
                'personas'
            )
            others_data = self._load_attribute_data(
                'others.json', 
                self.settings.OTHERS_ID, 
                'others'
            )
            transform_data = self._load_attribute_data(
                'transform.json', 
                self.settings.TRANSFORM_ID, 
                'transform'
            )
            
            content = user_prompt['content']
            
            # Replace placeholders
            if '{{INITIAL_QUESTION}}' in content:
                content = content.replace('{{INITIAL_QUESTION}}', question['content'])
            
            if '{{PERSONA}}' in content:
                content = content.replace('{{PERSONA}}', persona_data['content'])
            
            if '{{OTHERS}}' in content:
                content = content.replace('{{OTHERS}}', others_data['content'])
            
            if '{{TRANSFORM}}' in content:
                content = content.replace('{{TRANSFORM}}', transform_data['content'])
            
            return content
            
        except Exception as e:
            print(f"Error loading user system prompt: {str(e)}")
            return ""

    def _load_assistant_system_prompt(self) -> str:
        """Load assistant system prompt and replace placeholders"""
        try:
            # Load prompt from assistant_system_prompt.json
            assistant_prompt = self._load_prompt_from_json(
                'assistant_system_prompt.json', 
                self.settings.ASSISTANT_PROMPT_ID
            )
            # Load initial question from questions.json
            question = self._load_prompt_from_json(
                'questions.json', 
                self.settings.INITIAL_QUESTION_ID
            )
            
            # Get content from JSON files
            response_data = self._load_attribute_data(
                'response.json', 
                self.settings.RESPONSE_ID, 
                'responses'
            )
            update_data = self._load_attribute_data(
                'update.json', 
                self.settings.UPDATE_ID, 
                'update'
            )
            
            content = assistant_prompt['content']
            
            # Replace placeholders
            if '{{INITIAL_QUESTION}}' in content:
                content = content.replace('{{INITIAL_QUESTION}}', question['content'])
            
            if '{{RESPONSE}}' in content:
                content = content.replace('{{RESPONSE}}', response_data['content'])
            
            if '{{UPDATE}}' in content:
                content = content.replace('{{UPDATE}}', update_data['content'])
            
            return content
            
        except Exception as e:
            print(f"Error loading assistant system prompt: {str(e)}")
            return ""

    def _load_prompt_from_json(self, file_name: str, prompt_id: int) -> Dict:
        """Load prompt from JSON file"""
        try:
            if file_name == 'user_system_prompt.json':
                dir_path = os.path.join('data', 'prompts', 'user_system_prompt')
            elif file_name == 'assistant_system_prompt.json':
                dir_path = os.path.join('data', 'prompts', 'assistant_system_prompt')
            else:
                dir_path = os.path.join('data', 'prompts')
            
            file_path = os.path.join(dir_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for prompt in data['prompts']:
                    if prompt['id'] == prompt_id:
                        return prompt
            raise ValueError(f"Prompt ID {prompt_id} not found in {file_name}")
        except FileNotFoundError:
            raise FileNotFoundError(f"File {file_name} not found in prompts directory")

    def _load_attribute_data(self, file_name: str, attribute_id: int, key: str) -> Dict:
        """Load attribute data from JSON file"""
        try:
            file_path = os.path.join(
                'data', 'prompts',
                'assistant_system_prompt' if file_name in ['response.json', 'update.json'] else 'user_system_prompt',
                file_name
            )
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data[key]:
                    if item['id'] == attribute_id:
                        return item
            raise ValueError(f"Attribute ID {attribute_id} not found in {file_name}")
        except Exception as e:
            print(f"Error loading attribute data: {str(e)}")
            return {"label": "Error loading data"}

    def manage_message_history(self, messages: list) -> list:
        if len(messages) > self.settings.MAX_MESSAGE_PAIRS * 2:
            messages = messages[-(self.settings.MAX_MESSAGE_PAIRS * 2):]
        return messages

    def call_ai_api(self, prompt: str, is_ai2: bool) -> str:
        """Send request to AI model and get response"""
        try:
            # Select appropriate parameters
            model_name = self.settings.AI2_MODEL_NAME if is_ai2 else self.settings.AI1_MODEL_NAME
            max_tokens = self.settings.AI2_MAX_TOKENS if is_ai2 else self.settings.AI1_MAX_TOKENS
            temperature = self.settings.AI2_TEMPERATURE if is_ai2 else self.settings.AI1_TEMPERATURE
            
            system_prompt = self.ai2_system_prompt if is_ai2 else self.ai1_system_prompt
            messages = self.ai2_messages if is_ai2 else self.ai1_messages
            client = self.client1 if is_ai2 else self.client2
            
            # Add new message and manage history
            messages.append({"role": "user", "content": prompt})
            messages = self.manage_message_history(messages)
            
            # Display API request content
            request_body = {
                "model": model_name,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": system_prompt,
                "messages": messages
            }
            print("\nAPI Request:")
            print("=" * 50)
            print(json.dumps(request_body, ensure_ascii=False, indent=2))
            print("=" * 50)
            
            # Execute API request
            response = client.messages.create(
                model=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=messages
            )
            
            # Add response to history and manage again
            assistant_message = {"role": "assistant", "content": response.content[0].text}
            messages.append(assistant_message)
            messages = self.manage_message_history(messages)
            
            # Update self.ai2_messages or self.ai1_messages
            if is_ai2:
                self.ai2_messages = messages
                self.logger.log_message("Assistant", response.content[0].text)
            else:
                self.ai1_messages = messages
                self.logger.log_message("User", response.content[0].text)
            
            return response.content[0].text
            
        except Exception as e:
            error_message = f"API error occurred: {str(e)}"
            self.logger.log_message("Error", error_message)
            return error_message

class AutomationManager:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path

    def load_csv_settings(self) -> List[Dict[str, Any]]:
        settings_list = []
        try:
            with open(self.csv_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                print(f"CSV columns: {reader.fieldnames}")
                for row in reader:
                    if any(row.values()):
                        settings_list.append(row)
                        print(f"Row values: {row}")
        except Exception as e:
            print(f"Error loading CSV file: {str(e)}")
        return settings_list

    def update_csv_with_dialogue_file(self, row_index: int, dialogue_file: str) -> None:
        """Update dialogue column in CSV file for specific row"""
        try:
            # Read all CSV content
            with open(self.csv_path, 'r', encoding='utf-8-sig') as f:
                rows = list(csv.reader(f))
            
            # Update dialogue column for specified row
            rows[row_index + 1][13] = dialogue_file  # Add 1 to account for header row
            
            # Write updated content
            with open(self.csv_path, 'w', encoding='utf-8-sig', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(rows)
                
        except Exception as e:
            print(f"Error updating CSV file: {str(e)}")

    def get_latest_dialogue_file(self, keyword: str, max_turns: int) -> str:
        """Get the latest dialogue log filename"""
        log_dir = 'data/dialogue/raw'
        try:
            matching_files = [
                f for f in os.listdir(log_dir)
                if f.startswith(f'dialogue_{keyword}_') and f.endswith(f'_{max_turns}.txt')
            ]
            
            if matching_files:
                return max(matching_files, key=lambda x: os.path.getctime(os.path.join(log_dir, x)))
                
        except Exception as e:
            print(f"Error getting latest dialogue file: {str(e)}")
        
        return ""

    def run_automation(self):
        settings_list = self.load_csv_settings()
        
        # Read CSV to check already processed rows
        with open(self.csv_path, 'r', encoding='utf-8-sig') as f:
            reader = list(csv.reader(f))
            
            for i, settings_dict in enumerate(settings_list, 1):
                try:
                    if i >= len(reader) or len(reader[i]) <= 13:
                        print(f"Skipping row {i} (invalid row format)")
                        continue
                    
                    if reader[i][13].strip():
                        print(f"\nSkipping row {i} (already processed)")
                        continue
                    
                    print(f"\nProcessing row {i} of {len(settings_list)}...")
                    
                    settings = AutomationSettings(settings_dict)
                    dialogue = AIDialogue(settings)
                    
                    # Initial user question
                    first_user_question = "今日は何について話しますか？"
                    dialogue.logger.log_message("User", first_user_question)
                    
                    # Start dialogue from Socrates side
                    assistant_first_question = dialogue.call_ai_api(first_user_question, True)
                    print(f"Assistant (Turn 1): {assistant_first_question}\n")
                    
                    # Dialogue execution
                    for turn in range(settings.MAX_TURNS):
                        # User response
                        user_response = dialogue.call_ai_api(
                            assistant_first_question if turn == 0 else assistant_response, 
                            False
                        )
                        print(f"User (Turn {turn*2 + 2}): {user_response}\n")
                        
                        # AI2 (Socrates) response
                        if turn < settings.MAX_TURNS - 1:
                            assistant_response = dialogue.call_ai_api(user_response, True)
                            print(f"Assistant (Turn {turn*2 + 3}): {assistant_response}\n")
                    
                    # Get dialogue log filename and update CSV file
                    dialogue_file = self.get_latest_dialogue_file(
                        settings.DIALOGUE_KEYWORD,
                        settings.MAX_TURNS
                    )
                    if dialogue_file:
                        self.update_csv_with_dialogue_file(i-1, dialogue_file)
                    
                    time.sleep(0.1)  # Set short wait time if needed

                except Exception as e:
                    print(f"Error processing row {i}: {str(e)}")
                    continue

def main():
    automation = AutomationManager('data/config/automation.csv')
    automation.run_automation()

if __name__ == "__main__":
    main() 