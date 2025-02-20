import os
import json

# Global variables - Output settings
OUTPUT_FILE = 'test.json'       # Output file name
PREPEND_KEYWORD = "あなたは古代ギリシャの哲学者ソクラテスです。"  # Keyword to prepend to utterances

# Extraction range settings (skip if both are 0)
EXTRACTION_SETTINGS = [
    {
        'start': 3,
        'end': 4
    },
    {
        'start': 5,
        'end': 6
    },
    {
        'start': 7,
        'end': 8
    },
    {
        'start': 9,
        'end': 10
    },
    {
        'start': 11,
        'end': 12
    },
    {
        'start': 13,
        'end': 14
    },
    {
        'start': 15,
        'end': 16
    },
    {
        'start': 17,
        'end': 18
    },
    {
        'start': 19,
        'end': 20
    },
    {
        'start': 21,
        'end': 22
    },
    {
        'start': 23,
        'end': 24
    },
    {
        'start': 25,
        'end': 26
    }
]

def extract_dialogue_from_file(file_path):
    """Extract dialogues from file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Get text after === Dialogue Log ===
    dialogue_section = content.split("=== Dialogue Log ===")
    if len(dialogue_section) < 2:
        return None
    
    # Extract dialogue portion
    dialogue_text = dialogue_section[1].strip()
    
    # Split by User: and Assistant: to get utterances
    utterances = []
    current_speaker = None
    current_utterance = []
    
    lines = dialogue_text.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('User:') or line.startswith('Assistant:'):
            # Save previous utterance
            if current_speaker and current_utterance:
                utterances.append((current_speaker, '\n'.join(current_utterance)))
            
            # Start new utterance
            current_speaker = 'User' if line.startswith('User:') else 'Assistant'
            prefix_len = 5 if line.startswith('User:') else 10
            current_utterance = [line[prefix_len:].strip()]
        elif line.strip():
            # Add to current utterance if line is not empty
            current_utterance.append(line.strip())
    
    # Save last utterance
    if current_speaker and current_utterance:
        utterances.append((current_speaker, '\n'.join(current_utterance)))
    
    return utterances

def save_extracted_dialogue(utterances, original_filename, start, end):
    """Save extracted dialogues in Gemma-2 format to JSON file"""
    # Skip if start or end is 0
    if start == 0 or end == 0:
        return
    
    output_dir = os.path.join('data', 'dialogue', 'processed')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if start > len(utterances) or end > len(utterances):
        print(f"Warning: Requested utterance range {start}-{end} exceeds available utterances ({len(utterances)})")
        return
    
    output_path = os.path.join(output_dir, OUTPUT_FILE)
    
    # Convert dialogue data to Gemma-2 format
    chat_data = []
    for i in range(start-1, end):
        speaker, text = utterances[i]
        role = "user" if speaker == "User" else "model"
        # Convert newlines to \\n
        text = text.replace('\n', '\\n')
        
        # Add keyword to first utterance of each extraction range
        if i == start-1 and PREPEND_KEYWORD:
            text = PREPEND_KEYWORD + text
            
        chat_data.append({
            "role": role,
            "content": text
        })
    
    # Load existing JSON file if it exists
    existing_data = []
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = []
    
    # Add new dialogue data
    existing_data.append({
        "source_file": original_filename,
        "extract_range": f"{start}-{end}",
        "messages": chat_data
    })
    
    # Save to JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=2)
    
    print(f"Extracted dialogue saved to: {output_path}")

def process_dialogue_files(input_dir):
    """Process all dialogue files in the specified directory"""
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(input_dir, filename)
            utterances = extract_dialogue_from_file(file_path)
            if utterances:
                # Process each extraction setting
                for setting in EXTRACTION_SETTINGS:
                    save_extracted_dialogue(
                        utterances,
                        filename,
                        setting['start'],
                        setting['end']
                    )

# Main process
if __name__ == "__main__":
    input_directory = os.path.join('data', 'dialogue', 'raw')
    process_dialogue_files(input_directory) 