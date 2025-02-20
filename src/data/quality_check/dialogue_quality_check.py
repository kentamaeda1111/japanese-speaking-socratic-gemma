from anthropic import Anthropic
from dataclasses import dataclass
from typing import List, Tuple, Optional
import os
import re
import time
import csv
import shutil
from src.utils.config import get_api_keys

# Model parameters settings
MODEL_NAME = "claude-3-5-sonnet-20241022"
MAX_TOKENS = 3000
TEMPERATURE = 0.3

# Automation settings
BATCH_SIZE = 1  # Set the number of files to process in each batch to 1
WAIT_TIME = 0  # Remove wait time

api_keys = get_api_keys()
API_KEY = api_keys['claude_api_key_quality']

@dataclass
class AIConfig:
    """Data class for AI settings"""
    api_key: str
    model: Optional[str] = None

class AlreadyEvaluatedError(Exception):
    """Exception raised when a file has already been evaluated"""
    pass

def extract_dialogue_pairs(file_content: str) -> List[Tuple[str, str]]:
    """Extract all dialogue pairs from the dialogue log"""
    try:
        # Get text after "=== Dialogue Log ==="
        dialogue_section = file_content.split("=== Dialogue Log ===")[1].strip()
        
        # Split into lines
        lines = [line.strip() for line in dialogue_section.split('\n') if line.strip()]
        
        pairs = []
        current_user = None
        current_assistant = None
        
        for line in lines:
            if line.startswith("User:"):
                if current_user is not None and current_assistant is not None:
                    pairs.append((current_user, current_assistant))
                current_user = line
                current_assistant = None
            elif line.startswith("Assistant:"):
                current_assistant = line
            elif current_assistant is not None:
                current_assistant += " " + line
        
        if current_user is not None and current_assistant is not None:
            pairs.append((current_user, current_assistant))
            
        if not pairs:
            raise ValueError("No dialogue pairs found")
            
        print(f"Extracted {len(pairs)} dialogue pairs")
        return pairs
        
    except Exception as e:
        print(f"Error extracting dialogue pairs: {str(e)}")
        return []

def format_dialogue_for_prompt(pairs: List[Tuple[str, str]]) -> str:
    """Format dialogue pairs for prompt"""
    formatted_dialogue = ""
    for i, (user, assistant) in enumerate(pairs, 1):
        formatted_dialogue += f"\n【ペア{i}】\n{user}\n{assistant}\n"
    return formatted_dialogue

def evaluate_dialogue(dialogue: str, ai_config: AIConfig) -> str:
    """Send API request to Claude to evaluate dialogue"""
    client = Anthropic(api_key=ai_config.api_key)
    
    # Get number of dialogue pairs
    pair_count = dialogue.count("【ペア")
    
    # Dynamically generate format part
    format_template = ""
    for i in range(1, pair_count + 1):
        format_template += f"""
【ペア{i}】
口調：
論理性：
コメント：
"""
    
    prompt = f"""
---

オープンソースのLLMをファインチューニングし、ソクラテス風のチャットボットを作ろうと考えています。
そのためにソクラテス風のAIアシスタントによる、会話データをこの度生成しました。
あなたにはその会話データの品質チェックをする役割を担って頂きます。

あなたにお見せする対話データは
Userが発言をし、それに対してAIアシスタント(Assistant)が問いをたてる、というものです。
その「User１発言」と「Assistant１発言」のペアを『{pair_count}個』お見せします。
尚、{pair_count}個のペアはそれぞれは独立したものとして考えてください。

あなたには主にそれぞれのペアについて以下の３点をやっていただきます。
①ソクラテスの口調になっているか、についての点数付け
②ソクラテスの返答は論理矛盾がなく、自然なものか、についての点数付け
③そのペアについて簡単なコメント

もう少しそれぞれについて解説をします。

①ソクラテスの口調になっているか、についての点数付け
この項目が一番大事です。
なぜなら今回ファインチューニング時に重視したい点が、
ソクラテスの口調であるためです。
例えば「かね？」「だろうか？」「ではないかね？」等の文末表現や
「友よ」「君」等の対話者への呼びかけ方等。
以下が点数の目安です。

0:まったくソクラテスではない
1:ソクラテスの要素が感じられない
2:悪くはないが少し気になる点がある
3:ソクラテスのような発言になっている
4:まさにソクラテス！申し分ない

②ソクラテスの返答は論理矛盾がなく、自然なものか、についての点数付け
今回はソクラテスの"問いを立てる力"には期待をしていません。
"深い問い"である必要は一切ありません。
あくまでソクラテスの"口調"で話すチャットボットを作るためです。
そのため、この項目は口調程大事ではないのですが、
さすがに論理的ではない発言や、会話の流れを見たときに自然ではないもの、は許容できないため、
この点についても評価をしてもらいます。
以下が目安です。

0:意味不明な発言をしてしまっている。
1:会話がかみ合ってない気がする
2:悪くはないが、少し気になる点がある
3:しっかりと自然な流れで会話ができている
4:非常に良い切り返し！さすがソクラテス！

③そのペアについて簡単なコメント
最後に評価対象のペアに対してのコメントを簡単にしてください。非常に簡潔で大丈夫です。長くしないでください。

尚、以下があなたに使用をしていただくフォーマットです。
「口調：」と「論理性：」のあとに以下のような形でそれぞれの点数を記述し、
「コメント：」の後にコメントをいれてください。
必ず{pair_count}個のペア全部に着手してください。

{format_template}

以下が{pair_count}個の会話ペアです。

{dialogue}
"""
    
    print("\n=== Sending Prompt to Claude ===")
    print(prompt)
    print("===============================\n")
    
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        messages=[{"role": "user", "content": prompt}]
    )
    
    evaluation = response.content[0].text
    print("\n=== Claude's Response ===")
    print(evaluation)
    print("========================\n")
    
    return evaluation

def is_already_evaluated(file_path: str) -> bool:
    """Check if a file has already been evaluated"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            return first_line == "=== Quality Evaluation ==="
    except Exception as e:
        print(f"Error checking evaluation status: {str(e)}")
        return False

def update_csv_with_evaluation(csv_path: str, file_name: str, evaluation: str):
    """Updates evaluation results in CSV file"""
    try:
        # Read CSV
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        # Parse evaluation and store in array
        pairs_data = []
        current_pair = {}
        pair_count = 0
        
        for line in evaluation.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
                
            if line.startswith("【ペア"):
                pair_count += 1
                if current_pair and len(current_pair) == 2:
                    pairs_data.append((current_pair['tone'], current_pair['logic']))
                current_pair = {}
            elif "口調：" in line:
                current_pair['tone'] = line.split("：")[1].strip()
            elif "論理性：" in line:
                current_pair['logic'] = line.split("：")[1].strip()
        
        # Add the last pair
        if current_pair and len(current_pair) == 2:
            pairs_data.append((current_pair['tone'], current_pair['logic']))
        
        # Update header
        if len(rows) > 0:
            header = rows[0]
            # Remove existing evaluation columns if present
            dialogue_index = header.index('dialogue')
            header[dialogue_index+1:] = []
            
            # Add evaluation columns based on pair count
            for i in range(1, pair_count + 1):
                header.extend([f"tone_pair{i}", f"logic_pair{i}"])
            
            print(f"Updated header with {pair_count} pairs")
        
        # Find matching row and update evaluation
        dialogue_col_index = rows[0].index('dialogue')
        for row in rows[1:]:
            if file_name == row[dialogue_col_index]:
                # Clear existing evaluation scores
                row[dialogue_col_index + 1:] = []
                # Add new evaluation
                for tone, logic in pairs_data:
                    row.extend([tone, logic])
                print(f"Updated row: {row}")  # Debug info
                break
        
        # Write back to CSV
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        
        print(f"CSV updated for file: {file_name}")
    
    except Exception as e:
        print(f"Error updating CSV for {file_name}: {str(e)}")
        print("Evaluation content:")
        print(evaluation)
        raise

def process_dialogue_file(file_path: str, ai_config: AIConfig, csv_path: str):
    """Process dialogue file and add evaluation results"""
    try:
        # If already evaluated, read existing evaluation
        if is_already_evaluated(file_path):
            print(f"File is already evaluated, reading existing evaluation: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Extract section between === Quality Evaluation === and === Dialogue Log ===
                evaluation = content.split("=== Quality Evaluation ===")[1].split("=== Dialogue Log ===")[0].strip()
                
            # Update CSV
            update_csv_with_evaluation(csv_path, os.path.basename(file_path), evaluation)
            return
            
        # Process new evaluation
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        pairs = extract_dialogue_pairs(content)
        formatted_dialogue = format_dialogue_for_prompt(pairs)
        evaluation = evaluate_dialogue(formatted_dialogue, ai_config)
        
        with open(file_path, 'r+', encoding='utf-8') as f:
            original_content = f.read()
            f.seek(0)
            f.write(f"=== Quality Evaluation ===\n{evaluation}\n\n{original_content}")
        
        # Update CSV
        update_csv_with_evaluation(csv_path, os.path.basename(file_path), evaluation)
        
        print(f"Successfully processed: {file_path}")
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        raise

def process_files_in_batches(dialogue_dir: str, ai_config: AIConfig, csv_path: str):
    """Process files in batches"""
    evaluated_files = set()
    try:
        with open(csv_path, 'r', encoding='utf-8', newline='') as f:
            # Check CSV content
            content = f.read()
            print("CSV file content preview:")
            print(content[:500])  # Display first 500 characters
            
            # Reset file pointer
            f.seek(0)
            
            # Read as CSV
            reader = csv.reader(f, delimiter=',', quotechar='"')
            rows = list(reader)
            
            if not rows:
                print("CSV file is empty")
                return
                
            if 'dialogue' not in rows[0]:
                print(f"Available columns: {rows[0]}")
                raise ValueError("dialogue column not found in CSV")
                
            dialogue_col_index = rows[0].index('dialogue')
            
            # Check data in each row
            for i, row in enumerate(rows[1:], 1):
                print(f"Row {i} length: {len(row)}")  # Number of columns in each row
                print(f"Row {i} content: {row}")  # Content of each row
                
                if len(row) > dialogue_col_index:  # Prevent index error
                    if row[dialogue_col_index] and any(cell.strip() for cell in row[dialogue_col_index + 1:]):
                        evaluated_files.add(row[dialogue_col_index])
    except Exception as e:
        print(f"Error reading CSV: {str(e)}")
        print(f"Full error details: ", e.__class__.__name__)
        return

    # Get list of files to process (unevaluated files only)
    files_to_process = []
    for filename in os.listdir(dialogue_dir):
        if filename.endswith(".txt") and filename not in evaluated_files:
            file_path = os.path.join(dialogue_dir, filename)
            files_to_process.append(file_path)
    
    if not files_to_process:
        print("No new files to process")
        return
        
    print(f"Found {len(files_to_process)} files to process")
    
    # Process in batches
    for i in range(0, len(files_to_process), BATCH_SIZE):
        batch = files_to_process[i:i + BATCH_SIZE]
        print(f"\nProcessing batch {i//BATCH_SIZE + 1}")
        
        for file_path in batch:
            try:
                print(f"\nProcessing file: {os.path.basename(file_path)}")
                process_dialogue_file(file_path, ai_config, csv_path)
            except (AlreadyEvaluatedError, ValueError) as e:
                print(f"Skipping file: {str(e)}")
                continue
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue

def move_low_rated_files(csv_path: str, dialogue_dir: str, low_rated_dir: str):
    """Move low-rated dialogue files to separate directory"""
    try:
        with open(csv_path, 'r', encoding='utf-8-sig') as f: 
            content = f.read()
            print("=== CSV File Content Preview ===")
            print(content[:1000])  
            print("===============================")
            
        with open(csv_path, 'r', encoding='utf-8-sig') as f:  
            reader = csv.reader(f)
            header = next(reader, None)
            
            if not header:
                print("Warning: CSV file is empty or has no header")
                return
                
            header = [col.replace('\ufeff', '') for col in header]
            print(f"CSV Headers after BOM removal: {header}")
            
            try:
                dialogue_index = header.index('dialogue')
            except ValueError:
                print("Warning: Required column 'dialogue' not found in headers")
                return
            
            rows = list(reader)
            print(f"Number of data rows: {len(rows)}")
            if len(rows) == 0:
                print("Warning: No data rows found in CSV")
                return
            
            f.seek(0)
            next(reader) 
            
            for row in rows:
                if len(row) < dialogue_index + 1:
                    print(f"Warning: Row too short, missing dialogue column: {row}")
                    continue
                
                try:
                    dialogue_file = row[dialogue_index]
                    
                    if not dialogue_file:
                        print("Warning: Missing dialogue filename in row")
                        continue

                    source_path = os.path.join(dialogue_dir, dialogue_file)
                    
                    if not os.path.exists(source_path):
                        print(f"File not found: {source_path}")
                        continue
                    
                    scores = []
                    try:
                        actual_turns = int(dialogue_file.split('_')[-1].split('.')[0])
                        expected_pairs = actual_turns  
                    except (IndexError, ValueError):
                        print(f"Warning: Could not extract turn count from filename: {dialogue_file}")
                        expected_pairs = None
                    
                    for i, col in enumerate(header):
                        if col.startswith(('tone_pair', 'logic_pair')):
                            if i < len(row) and row[i]:  
                                try:
                                    score = row[i].strip()
                                    if score and score.isdigit():
                                        scores.append(int(score))
                                except (AttributeError, ValueError) as e:
                                    print(f"Warning: Invalid score value for {col}: {row[i]}")
                                    continue
                    
                    if scores:
                        avg_score = sum(scores) / len(scores)
                        if expected_pairs:
                            actual_pairs = len(scores) // 2  
                            if actual_pairs < expected_pairs:
                                print(f"Warning: Missing scores for {dialogue_file}. Expected {expected_pairs} pairs, got {actual_pairs} pairs")
                        
                        if avg_score <= 2:
                            dest_path = os.path.join(low_rated_dir, dialogue_file)
                            shutil.move(source_path, dest_path)
                            print(f"Moved low-rated file: {dialogue_file} (avg_score: {avg_score:.2f})")
                    else:
                        print(f"Warning: No valid scores found for {dialogue_file}")
                        
                except Exception as e:
                    print(f"Error processing row: {e}")
                    continue
    
    except Exception as e:
        print(f"Error moving low-rated files: {str(e)}")
        import traceback
        traceback.print_exc()

def remove_parentheses_expressions(dialogue_dir: str):
    """Remove expressions enclosed in parentheses from dialogue logs"""
    try:
        for filename in os.listdir(dialogue_dir):
            if not filename.endswith('.txt'):
                continue
                
            file_path = os.path.join(dialogue_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Get the part after "=== Dialogue Log ==="
            parts = content.split("=== Dialogue Log ===")
            if len(parts) != 2:
                continue
                
            header = parts[0]
            dialogue = parts[1]
            
            # Remove expressions enclosed in parentheses
            cleaned_dialogue = re.sub(r'（[^）]*）', '', dialogue)
            
            # Update file if changes are made
            if cleaned_dialogue != dialogue:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(header + "=== Dialogue Log ===" + cleaned_dialogue)
                print(f"Removed parentheses expressions from: {filename}")
            
    except Exception as e:
        print(f"Error removing parentheses expressions: {str(e)}")

def analyze_evaluation_scores(csv_path: str):
    """Analyzes the distribution of tone and logic evaluation scores"""
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
            
            if len(rows) < 2:  # Requires header row + at least one data row
                raise ValueError("CSV file has insufficient data")
            
            header = rows[0]
            # Exclude empty rows
            data_rows = [row for row in rows[1:] if any(cell.strip() for cell in row)]
            
            print(f"Analyzing header: {header}")  # Debug info
            print(f"Number of data rows (excluding empty rows): {len(data_rows)}")  # Debug info
            
            # Get indices for tone and logic pair columns
            tone_indices = [i for i, col in enumerate(header) if col.startswith('tone_pair')]
            logic_indices = [i for i, col in enumerate(header) if col.startswith('logic_pair')]
            
            if not tone_indices or not logic_indices:
                raise ValueError("Required columns not found in CSV header")
            
            print(f"Found tone indices: {tone_indices}")  # Debug info
            print(f"Found logic indices: {logic_indices}")  # Debug info
            
            # Initialize counters
            tone_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
            logic_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
            total_tone = 0
            total_logic = 0
            expected_total = 0
            
            # Count evaluation scores and calculate expected total based on actual turns
            for row in data_rows:
                if 'dialogue' in header:
                    dialogue_col = header.index('dialogue')
                    if len(row) > dialogue_col and row[dialogue_col]:
                        # Extract turns from filename (e.g., "dialogue_test1_20250123_142856_2.txt" -> 2)
                        try:
                            turns = int(row[dialogue_col].split('_')[-1].split('.')[0])
                            expected_total += turns
                        except (IndexError, ValueError):
                            print(f"Warning: Could not extract turns from filename: {row[dialogue_col]}")
                            continue
                
                # Count tone pair scores
                for idx in tone_indices:
                    if idx < len(row) and row[idx].strip() and row[idx].isdigit():
                        score = int(row[idx])
                        tone_counts[score] = tone_counts.get(score, 0) + 1
                        total_tone += 1
                
                # Count logic pair scores
                for idx in logic_indices:
                    if idx < len(row) and row[idx].strip() and row[idx].isdigit():
                        score = int(row[idx])
                        logic_counts[score] = logic_counts.get(score, 0) + 1
                        total_logic += 1
            
            # Display results
            print("\n=== Evaluation Score Analysis ===")
            print(f"\nTotal dialogues: {len(data_rows)}")
            print(f"Expected total pairs (based on actual turns): {expected_total}")
            
            print("\nTone Score Distribution:")
            print(f"Total evaluated: {total_tone}")
            for score in range(0, 5):
                count = tone_counts[score]
                percentage = (count / total_tone * 100) if total_tone > 0 else 0
                total_percentage = (count / expected_total * 100) if expected_total > 0 else 0
                print(f"Score {score}: {count} (Valid: {percentage:.1f}%, Total: {total_percentage:.1f}%)")
            
            print("\nLogic Score Distribution:")
            print(f"Total evaluated: {total_logic}")
            for score in range(0, 5):
                count = logic_counts[score]
                percentage = (count / total_logic * 100) if total_logic > 0 else 0
                total_percentage = (count / expected_total * 100) if expected_total > 0 else 0
                print(f"Score {score}: {count} (Valid: {percentage:.1f}%, Total: {total_percentage:.1f}%)")
            
            # Calculate missing data based on actual turns
            missing_tone = expected_total - total_tone
            missing_logic = expected_total - total_logic
            
            print(f"\nMissing Data (based on actual dialogue turns):")
            print(f"Tone pairs: {missing_tone} ({(missing_tone/expected_total*100):.1f}% of expected {expected_total} pairs)")
            print(f"Logic pairs: {missing_logic} ({(missing_logic/expected_total*100):.1f}% of expected {expected_total} pairs)")
            
    except Exception as e:
        print(f"Error analyzing evaluation scores: {str(e)}")

def wait_for_csv_update(csv_path: str, max_retries: int = 3, wait_time: int = 1):
    """Wait for CSV file update"""
    for i in range(max_retries):
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if content.strip():
                    return True
            print(f"Waiting for CSV update (attempt {i+1}/{max_retries})")
            time.sleep(wait_time)
        except Exception as e:
            print(f"Error checking CSV (attempt {i+1}/{max_retries}): {str(e)}")
            time.sleep(wait_time)
    return False

def main():
    # Settings
    dialogue_dir = "data/dialogue/raw"
    low_rated_dir = "data/dialogue/low_rated"
    csv_path = "data/config/automation.csv"
    
    # AI configuration
    ai_config = AIConfig(
        api_key=API_KEY,
        model=MODEL_NAME
    )
    
    print("=== Starting Dialogue Evaluation Process ===")
    
    try:
        # First, evaluate files
        print("\n1. Evaluating new files")
        process_files_in_batches(dialogue_dir, ai_config, csv_path)
        
        # Wait for CSV update
        if not wait_for_csv_update(csv_path):
            print("Error: CSV file update could not be verified")
            return
            
        # After evaluation, run analysis
        print("\n2. Running evaluation score analysis")
        analyze_evaluation_scores(csv_path)
        
        # Finally, move low-rated files
        print("\n3. Processing low-rated files")
        os.makedirs(low_rated_dir, exist_ok=True)
        move_low_rated_files(csv_path, dialogue_dir, low_rated_dir)
        
    except FileNotFoundError:
        print(f"Error: Required files or directories not found")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
    
    print("\n=== Process Complete ===")

if __name__ == "__main__":
    main() 