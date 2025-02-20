from anthropic import Anthropic
from typing import Dict, List, Any
import json
import os
from pathlib import Path
import time
from src.utils.config import get_api_keys

# Model parameters
MODEL_NAME = "claude-3-5-sonnet-20241022"
MAX_TOKENS = 4000
TEMPERATURE = 0.3

# Get API key
api_keys = get_api_keys()
API_KEY = api_keys['claude_api_key_quality2']

def create_evaluation_prompt(dialogue_pairs: List[Dict[str, Any]]) -> str:
    """Create evaluation prompt for Claude"""
    
    # Format dialogue pairs for prompt
    formatted_pairs = ""
    for i, pair in enumerate(dialogue_pairs, 1):
        formatted_pairs += f"\n【ペア{i}】\nUser: {pair['claude']['content']}\nAssistant: {pair['gemma']['content']}\n"

    prompt = f"""
オープンソースのLLMをファインチューニングし、ソクラテス風のチャットボットを作りました。
あなたにはこのチャットボットの応答の品質をチェックしてもらいたいと思います。

あなたにお見せする対話データは
Userが発言をし、それに対してAIアシスタント(Assistant)が応答する、というものです。
その「User１発言」と「Assistant１発言」のペアを『{len(dialogue_pairs)}個』お見せします。
尚、{len(dialogue_pairs)}個のペアはそれぞれは独立したものとして考えてください。

あなたには主にそれぞれのペアについて以下の4つの軸から評価をやっていただきます。

１）ソクラテス的な口調
２）ソクラテス的な対話姿勢
３）形式的正確性
４）対話の自然さ・論理性

尚、それぞれの評価項目については点数をつけてもらいます。
以下がそれぞれの項目の詳細な解説および点数をつける際の目安です。

１）ソクラテス的な口調
応答が古代ギリシャの賢人ソクラテスらしい老練な話し方をしているかを評価します。具体的には「〜かね」「〜だね」「～かな？」「～だろうか？」などの特徴的な文末表現や、「友よ」「君」等の対話者への呼びかけ方等、口調に注目をしてください。丁寧語はソクラテスらしい口調には該当しないので注意してください。

スコアの目安
0: ソクラテスの口調の要素が皆無
1: 丁寧語を使ってしまっていたり、ソクラテス口調から外れてしまっている
2: 悪くはないが少し気になる点がある
3: ソクラテスのような口調が保たれている
4: まさにソクラテスが話しそうな口調！申し分ない

２）ソクラテス的な対話姿勢
直接的な答えを与えるのではなく、相手に問いたてたりすることによって、自発的な思考を促しているかを評価してください。

スコアの目安
0: 一方的な説明や断定的な回答のみ等、まったくあてはまらない
1: 相手に思考を促す、という観点で弱い
2: 相手に思考を促す姿勢は感じるが少し気になる点がある
3: 相手に思考を促せてる
4: 相手に思考を促せてるだけではなく、問いの立て方に非常にセンスを感じる

３）形式的正確性
日本語以外の言語（英語など）、不要な記号、絵文字、異常な改行、余分なスペース、HTMLタグの混入などがないかを評価してください。

スコアの目安
0: 多数の形式的問題が混在している
1: 明らかな形式的問題が１～２個ある（外国語の混入やHTMLタグなど）
2: 軽微な形式的問題がある（余分なスペース、微妙な改行位置など）
3: 完璧かどうかはおいておいて問題とされるほどのものは見当たらない
4: 一切の崩れや異常がなく、完璧な形式

４）対話の自然さ・論理性
文法的な正確さ、論理の一貫性、そして会話の流れの自然さを評価します。違和感なく対話が進んでいるか、また相手の発言に対して適切に応答できているかを確認してください

スコアの目安
0:意味不明な発言をしてしまっている
1:会話がかみ合ってない気がする
2:悪くはないが、少し気になる点がある
3:しっかりと自然な流れで会話ができている
4:非常に良い切り返し！さすがソクラテス！

そして、その点数を付けた際に、なぜその点数を付けたのかもコメントがほしいです。ただ非常に完結にしてください。

尚、以下があなたに使用をしていただくフォーマットです。
「スコア：」のあとに数字を記述し、
「コメント：」の後に簡潔なコメントをいれてください。
必ず{len(dialogue_pairs)}個のペア全部に着手してください。

【ペア1】
■ソクラテス的な口調
スコア：
コメント：

■ソクラテス的な対話姿勢
スコア：
コメント：

■形式的正確性
スコア：
コメント：

■対話の自然さ・論理性
スコア：
コメント：

【ペア2】
■ソクラテス的な口調
スコア：
コメント：

■ソクラテス的な対話姿勢
スコア：
コメント：

■形式的正確性
スコア：
コメント：

■対話の自然さ・論理性
スコア：
コメント：
...

以下が評価対象となる{len(dialogue_pairs)}個の対話ペアです。

{formatted_pairs}

"""
    return prompt

def parse_evaluation_response(response: str) -> List[Dict[str, Dict[str, Dict[str, str]]]]:
    """Parse Claude's evaluation response into structured format"""
    evaluations = []
    current_pair = None
    current_category = None
    
    for line in response.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('【ペア'):
            if current_pair is not None:
                evaluations.append(current_pair)
            current_pair = {
                "tone": {"quantitative": "", "qualitative": ""},
                "approach": {"quantitative": "", "qualitative": ""},
                "format": {"quantitative": "", "qualitative": ""},
                "logic": {"quantitative": "", "qualitative": ""}
            }
            
        elif line.startswith('■ソクラテス的な口調'):
            current_category = "tone"
        elif line.startswith('■ソクラテス的な対話姿勢'):
            current_category = "approach"
        elif line.startswith('■形式的正確性'):
            current_category = "format"
        elif line.startswith('■対話の自然さ・論理性'):
            current_category = "logic"
            
        elif line.startswith('スコア：') and current_pair and current_category:
            score = line.split('：')[1].strip()
            current_pair[current_category]["quantitative"] = score
        elif line.startswith('コメント：') and current_pair and current_category:
            comment = line.split('：')[1].strip()
            current_pair[current_category]["qualitative"] = comment
            
    if current_pair is not None:
        evaluations.append(current_pair)
        
    return evaluations

def update_csv_with_evaluation(csv_path: str, filename: str, evaluations: List[Dict]) -> None:
    """Update CSV file with evaluation scores"""
    import csv
    import pandas as pd
    
    try:
        # Read existing CSV
        df = pd.read_csv(csv_path)
        
        # Find the row for this dialogue file
        row_idx = df[df['dialogue'] == filename].index
        
        if len(row_idx) == 0:
            print(f"Warning: Dialogue file {filename} not found in CSV")
            return
            
        # Get number of pairs
        num_pairs = len(evaluations)
        
        # Create column names if they don't exist
        for pair_num in range(1, num_pairs + 1):
            for metric in ['tone', 'approach', 'format', 'logic']:
                col_name = f"{metric}_pair{pair_num}"
                if col_name not in df.columns:
                    # Initialize new columns as float type
                    df[col_name] = pd.Series(dtype='float64')
        
        # Update scores
        for pair_num, evaluation in enumerate(evaluations, 1):
            for metric in ['tone', 'approach', 'format', 'logic']:
                col_name = f"{metric}_pair{pair_num}"
                # Convert score to float before assignment
                try:
                    score = float(evaluation[metric]['quantitative'])
                    df.loc[row_idx, col_name] = score
                except (ValueError, TypeError) as e:
                    print(f"Warning: Could not convert score to float for {col_name}: {evaluation[metric]['quantitative']}")
                    continue
        
        # Save updated CSV
        df.to_csv(csv_path, index=False)
        print(f"Updated CSV for {filename}")
        
    except Exception as e:
        print(f"Error updating CSV for {filename}: {str(e)}")
        raise

def evaluate_dialogue_file(file_path: str) -> None:
    """Evaluate a single dialogue file"""
    try:
        # Read dialogue file
        with open(file_path, 'r', encoding='utf-8') as f:
            dialogue_data = json.load(f)
            
        # Create Claude client
        client = Anthropic(api_key=API_KEY)
        
        # Generate evaluation prompt
        prompt = create_evaluation_prompt(dialogue_data['pairs'])
        
        # Get evaluation from Claude
        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Parse evaluation response
        evaluations = parse_evaluation_response(response.content[0].text)
        
        # Update dialogue data with evaluations
        for pair, evaluation in zip(dialogue_data['pairs'], evaluations):
            pair['evaluation'] = evaluation
            
        # Write updated dialogue data back to file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(dialogue_data, f, ensure_ascii=False, indent=2)
            
        # Update CSV file
        csv_path = "data/config/automation_gemma.csv"
        filename = os.path.basename(file_path)
        update_csv_with_evaluation(csv_path, filename, evaluations)
            
        print(f"Successfully evaluated: {file_path}")
        
    except Exception as e:
        print(f"Error evaluating {file_path}: {str(e)}")
        raise

def process_dialogue_directory(directory: str) -> None:
    """Process all dialogue files in directory"""
    try:
        for filename in os.listdir(directory):
            if filename.endswith('.json'):
                file_path = os.path.join(directory, filename)
                
                # Check if file already has evaluations
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if all('evaluation' in pair and 
                          all(pair['evaluation'][key]['quantitative'] 
                              for key in ['tone', 'approach', 'format', 'logic'])
                          for pair in data['pairs']):
                        print(f"Skipping already evaluated file: {filename}")
                        continue
                
                print(f"\nProcessing: {filename}")
                evaluate_dialogue_file(file_path)
                time.sleep(1)  # Rate limiting
                
    except Exception as e:
        print(f"Error processing directory: {str(e)}")
        raise

def main():
    dialogue_dir = "data/dialogue/raw_gemma"
    process_dialogue_directory(dialogue_dir)

if __name__ == "__main__":
    main()
