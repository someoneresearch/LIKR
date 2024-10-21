import openai
import time
import os
import re
from tqdm import tqdm

api_base = "https://api.openai.com/v1"
api_key = ""  # use your API key

openai.api_base = api_base
openai.api_key = api_key

def parse_line(line):
    line = line.strip()
    if not line:
        return None, None
    parts = line.split('\t')
    user_id = parts[0]
    movies = parts[1:]
    return user_id, movies

output_folder = 'llmoutput'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

with open('inputed_title_train.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for line in tqdm(lines, desc="Processing users", unit="user"):
    user_id, movies = parse_line(line)
    if user_id is None:
        continue
    prompt = f'''I am thinking of recommending the next movie for a user to watch. The user has watched the following movies in this order, with the more recently watched movies appearing at the end:

{' '.join(movies)}

Based on the metadata types such as [actor], [cinematographer], [composer], [country], [director], [editor], [prodcompany], [producer], and [writer], please choose the metadata that is especially important, and provide the elements that the next movie should have for this user.

Please format the output as follows and provide it as a text file:

Output format
[metadata]: element

Example output
[actor]: Will_Smith
[actor]: Humphrey_Bogart
[director]: James_Whale
[director]: Billy_Wilder

Please also tell us the reasons.
'''

    try:
        response = openai.ChatCompletion.create(
            model="o1-preview",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        reply = response['choices'][0]['message']['content']

        metadata_lines = []
        for line in reply.split('\n'):
            line = line.strip()
            if re.match(r'^\[.+\]: .+', line):
                metadata_lines.append(line)

        output_filename = os.path.join(output_folder, f'user_{user_id}_output.txt')
        with open(output_filename, 'w', encoding='utf-8') as f_out:
            f_out.write('\n'.join(metadata_lines))
    except Exception as e:
        print(f"\nError processing user {user_id}: {e}")
    time.sleep(1)
