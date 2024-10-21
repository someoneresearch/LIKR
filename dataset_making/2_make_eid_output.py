import os
import pandas as pd
from tqdm import tqdm

metadata_folder = './metadata/'
folder_path = './llmoutput'
output_folder = './eid_output/'

metadata_files = {
    'actor': os.path.join(metadata_folder, 'actor.txt'),
    'cinematographer': os.path.join(metadata_folder, 'cinematographer.txt'),
    'composer': os.path.join(metadata_folder, 'composer.txt'),
    'country': os.path.join(metadata_folder, 'country.txt'),
    'director': os.path.join(metadata_folder, 'director.txt'),
    'editor': os.path.join(metadata_folder, 'editor.txt'),
    'prodcompany': os.path.join(metadata_folder, 'prodcompany.txt'),
    'producer': os.path.join(metadata_folder, 'producer.txt'),
    'writter': os.path.join(metadata_folder, 'writter.txt')
}

os.makedirs(output_folder, exist_ok=True)

user_metadata = {}

file_list = [filename for filename in os.listdir(folder_path) if filename.startswith('user_') and filename.endswith('_output.txt')]

for filename in tqdm(file_list, desc="Processing user files"):
    user_id = filename.split('_')[1]
    output_file = f'user_{user_id}.txt'
    
    user_metadata = {}
    
    with open(os.path.join(folder_path, filename), 'r') as f:
        for line in f:
            if line.strip():
                try:
                    key, value = line.strip().replace('[', '').replace(']', '').split(': ')
                    key = key.lower()
                    if key in user_metadata:
                        user_metadata[key].append(value)
                    else:
                        user_metadata[key] = [value]
                except ValueError:
                    continue

    result_set = set()

    for metadata_type, file_path in metadata_files.items():
        if metadata_type in user_metadata:
            df = pd.read_csv(file_path, delimiter='\t')
            
            for metadata_value in user_metadata[metadata_type]:
                metadata_value_normalized_space = metadata_value.replace('_', ' ')
                metadata_value_normalized_underscore = metadata_value.replace(' ', '_')
                
                match_space = df[df['name'] == metadata_value_normalized_space]
                match_underscore = df[df['name'] == metadata_value_normalized_underscore]
                
                if not match_space.empty:
                    eid = match_space['eid'].values[0]
                    result_set.add(f"{metadata_type}: {eid}")
                elif not match_underscore.empty:
                    eid = match_underscore['eid'].values[0]
                    result_set.add(f"{metadata_type}: {eid}")

    with open(os.path.join(output_folder, output_file), 'w') as f:
        for line in result_set:
            f.write(f"{line}\n")
