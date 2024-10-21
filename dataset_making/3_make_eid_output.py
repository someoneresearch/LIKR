import os

def load_metadata(file_path):
    metadata_map = {}
    with open(file_path, 'r') as f:
        for line in f:
            new_id, eid = line.strip().split('\t')
            metadata_map[eid] = new_id
    return metadata_map

def replace_ids(input_folder, output_folder, metadata_folder):
    metadata_types = ['actor', 'cinematographer', 'composer', 'country', 
                      'director', 'editor', 'prodcompany', 'producer', 'writter']
    
    metadata_maps = {}
    for mtype in metadata_types:
        metadata_file = os.path.join(metadata_folder, f"{mtype}.txt")
        metadata_maps[mtype] = load_metadata(metadata_file)

    for filename in os.listdir(input_folder):
        input_file = os.path.join(input_folder, filename)
        output_file = os.path.join(output_folder, filename)
        
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            for line in infile:
                for mtype in metadata_types:
                    if f"{mtype}:" in line:
                        eid = line.strip().split(': ')[1]
                        if eid in metadata_maps[mtype]:
                            new_id = metadata_maps[mtype][eid]
                            line = line.replace(eid, new_id)
                outfile.write(line)

input_folder = 'eid_output'
output_folder = 'newid_output'
metadata_folder = 'metadata_index'

os.makedirs(output_folder, exist_ok=True)

replace_ids(input_folder, output_folder, metadata_folder)
