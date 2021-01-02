import json 
import re 
from sklearn.model_selection import train_test_split

def write_file(file_name, source, target):
    with open(file_name, 'w+') as train_f:
        for i in range(len(source)):
            train_f.write('{},{}\n'.format(source[i], target[i]))

if __name__ == '__main__':
    with open('github-typo-corpus.v1.0.0.jsonl') as f:
        data_list = list(f)
        
    clean_data = []

    for json_str in data_list:
        typo_dict = json.loads(json_str)
        for edit in typo_dict['edits']:
            if edit['src']['lang'] == 'eng' and edit['is_typo'] == True:
                source = edit['src']['text']
                target = edit['tgt']['text']
                source = re.sub('<.+?>', '', source)
                target = re.sub('<.+?>', '', target)
                source = re.sub("[^a-zA-Z'. ]+", ' ', source)
                target = re.sub("[^a-zA-Z'. ]+", ' ', target)
                source = re.sub('\s+', ' ', source).lower().strip()
                target = re.sub('\s+', ' ', target).lower().strip()
                if len(source) > 30:
                    clean_data.append((source, target))

    source_train, source_val, target_train, target_val = train_test_split([x[0] for x in clean_data], [x[1] for x in clean_data], test_size=0.08, random_state=0)
    source_val, source_test, target_val, target_test = train_test_split(source_val, target_val, test_size=0.2, random_state=0) 

    write_file('github_train.txt', source_train, target_train)
    write_file('github_val.txt', source_val, target_val)
    write_file('github_test.txt', source_test, target_test)
