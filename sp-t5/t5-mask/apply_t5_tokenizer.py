import json
from transformers import PreTrainedTokenizerFast

fileName = "./t5-mask/opinion_keywords_for_masking.txt"
tokenizer = PreTrainedTokenizerFast(tokenizer_file="sp-t5/tokenizer.json")

with open(fileName, 'r') as file:
    token_list = [list(filter(lambda x: x not in [3, 1],
                              tokenizer.encode(l.strip())))
                  for l in file]
    decoded_token_list = [tokenizer.decode(l) for l in token_list]
    print(decoded_token_list[:10])
    print(token_list[:10])
    
    data = {
        "to_mask": token_list,
        "_mode": "original",
        "mode": "keyword"
    }

    with open('t5-mask/mask.json', 'w') as json_file:
        json.dump(data, json_file, indent=2)