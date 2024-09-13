import json
import argparse

def convert_vocab_to_json(input_file, output_file):
    vocab = {"<s>": 0,"<pad>": 1,"</s>": 2,"<unk>": 3}
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            word, freq = line.strip().split()
            vocab[word] = len(vocab.keys())
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--voc_text_file", type=str, help="Path to the fairseq vocabulary file in text format (e.g.spm_unigram10000.txt).")
    parser.add_argument("--voc_json_file", type=str, help="Path to the output vocabulary file in json format (e.g.spm_unigram10000.json).")
    args = parser.parse_args()
    convert_vocab_to_json(args.voc_text_file,args.voc_json_file)