from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
import json
import os
from transformers import BertTokenizer
import os
from scapy.all import rdpcap
import binascii
import json

def json_to_txt(json_path, txt_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)  # Read the JSON list

    with open(txt_path, 'w', encoding='utf-8') as f:
        for line in data:
            f.write(line.strip() + '\n')  # Write one element per line

    print(f"✅ Successfully converted {json_path} to {txt_path}")

def build_BPE(tokenizer_path,corpora_path):
    # generate source dictionary,0-65535
    num_count = 65536
    not_change_string_count = 5
    i = 0
    source_dictionary = {}
    tuple_sep = ()
    tuple_cls = ()
    #'PAD':0,'UNK':1,'CLS':2,'SEP':3,'MASK':4
    while i < num_count:
        temp_string = '{:04x}'.format(i)
        source_dictionary[temp_string] = i
        i += 1
    # Initialize a tokenizer
    tokenizer = Tokenizer(models.WordPiece(vocab=source_dictionary,unk_token="[UNK]",max_input_chars_per_word=4))

    # Customize pre-tokenization and decoding
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tokenizer.decoder = decoders.WordPiece()
    tokenizer.post_processor = processors.BertProcessing(sep=("[SEP]",1),cls=('[CLS]',2))

    # And then train
    trainer = trainers.WordPieceTrainer(vocab_size=65536, min_frequency=2)
    tokenizer.train([corpora_path, corpora_path], trainer=trainer)

    # And Save it
    tokenizer.save(tokenizer_path, pretty=True)
    return 0

def build_vocab(tokenizer_path,vocab_path):
    json_file = open(tokenizer_path,'r')
    json_content = json_file.read()
    json_file.close()
    vocab_json = json.loads(json_content)
    vocab_txt = ["[PAD]","[SEP]","[CLS]","[UNK]","[MASK]"]
    for item in vocab_json['model']['vocab']:
        vocab_txt.append(item) # append key of vocab_json
    with open(vocab_path,'w') as f:
        for word in vocab_txt:
            f.write(word+"\n")
    return 0

tokenizer_path ="C:\\Users\\Administrator\\bertgnn\\vocab\\bert_tokenizer\\wordpiece.tokenizer.json"
bert_tokenizer_path = "C:\\Users\\Administrator\\bertgnn\\vocab\\bert_tokenizer"
pretrain_data_path = "bert_pretrain_payloads.json"
corpora_path="C:\\Users\\Administrator\\bertgnn\\vocab\\vocab_train_data.txt"
vocab_path="C:\\Users\\Administrator\\bertgnn\\vocab\\bert_tokenizer\\vocab.txt"
# Example call
json_to_txt('bert_pretrain_payloads.json', corpora_path)
build_BPE(tokenizer_path,corpora_path)
build_vocab(tokenizer_path,vocab_path)

# Create BertTokenizer
tokenizer = BertTokenizer(vocab_file=vocab_path)
tokenizer.save_pretrained(bert_tokenizer_path)
tokenizer = BertTokenizer.from_pretrained(bert_tokenizer_path)


# —————————— Test the tokenizer ——————————————
def cut(obj, sec):
    result = [obj[i:i+sec] for i in range(0,len(obj),sec)]
    remanent_count = len(result[0])%4
    if remanent_count == 0:
        pass
    else:
        result = [obj[i:i+sec+remanent_count] for i in range(0,len(obj),sec+remanent_count)]
    return result


def bigram_generation(packet_string,flag=False):
    result = ''
    sentence = cut(packet_string,1)
    token_count = 0
    for sub_string_index in range(len(sentence)):
        if sub_string_index != (len(sentence) - 1):
            token_count += 1
            if token_count > 256:
                break
            else:
                merge_word_bigram = sentence[sub_string_index] + sentence[sub_string_index + 1]
        else:
            break
        result += merge_word_bigram
        result += ' '
    if flag == True:
        result = result.rstrip()

    return result

def process_pcap_file(pcap_path):
    payloads=[]
    packets = rdpcap(pcap_path)[0:10]
    for packet in packets:
        # Process only TCP or UDP packets
        if packet.haslayer('TCP') or packet.haslayer('UDP'):
            payload = bytes(packet.payload)
            payload = binascii.hexlify(payload).decode('utf-8')
            if payload:
                payloads.append(bigram_generation(payload))
    return payloads
# Test if the tokenizer works well
payloads=process_pcap_file("D:\\wmx\\ISCX-VPN-Service\\VPN\\Chat\\vpn_hangouts_chat1a.pcap")

# Example input: select an extracted payload for generation
input_text = payloads[3]  # Use the first payload data
print(payloads[3])
print(tokenizer.tokenize(input_text))

# Use the trained tokenizer to encode the input
encoded = tokenizer.encode(input_text,
                           add_special_tokens=True,
                                truncation=True,
                                padding='max_length',
                                max_length=512,
                                return_tensors='pt')     # Return PyTorch tensor, default is a list
print(encoded)
