import os
import preprocess_mae as mae
import segm_prep_utils as spu
import requests
from allennlp.data.tokenizers.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer
from allennlp.data.tokenizers.character_tokenizer import CharacterTokenizer

def transform_folder_to_list(folder_path):
    output=[]
    folder_dir=os.listdir(folder_path)
    for file_name in folder_dir:
        if file_name.endswith("xml"):
            print(file_name)
            dic={}
            file_path=folder_path+"/"+file_name
            dic=convert_mae(file_path)
            output.append(dic)
    return output

def convert_mae(file_path):
    li = []
    dic = {}
    text, tags, = mae.separate_xml_from_text(file_path)
    dic["text"] = "".join(text).lower()
    for tag in tags:
        words = tag.split(" ")
        start_pos=tag.find('start=')
        end_pos=tag.find('end=')
        text_pos=tag.find('text=')
        type_pos=tag.find('TYPE=')
        comment_pos=tag.find('comment=')
        label = words[0][1:]
        start=int(tag[start_pos+7:end_pos-2])
        end=int(tag[end_pos+5:text_pos-2])
        text=tag[text_pos+6:type_pos-2]
        label = label+'-'+tag[type_pos+6:comment_pos-2]
        # for word in words:
        #     if word.startswith("TYPE="):
        #         detail = word[6:-1]
        #         if label != detail:
        #             label = label + '-' + detail
        #         continue
        #     if word.startswith("start="):
        #         start = int(word[7:-1])
        #         continue
        #     if word.startswith("end="):
        #         end = int(word[5:-1])
        #         continue
        #     if word.startswith("text="):
        #         text = word[6:-1]
        #         print(text)
        temp_dic = {"label": label.lower(), "start": start, "end": end, "text": text.lower()}
        li.append(temp_dic)
    dic["seq_annotations"] = li
    return dic


if __name__=="__main__":

    transform_folder_to_list
    file="/Users/laynewei/Desktop/UCSF_intern/Data/training-PHI-Gold-Set1/280-01.xml"
    document=convert_mae(file)
    #tokenizers
    spacy_tokenizer = spu.get_spacy_tokenizer(special_cases_file='special-cases-spacy.yaml')
    pre_trained_tokenizer=PretrainedTransformerTokenizer
    character_tokenizer=CharacterTokenizer

    # slow process
    convert = lambda x: spu.annotation_json2septext(
        ' '+x['text'], x['seq_annotations'],
        sep='#%#',
        #tokenize=pre_trained_tokenizer,
        tokenize=spacy_tokenizer,
        # tokenize=CharacterTokenizer,
        sep_label='punctuation',
        start='start',
        end='end',
    ).lower()
    result=convert(document)
    splitted_result=result.split(' ')