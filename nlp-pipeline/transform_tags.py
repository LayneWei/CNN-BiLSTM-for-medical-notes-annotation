# import format_text_segmentation as fts
def transform():
    file_name= input("type in the file name")
    file = open(file_name,"r")
    lines=file.readlines()
    output=[]
    line_number=0
    for i,l in enumerate(lines):
        if "<TAGS>" in l:
            xml_split_line = i+1
            break
    for line in lines[xml_split_line:]:
        if "</TAGS>" in line:
            break
        words=line.split(" ")
        label=words[0][1:]
        for word in words:
            if word.startswith("TYPE="):
                detail=word[6:-1]
                if label!=detail:
                    label=label+'-'+detail
                continue
            if word.startswith("start="):
                start= int(word[7:-1])
                continue
            if word.startswith("end="):
                end= int(word[5:-1])
                continue
            if word.startswith("text="):
                text=word[6:]
        dic={"label":label,"start":start,"end":end,"text":text}
        output.append(dic)
    # for label in output:
    #     print(label)
    final_output=[]
    exist=set()
    for label in output:
        text=label['text']
        if text not in exist:
            dic={}
            seq_annotation=[]
            exist.add(text)
            dic['text']=text
            temp_dic={"start":label["start"],"end":label["end"],"label":label["label"]}
            seq_annotation.append(temp_dic)
            dic["seq_annotations"]=seq_annotation
            final_output.append(dic)
        else:
            for final_label in final_output:
                if final_label["text"]==text:
                    temp_dic = {"start": label["start"], "end": label["end"], "label": label["label"]}
                    final_label["seq_annotations"].append(temp_dic)
    for final_label in final_output:
        print(final_label)
    # for final_label in final_output:
    #     fts.convert(final_output)


if __name__=="__main__":
    transform()


