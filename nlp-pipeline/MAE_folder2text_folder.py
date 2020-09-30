import preprocess_mae as mae
import os

def extract_text_():
    input_folder_path = input("Type in the folder path:")
    parent=os.path.abspath(os.path.dirname(input_folder_path))
    output_foler_path = (parent+"/extracted_text")
    os.makedirs(output_foler_path)
    input_folder_dir = os.listdir(input_folder_path)
    for file in input_folder_dir:
        portion=os.path.splitext(file)
        file_path = input_folder_path+"/"+file

        with open(file_path) as f:
            text,label,=mae.separate_xml_from_text(file_path)
        output_file_path=output_foler_path+"/"+portion[0]+".txt"
        print(output_file_path)
        f.close()

        output_file=open(output_file_path,mode="a")
        output_file.writelines(text)
        output_file.close()

if __name__ == "__main__":
    extract_text_()