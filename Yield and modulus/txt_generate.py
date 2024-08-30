import pandas as pd
import re


def parse_elements(value):
    if value.startswith('at_'):
        value = value[3:]  

    if value.startswith('Ti'):
        
        elements = re.findall(r'-(\d*\.?\d+)([A-Za-z]+)', value)
        result = {elem: float(amt) for amt, elem in elements}
        
        ti_content = 100 - sum(result.values())
        result['Ti'] = ti_content
        return result
    return value

df = pd.read_excel("/Users/wangping/Desktop/NLP-data/data.xlsx").iloc[:, 0:1]
df['parsed'] = df['eid'].apply(parse_elements)


element_order = ['Ti', 'Fe', 'Zr', 'Nb', 'Sn', 'Ta', 'Mo', 'Al', 'V', 'Cr', 'Hf', 'Mn', 'W', 'Si', 'Cu']


def reorder_and_format(parsed_dict):
    reordered_dict = {element: parsed_dict.get(element, 0) for element in element_order}

    #print(reordered_dict)
    with pd.ExcelWriter("/Users/wangping/Desktop/NLP-data/data2.xlsx", engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
        pd.DataFrame([reordered_dict]).to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)
    formatted_string = '-'.join(f'{reordered_dict[element]}{element}' for element in element_order)

    return formatted_string

with pd.ExcelWriter("/Users/wangping/Desktop/NLP-data/data2.xlsx", engine='openpyxl', mode='w') as writer:
   pd.DataFrame(columns=['Ti', 'Fe', 'Zr', 'Nb', 'Sn', 'Ta', 'Mo', 'Al', 'V', 'Cr', 'Hf', 'Mn', 'W', 'Si', 'Cu']).to_excel(writer, index=False)


df['parsed_reordered'] = df['parsed'].apply(reorder_and_format)


file_path = "/Users/wangping/Desktop/NLP-data/217samples-217alloys.txt"
df['parsed_reordered'].to_csv(file_path, index=False, header=False, sep='\n')