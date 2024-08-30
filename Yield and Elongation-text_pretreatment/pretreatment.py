import re

raw_file = '/Users/wangping/Desktop/pythonProject/text_pretreatment/scopus.txt'
pretreatment_file = '/Users/wangping/Desktop/pythonProject/text_pretreatment/raw_input_file.txt'
with open(raw_file) as file1:
    content = file1.read()

texts = re.findall(r'[\u4e00-\u9fa5]+: .* ©', content)  

with open(pretreatment_file, "w") as file2:
    for abstract in texts:
        abstract = abstract[4:-3].split('. ')
        for sentence in abstract:
            file2.write(sentence + '.' + '\n')
        file2.write('\n')
