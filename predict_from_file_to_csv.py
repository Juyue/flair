import os
from flair.models import SequenceTagger
from flair.data import Sentence
import csv
data_folder = '/home/jupyter/data_ner/'
TAG_Dict = {'PER','LOC','MIS','ORG'}
def from_model_to_solution(model_folder):
    model_file = os.path.join(model_folder, 'final-model.pt')
    model = SequenceTagger.load_from_file(model_file)
    
    print("load model okay")
    file_str_bank = ['dev', 'test']
    for file_str in file_str_bank:
        raw_file = file_str + '_raw.txt'
        file_prediction = os.path.join(data_folder, raw_file)
        text_sentence = from_file_to_text(file_prediction)
        
        tmp_file = file_str + '_tmp.txt'
        from_text_to_tmp_file(model, text_sentence, tmp_file)

        output_file = file_str + '_output.csv'
        from_tmp_file_to_output_file_v2(tmp_file, output_file, file_str+'.txt')
    
# string processing. skip the \n. use the 
def from_file_to_text(filename):
    text_sentence = []
    with open(filename, 'r', encoding = "utf-8") as f:
        sent = []
        for ww in f.readlines():
            if ww == '.' or ww == '\n' or ww == '.\n':
                if sent:
                    sent.append('.')
                    text_sentence.append(sent)
                sent = []
                continue
            tmp,_ = ww.split('\n')
            if tmp == '..':
                tmp = '.'
            sent.append(tmp)
    return text_sentence


# 10 create example sentence. load the dev and est file. predict those sentences..
def from_text_to_tmp_file(model, text_sentence, tmp_file):
    output_str = []        
    for t_s in text_sentence:
        # change into sentence format.
        sentence = Sentence(' '.join(t_s), use_tokenizer=True)
        
        # 11 predict tags and print
        model.predict(sentence)
        
        # create a string of current tagging result.
        text_string = [tt.text + ' ' + tt.get_tag('ner').value + ' {:f}'.format(tt.get_tag('ner').score) for tt in sentence.tokens]
        text_string = '\n'.join(text_string) + '\n' + '\n'
        output_str.append(text_string)
        
    # 12 write the result into a file.
    with open(tmp_file, 'w', encoding = 'utf-8') as f:
            f.writelines(output_str)

# utilize the merge_result_lines.py...
def from_tmp_file_to_output_file_v2(tmp_file, output_file, f_name_str):

    f_input = open(tmp_file, 'r', encoding='utf-8')
    f_output =  open(output_file, 'w', encoding='utf-8', newline='\n')
    
    for ii, line in enumerate(f_input.readlines()):
        # you have to loop through until there
        # read it. if it is 'O' skip
        if line == '\n':
            continue
        word, tag, score = line.split()
        if tag == 'O':
            continue
        else:
            # write it.
            lines_to_write = '{},{},{},{:.2f},{},\"{}\"\n'.format(f_name_str,ii + 1, ii + 1,float(score), tag, word)
            f_output.write(lines_to_write)    
    print(lines_to_write)
    f_output.close()
    f_input.close()

# def from_tmp_file_to_output_file(tmp_file, output_file, f_name_str):

#     f_input = open(tmp_file, 'r', encoding='utf-8')
#     f_output =  open(output_file, 'w', encoding='utf-8', newline='\n')
#     f_output.write('Filename,Start,End,Type,Score,Surface\n')
#     # csv_writer = csv.writer(f_output, delimiter=',', newline='\n', quoting=csv.QUOTE_NONE)
#     # csv_writer.writerow(['Filename', 'Start', 'End', 'Type', 'Surface'])
    
#     start = None
#     tag_type = None
#     phrase = []
    
#     for ii, line in enumerate(f_input.readlines()):
#         # you have to loop through until there
#         # read it. if it is 'O' skip
#         if line == '\n':
#             continue
#         word, tag, score = line.split()
#         if tag == 'O':
#             if start is None:
#                 continue
#             else:
#                 # could output. you should..
#                 if not (tag_type in TAG_Dict):
#                     tag_type = 'MIS' 
#                 lines_to_write = '{},{},{},{},{},"{}"\n'.format(f_name_str, start + 1, ii, tag_type, score, ' '.join(phrase))
#                 f_output.write(lines_to_write)
#                 start, tag_type, word = None, None, []
#         else:
#             tag_init, tag_type_curr = tag.split('-')
#             if (tag_init == 'B'):
#                 if start is not None:
#                     if not (tag_type in TAG_Dict):
#                         tag_type = 'MIS' 
#                     lines_to_write = '{},{},{},{},{},"{}"\n'.format(f_name_str, start + 1, ii, tag_type, score, ' '.join(phrase))
#                     f_output.write(lines_to_write)
#                 start, tag_type, phrase = ii, tag_type_curr, []
#                 phrase.append(word)
#             else:
#                 if start is None:
#                     # actually, this will be wrong. I before a B. but it is okay..
#                     start = ii    
#                 phrase.append(word)
        
    
        
    