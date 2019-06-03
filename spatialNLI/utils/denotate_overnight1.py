import os
from collections import defaultdict
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from redenote import get_fields,renotate
import re

smap = {}
with open('map.txt','r') as map_file:
    lines = map_file.readlines()
    for line in lines:
        key = line.split(':')[0]
        values = line.split(':')[1].split(',')
        smap[key] = values[0].replace('\n','')

def _chech_sketch(S):
    signs = ['select',' nl ',' ng ','equal min','equal max',' neq ','where ( count',' or ',' between ']
    for sign in signs:
        if sign in S:
            return False
    if re.search('\( \w+ \)',S):
        return False
    return True

fdict = defaultdict(list)
def denotate(s='test'):

    count1, count2 = 0, 0
    all_fields, _ = get_fields()
    all_fields = all_fields[sub]

    ta_file = os.path.join(path, '%s_%s.ta'%(sub,s))
    qu_file = os.path.join(path, '%s_%s.qu'%(sub,s))

    #newly generated question file
    question_file = os.path.join(path, '%s.qu'%s)
    lon_file = os.path.join(path, '%s_%s.lon'%(sub,s))
    #original lon file with qualified lon
    lon_file0 = os.path.join(path, '%s_%s0.lon'%(sub,s))
    sym_pair_file = os.path.join(path, '%s_sym_pairs.txt'%s)
    ground_truth_file = os.path.join(path, '%s_ground_truth.txt'%sub)

    if os.path.exists(ground_truth_file) and s == 'train':
        os.remove(ground_truth_file)

    with gfile.GFile(ta_file, mode='r') as t, gfile.GFile(qu_file, mode='r') as q,\
        gfile.GFile(question_file, mode='w') as re,\
        gfile.GFile(lon_file0, mode='w') as lon0,\
        gfile.GFile(lon_file, mode='r') as lon,\
        gfile.GFile(sym_pair_file, mode='w') as sym_file,\
        gfile.GFile(ground_truth_file, mode='a') as truth_file:
    
        templates = t.readlines()
        questions = q.readlines()
        lons = lon.readlines()
        assert len(templates)==len(questions)
        for template,question,lon in zip(templates,questions,lons):
            sym_line = ''
            t_tokens = template.split()
            q_tokens = question.split()
            assert len(t_tokens)==len(q_tokens)
            new = ''
            for t_token,q_token in zip(t_tokens,q_tokens):
                if t_token=='<nan>' or t_token=='<count>':
                    new += q_token
                else:
                    words = t_token.split(':')
                    sym = ('<'+words[0][1]+words[2]+'>')
                    new += sym
                    if words[0][1]=='f':
                        new += ' '
                        new += q_token
                        new += ' '
                        new += '<eof>'  
                    if words[0][1]=='f' or words[0][1]=='v':
                        if q_token in smap:
                            q_token = smap[q_token]
                        sym_line += (sym + '=>' + q_token + '<>')
                new += ' '
            new += '<eos> '
            sym_line += ('\n')

            for i,f in enumerate(all_fields):
                new += '<c'+str(i)+'> '+f+' <eoc> '
                sym_line += ('<c'+str(i)+'>' + '=>' + f + '<>')
           
            if _chech_sketch(lon):
                re.write(new+'\n')
                lon0.write(lon)
                truth_file.write(lon)
                sym_file.write(sym_line)
                count1 += 1


    print('question file done.')

    lox_file = os.path.join(path, '%s_%s.lox'%(sub,s))
    lon_file = os.path.join(path, '%s_%s.lon'%(sub,s))
    #newly generated logic file
    lo_file = os.path.join(path, '%s.lon'%s)
    with gfile.GFile(lox_file, mode='r') as lox,gfile.GFile(lon_file, mode='r') as lon, gfile.GFile(lo_file, mode='w') as re:
        loxs = lox.readlines()
        lons = lon.readlines()
        n = len(lons)
        error = 0
       
        assert len(lons)==len(loxs)
        #newline is redenotdated file
        for lox, lon in zip(loxs,lons):
            lo_tokens = lox.split()
            lon_tokens = lon.split()

            t_tokens = template.split()
            q_tokens = question.split()
            assert len(t_tokens)==len(q_tokens)
     
            new = ''
            for idx,(lo_token,lon_token) in enumerate(zip(lo_tokens,lon_tokens)):
                if ':' in lo_token and len(lo_token.split(':'))==3:
                    words = lo_token.split(':')
                    if False and words[0][1]=='f':
                        new += ('<'+words[0][1]+words[2]+'>')
                        new += ' '
                        new += lon_token
                        new += ' '
                        new += '<eof>'
                    else:
                        new += ('<'+words[0][1]+words[2]+'>')
                elif lo_token=='<count>':
                    new += lon_token
                elif lo_token == 'true':
                    ws = lo_tokens[idx-2].split(':')
                    new += ('<v'+words[2]+'>')
                else:
                    new += lo_token
                new += ' '

            if _chech_sketch(lon):
                re.write(new+'\n')
                count2 += 1
                
    assert count1 == count2
    print('logic file done.')

if __name__ == "__main__":
    for sub in ['basketball','calendar','housing','recipes','restaurants']:
        path = os.path.abspath(__file__)
        path = os.path.dirname(path).replace('utils','data/DATA/overnight_source/%s'%sub)
        denotate('train')
        denotate('test')
        renotate('train',sub)
        renotate('test',sub)
        

