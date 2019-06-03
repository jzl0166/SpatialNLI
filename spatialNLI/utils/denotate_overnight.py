import os
from collections import defaultdict
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from redenote import get_fields,renotate
import re
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

    with gfile.GFile(ta_file, mode='r') as t, gfile.GFile(qu_file, mode='r') as q, gfile.GFile(question_file, mode='w') as re, gfile.GFile(lon_file0, mode='w') as lon0, gfile.GFile(lon_file, mode='r') as lon:
        templates = t.readlines()
        questions = q.readlines()
        lons = lon.readlines()
        assert len(templates)==len(questions)
        for template,question,lon in zip(templates,questions,lons):
            t_tokens = template.split()
            q_tokens = question.split()
            assert len(t_tokens)==len(q_tokens)
            new = ''
            for t_token,q_token in zip(t_tokens,q_tokens):
                if t_token=='<nan>' or t_token=='<count>':
                    new += q_token
                else:
                    words = t_token.split(':')
                    new += ('<'+words[0][1]+words[2]+'>')
                    if words[0][1]=='f':
                    #if words[0][1]=='f' or words[0][1]=='v':
                        new += ' '
                        new += q_token
                        new += ' '
                        new += '<eof>'  
                new += ' '
            new += '<eos> '
            
            for i,f in enumerate(all_fields):
                new += '<c'+str(i)+'> '+f+' <eoc> '
           
            if _chech_sketch(lon):
                re.write(new+'\n')
                lon0.write(lon)
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
        

