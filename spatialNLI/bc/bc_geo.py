from StringIO import StringIO
import re
import glove
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

FILE = 'geobase.sql'
# import word_classifier as wc
# tf_model = wc.TF()
PARSE_PATH = '/nfs_shares/jzl0166_home/binary_classifier/data'


class BC:

    remove_stop = ['how','through','most','to','not','no']
    phrase_length = 3
    dim = 25

    def __init__(self, parse=PARSE_PATH, split='train'):

        qu_file='%s_org.qu'%split
        qu_file = os.path.join(parse, rawfile)

        ans_file='%s_ground_truth.txt'%split
        ans_file = os.path.join(parse, rawfile)
        
        word_list = ['city','state','river'] #potential label for ambiguous phrase
        
        if split == 'train':
            times = len(word_list)-1 #repeat times of the '1' in the training set for one ambiguous phrase
        else:
            times = 1 #for test, don't need to repeat '1'

        p_length = self.phrase_length
        dim = self.dim
        remove_stop = self.remove_stop
        
        self.read_word()
        self.g = glove.Glove()
        
        res = []

        with open(qu_file,'r') as f1,\
            open(ans_file,'r') as f2:
            l1, l2 = f1.readlines(), f2.readlines()
            for qu,ans in zip(l1,l2):
                
                qu = qu.replace(',',' ')
                qu = qu.replace('?','')
                qu = qu.replace('.', '')
                qu = qu.replace("'", '')
                self.qu = qu

                count = 0
                words = qu.strip('\n').split(' ')

                self.stop_words = set(stopwords.words('english'))
                for rw in remove_stop:
                    self.stop_words.remove(rw)

                self.w_filter = words
                
                for k,w in enumerate(words):
                    if w not in self.stop_words:
                        self.w_filter[k] = w
                    else:
                        self.w_filter[k] = ''

                self.qu = self.w_filter
                for le in range(p_length-1, -1, -1):
                    for idx in range(len(self.qu)-le):
                        
                        word = self.check_phrase(idx,le) 
                        if word != None:
                            word = ' '.join(word)
                            candidate = self.find_const_w(word, idx, le)#pick up the phrase
                            
                            if 'place' in candidate:
                                candidate.remove('place')
                            
                            if len(candidate) > 1:#the candidate phrase for spatial comprehension model(has two potential label)
                                
                                for c in candidate:
                                    if c + 'id' in ans:
                                        
                                        qu_ = qu.replace(word,'<f0> '+ word +' <eof>')
                                        qu_ = qu_.strip('\n').split(' ')
                                        
                                        for i in range(len(qu_),dim):
                                            qu_.append('<pad>')
                                        
                                        qu_ = ' '.join(qu_)
                                        for j in range(0,times):
                                            line1 = qu_ + '\t' + c+ '\t' + '1'
                                            res.append(line1)

                                        word_list.remove(c)
                                        for j in word_list:
                                            line2 = qu_ + '\t' + j+ '\t' + '0'
                                            res.append(line2)
                                        word_list.append(c)
                                        
                                        break

        print('\nSaving bc_data_geo')
        with open(os.path.join(parse, '%s_model_const.txt'%self.split), 'w') as f:
            f.write('\n'.join(res))




    def find_const_w(self, word, idx, le):
        candidate = []
        dic_can = {'country': self.country,
                   'mountain': self.mountain,
                   'lake': self.lake,
                   'place': self.state_place,
                   'city': self.city,
                   'river': self.river,
                   'state': self.state_name}

        for key, v in dic_can.items():
            if word in v:
                candidate.append(key)
        
        return candidate

#restore the phrase(one or more words)
    def check_phrase(self,idx,le): 
        #'' should not be in a phrase ,'<' means this phrase has been annotated
        word = [] 
        for l in range(0,le+1):
            word.append(self.qu[idx + l])

        return word

    def read_word(self):
        self.city = self._import_data('city', ' name')
        self.city += self._import_data('state', ' capital')
        self.state_name = self._import_data('state', 'name')
        self.state_abbre = self._import_data('state', ' abbreviation')
        self.state_place = self._import_data('state', ' highest_point')
        self.state_place += self._import_data('state', ' lowest_point')
        self.country = self._import_data('country', 'name')
        self.lake = self._import_data('lake', 'name')
        self.mountain = self._import_data('mountain', 'name')
        self.river = self._import_data('river', 'name')

    #some words such as the 'arkansas' river, appears in the state place table as 'arkansas river', remove this kind of word out of state_place
        self.state_place = list(set(self.state_place))
        for i_state in self.state_place:
            for i_river in self.river:
                if i_river + ' river' == i_state:
                    self.state_place.remove(i_state)

            for i_lake in self.lake:
                if 'lake ' + i_lake == i_state:
                    self.state_place.remove(i_state)

    def _import_data(self, db_name, col_name, file = FILE):
        table_data = self._read_dump(file, db_name)
        table = pd.read_csv(table_data)
        names = list(table[col_name])
        name_ = []
        for name in names:
            name = name.replace(" '",'')
            name = name.replace("'",'')
            name_.append(name)
        name = name_

        return name


    def _read_dump(self, dump_filename, target_table):
        sio = StringIO()

        fast_forward = True
        with open(dump_filename, 'rb') as f:
            for line in f:
                line = line.strip()
                if line.lower().startswith('insert') and target_table in line:
                    fast_forward = False
                if fast_forward:
                    continue
                data = re.findall('\([^\)]*\)', line)
                try:
                    newline = data[0]
                    newline = newline.strip(' ()')
                    newline = newline.replace('`', '')
                    sio.write(newline)
                    sio.write("\n")
                except IndexError:
                    pass
                if line.endswith(';'):
                    break
        sio.pos = 0
        return sio


if __name__ == '__main__':
    bc = BC(split = 'train')
    bc = BC(split = 'test')
    bc = BC(split = 'dev')