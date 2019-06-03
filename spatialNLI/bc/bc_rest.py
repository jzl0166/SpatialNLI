#from StringIO import StringIO
import re
import glove
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

FILE = 'rest-db-name'
import word_classifier as wc
tf_model = wc.TF()
PARSE_PATH = '/nfs_shares/jzl0166_home/binary_classifier/data'
#PARSE_PATH = '/Users/lijing/Desktop/res250/data'

class BC:

    remove_stop = ['how','through','most','to','not','no']
    phrase_length = 3
    dim = 25

    def __init__(self, parse=PARSE_PATH, split='train'):

        qu_file='%s_org.qu'%split
        qu_file = os.path.join(parse, rawfile)

        ans_file='%s_ground_truth.txt'%split
        ans_file = os.path.join(parse, rawfile)
        
        word_list = ['street','city','region','rest','foodtype']

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
                            candidate = self.find_const_w(word, idx, le)
                            
                            if 'place' in candidate:
                                candidate.remove('place')
                            
                            if len(candidate) > 1:
                                ans_max = []
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

        print('\nSaving bc_data_rest')
        with open(os.path.join(parse, '%s_model_const.txt'%self.split), 'w') as f:
            f.write('\n'.join(res))




    def find_const_w(self, word, idx, le):
        candidate = []
        dic_can = {'street': self.street,
                   'city': self.city,
                   'region': self.region,
                   'county': self.county,
                   'rest': self.rest,
                   'foodtype': self.foodtype,
                   'rating': self.rating}

        for k, v in dic_can.items():
            if word in v:
                candidate.append(k)
        
        return candidate

#check the phrase(one or more words)
    def check_phrase(self,idx,le): 
        #'' should not be in a phrase ,'<' means this phrase has been annotated
        word = [] 
        for l in range(0,le+1):
            word.append(self.qu[idx + l])

        return word

    def read_word(self):
        self.street = self._import_data('street')
        self.city = self._import_data('city')
        self.county = self._import_data('county')
        self.region = self._import_data('region')
        self.rest = self._import_data('rest')
        self.foodtype = self._import_data('foodtype')
        self.rating = self._import_data('rating')

    def _import_data(self, target_table, filename = FILE):
        res = []
        with open(filename, 'rb') as f:
            for line in f:
                line = line.replace('food_type','foodtype')
                if line[0:len(target_table)] == target_table:
                    line = line.replace('foodtype','food_type')
                    line = line.strip()
                    data = re.match(r'^.*\(\[.*\].*\[.*\], \'(.*)\'\).*$', line)
                    try:
                        newline = data.group(1)
                        newline = newline.replace('-',' ')
                        newline = newline.replace('_',' ')
                        newline = newline.replace('?','')
                        newline = newline.replace("'",'')
                        newline = newline.replace("[",' [ ')
                        newline = newline.replace("]",' ] ')
                        newline = newline.replace('( * )','')
                        res.append(newline)
                    except IndexError:
                        pass
        return res


if __name__ == '__main__':
    bc = BC(split = 'train')
    bc = BC(split = 'test')
    bc = BC(split = 'dev')
