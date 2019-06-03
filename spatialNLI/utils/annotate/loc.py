def find_loc(qu_annot, qu_pairs, w_filter):
    #in some cases, the word['in','of','with','has','have'] could be paired with 'loc' in lambda
    loc_dic = []
    loc_idx = []
    for idx,word in enumerate(w_filter):
        if word in ['in','of','with','has','have','contains','on','where','whose']:
            loc_dic.append(word)
            loc_idx.append(idx)

    wrong_w = 0
    for w in ['how many citizens','how many inhabitants','how many residents','how many people','how many population','square kilometers']:
        if w in line: 
            wrong_w += 1

    if len(loc_dic) > 0 and wrong_w == 0:
        for i,word in enumerate(loc_dic):
            loc_flag = _find_loc(word,loc_idx[i])
            if loc_flag == True:
                mark_1('loc', word, loc_idx[i],'f')
    elif len(loc_dic) > 1:
        count_loc = 0
        for i,word in enumerate(loc_dic):
            loc_flag = _find_loc(word,loc_idx[i]) 
            if loc_flag == True:
                count_loc += 1
        if count_loc > 1:
            mark_1('loc', loc_dic[1], loc_idx[i],'f')
    if 'how many people live in the capital of' in line:
        mark_1('loc', 'of', w_filter.index('of'),'f')

    return qu_annot,qu_pairs,w_filter


def _loc_check_phrase(self,word,idx,start_loc,loc_w):
    loc_word = ['states','all states','all the states','the largest state','the largest city','the state','rivers','a city','the states','the lowest point','the highest point','cities','a state','a river','continantal us','the largest state','the highest elevation','the lowest elevation','the longest river','the smallest state','the most major cities','those','the city','the biggest city','the most rivers','a major river','the most cities','fewest cities','most rivers','the largest capital city','each state']
    length = min(len(self.w_filter)-idx-1,5)
    for le in range(length, -1, -1): #see if the phrases after this word in the const, if it is, the word is paired with 'loc'
        if idx+1 < len(self.w_filter):
            pos_w = []
            for l in range(start_loc,le+1):
                pos_w.append(self.w_filter[idx + l])
            if pos_w != []:
                pos_w = ' '.join(pos_w)
                if pos_w in self.const_w:
                    return True
                elif loc_w == True and pos_w in loc_word:
                    return True
    return False

def _loc_check_phrase_left(self,word,idx,start_loc):
    for le in range(4, 1, -1):
        pre_w = [] 
        for l in range(idx-le,idx-start_loc):
            pre_w.append(self.w_filter[l])

        if pre_w != None:
            pre_w = ' '.join(pre_w)
            if pre_w in self.const_w and 'population' not in self.w_filter and self.w_filter[idx-1] not in ['city']:
                return True
    return False

def _find_loc(self,word,idx):
    if word == 'where':
        if ' '.join(self.w_filter[idx+1:idx+3]) == 'are mountains':
            return True
        elif ' '.join(self.w_filter[idx+1:idx+5]) == 'is the smallest city':
            return True
        elif self.w_filter[idx+1] in ['are','is']:
            return self._loc_check_phrase(word,idx,2,False)
    
    if word == 'on' and self.w_filter[idx-1] != 'lie':
        return True
    if word == 'whose':
        return True
    if word == 'of' and self.w_filter[idx-1] == 'number':
        return False
    if word in ['has','have']:
        if idx + 2 < len(self.w_filter) and self.w_filter[idx+1] == 'the' and self.w_filter[idx+2] == 'city':
            return True
        if 'citizens' in self.w_filter:
            return False
        if idx + 1 < len(self.w_filter) and self.w_filter[idx+1] == 'no':
            return False
        if len([w for w in ['population','populations','area','density','people'] if w in self.w_filter]) == 0:
            return True
        else:
            return False
    if word in ['in','of'] and self.w_filter[idx-1] in ['population','populations','people','area','density','largest','length','name','height','elevation','state','which']:
        return False
    if idx+1 < len(self.w_filter) and self.w_filter[idx+1] in ['meters']:
        return False
    if word == 'contains':
        return True
    
    if self. _loc_check_phrase(word,idx,1,True):
        return True
    if idx+1 < len(self.w_filter) and self.w_filter[idx+1] == 'capital' :
        if self. _loc_check_phrase(word,idx,2,False):
            return True
    if idx+1 < len(self.w_filter) and ' '.join(self.w_filter[idx+1:idx+3]) == 'the capital': 
        if self. _loc_check_phrase(word,idx,3,False):
            return True
    
    if self._loc_check_phrase_left(word,idx,0):
        return True
    
    if self.w_filter[idx-1] == 'located' :
        if self._loc_check_phrase_left(word,idx,1):
            return True

    if ' '.join(self.w_filter[idx-3:idx]) == 'is the capital':
        if self._loc_check_phrase_left(word,idx,3):
            return True
    if self.w_filter[idx-1] == 'capital' and self.w_filter[idx-2] == 'the':
        if self._loc_check_phrase_left(word,idx,2):
            return True
    return False

def mark_1(self, flag, word,idx,typ):
    self.w_filter[idx] = '<>' + word

    if typ == 'f':
        prefix = '<f' +str(self.count) +'>'#get the prefix<> for pairs
        pair = prefix + '=>' + flag +'<>'#pairs for state name
        self.qu_pairs.append(pair)
        word_ =' <f' +str(self.count) +'> '+ word + ' <eof>'

        self.count +=1
        self.qu_annot[idx]=word_
    
    elif typ == 'c':
        prefix = '<c' +str(self.count_c) +'>'#get the prefix<> for pairs
        pair = prefix + '=>' + flag +'<>'#pairs for state name
        self.qu_pairs.append(pair)
        word_ =' <c' +str(self.count_c) +'> '+ word + ' <eoc>'

        self.count_c +=1
        self.qu_annot[idx]=word_