import os
from collections import defaultdict
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import re

def get_fields():
    all_dicts = {}
    all_dicts['all']= {}

    subset = 'basketball'
    subset_dict = {}
    subset_dict['number_of_assists']=['2', '3', '4']
    subset_dict['number_of_blocks']=['2', '3', '4']
    subset_dict['number_of_turnovers']=['2', '3', '4']
    subset_dict['number_of_points']=['2', '3', '4']
    subset_dict['number_of_fouls']=['2', '3', '4']
    subset_dict['number_of_steals']=['2', '3', '4']
    subset_dict['number_of_rebounds']=['2', '3', '4']
    subset_dict['number_of_played_games']=['2', '3', '4']
    subset_dict['player']=['lebron_james', 'kobe_bryant']
    subset_dict['team']=['los_angeles_lakers','cleveland_cavaliers']
    subset_dict['position']=['point_guard','forward']
    subset_dict['season']=['2004', '2010']
    all_dicts[subset] = subset_dict

    #------------------------------------------------------
    subset = 'calendar'
    subset_dict = {}
    subset_dict['location']=['central_office', 'greenberg_cafe']
    subset_dict['important']=['true']
    subset_dict['meeting']=['annual_review', 'weekly_standup']
    subset_dict['attendee']=['alice', 'bob']
    subset_dict['length']=['one', 'two', 'three']
    subset_dict['end_time']=['1pm', '3pm', '10am']
    subset_dict['start_time']=['1pm', '3pm', '10am']
    subset_dict['date']=['jan_2nd', 'jan_3rd']
    all_dicts[subset] = subset_dict

    #------------------------------------------------------
    subset = 'housing'
    subset_dict = {}
    subset_dict['neighborhood']=['midtown_west', 'chelsea']
    subset_dict['housing_type']=['flat', 'apartment','condo']
    subset_dict['housing_unit']=['123_sesame_street', '900_mission_avenue']
    subset_dict['posting_date']=['feb_3rd', 'jan_2nd']
    subset_dict['size']=['800', '1000']
    subset_dict['monthly_rent']=['1500', '2000']
    subset_dict['cats']=['true']
    subset_dict['dogs']=['true']
    subset_dict['private_bath']=['true']
    subset_dict['private_room']=['true']    
    all_dicts[subset] = subset_dict   

    #------------------------------------------------------
    subset = 'recipes'
    subset_dict = {}
    subset_dict['cuisine']=['']
    subset_dict['recipe']=['rice_pudding', 'quiche']
    subset_dict['preparation_time']=['']
    subset_dict['cooking_time']=['']
    subset_dict['ingredient']=['spinach', 'milk', '']
    subset_dict['posting_date']=['2004', '2003', '2010']
    subset_dict['meal']=['lunch', 'dinner']
    all_dicts[subset] = subset_dict

    #-----------------------------------------------------
    subset = 'restaurants'
    subset_dict = {}
    subset_dict['neighborhood']=['midtown_west', 'chelsea']
    subset_dict['reviews']=['30', '40']
    subset_dict['reservations']=['true']
    subset_dict['cuisine']=['thai', 'italian']
    subset_dict['restaurant']=['thai_cafe', 'pizzeria_juno']
    subset_dict['credit_cards']=['true']
    subset_dict['delivery']=['true']
    subset_dict['meal']=['lunch', 'dinner']
    subset_dict['kids']=['true']
    subset_dict['star_rating']=['2', '3', '4', '5']
    subset_dict['groups']=['true']
    subset_dict['price_rating']=['2', '3', '4', '5']
    subset_dict['takeout']=['true']
    subset_dict['outdoor_seating']=['true']
    subset_dict['waiter_service']=['true']
    all_dicts[subset] = subset_dict

    fields = {}
    for subset in all_dicts.keys():
        dictionary = all_dicts[subset]
        fields[subset] = dictionary.keys()

    return fields, all_dicts

def renotate(s='test',sub='housing'):

    path = os.path.abspath(__file__)
    path = os.path.dirname(path).replace('utils','data/DATA/overnight_source/%s'%sub)

    fields, all_dicts  = get_fields()
    all_fields = fields[sub]
    dictionary = all_dicts[sub]

    error = 0
    qu_file = os.path.join(path, '%s.qu'%(s))
    lon_file = os.path.join(path, '%s.lon'%(s))
    new_lon_file = os.path.join(path, 'new_%s.lon'%s)
    new_qu_file = os.path.join(path, 'new_%s.qu'%s)
    ori_lon_file = os.path.join(path, '%s_%s0.lon'%(sub,s))
    with gfile.GFile(qu_file, mode='r') as qu, gfile.GFile(lon_file, mode='r') as lon,gfile.GFile(new_lon_file, mode='w') as new_lon, gfile.GFile(new_qu_file, mode='w') as new_qu, gfile.GFile(ori_lon_file, mode='r') as ori_lon:
        qu_lines = qu.readlines()
        lon_lines = lon.readlines()
        ori_lons = ori_lon.readlines()
        assert len(qu_lines) == len(lon_lines)
        assert len(lon_lines) == len(ori_lons)
        for Q, S, S0 in zip(qu_lines,lon_lines,ori_lons):
            #append 
            for i in range(4):
                if '<v'+str(i)+'>' in Q and '<v'+str(i)+'>' in S and '<f'+str(i)+'>' not in Q:
                    sym = '<v'+str(i)+'>'
                    idx = S.split().index(sym)
                    word = S0.split()[idx]
                    fs = []
                    for f in all_fields:
                        values = dictionary[f]
                        if word in values:
                            fs.append(f)
                    
                    if len(fs)>1:
                        print('--------')
                        print(word)
                        print(fs)
                    #assert len(fs)==1
                    f = fs[0]
                    Q = Q.replace(sym, '<f'+str(i)+'> '+f+' <eof> '+sym)

                
                if '<v'+str(i)+'>' not in Q and '<f'+str(i)+'>' in Q:
                    sym = '<f'+str(i)+'>'
                    idx = S.split().index(sym)
                    word = S0.split()[idx]
                    if idx+1>=len(S0.split()) or (idx+1<len(S0.split()) and S0.split()[idx+1] != 'where'):
                        values = dictionary[word]
                        if values[0] == 'true':
                            Q = re.sub('('+sym+' \w+ <eof>)',r'\1 <v'+str(i)+'> true <eof>', Q)
                            #Q = Q.replace(sym, sym+' <v'+str(i)+'> true <eof>') 
            
            #check over annotation rate
            for sym in ['<f0>','<f1>','<f2>','<f3>']:
                if sym in S:
                    if sym not in Q:
                        idx = S.split().index(sym)
                        newp = S0.split()[idx]
                        new_sym = '<c'+str( all_fields.index(newp) )+'>'
                        S = S.replace(sym,new_sym)
                        #S = S.replace(sym,newp)
                        
                        error += 1
            for i,f in enumerate(all_fields):
                S = S.replace(f,'<c'+str(i)+'>')         
            
            #switch i
            if '<f0>' in Q and '<v0>' in Q and '<f1>' in Q and '<v1>' not in Q:
                Q = Q.replace('<f1>','<ftmp>')
                Q = Q.replace('<f0>','<f1>').replace('<v0>','<v1>')
                Q = Q.replace('<ftmp>','<f0>')
                S = S.replace('<f1>','<ftmp>')
                S = S.replace('<f0>','<f1>').replace('<v0>','<v1>')
                S = S.replace('<ftmp>','<f0>')
            new_lon.write(S)
            new_qu.write(Q)
    print('over annotation rate:')
    print(error*1.0/len(qu_lines))


if __name__ == "__main__":
    renotate('train','housing')
    renotate('test','housing')




