#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 19:15:40 2017

@author: eric.hensleyibm.com
"""
   
import mysql.connector   
import pandas as pd
import numpy as np
import tuned_line_models
import matplotlib.pyplot as plt
import tuned_ml_models
from sklearn.utils import resample
import requests
from lxml import html
import re
import unicodedata
import datetime
from mysql.connector import IntegrityError
from datetime import date
 
def process_raw_ml(traintest):
    from pulldata_ml_test import pulldata_ml_test
    from pulldata_ml_train import pulldata_ml_train
    import pandas as pd
    import numpy as np

    if traintest == 'train':
        x = pulldata_ml_train()
        x = x.dropna(how='any')
    elif traintest == 'test':
        x = pulldata_ml_test()
        x = x.dropna(how='any')
    elif traintest == 'new':
        x = pulldata_ml_test()
        x = x[pd.isnull(x[3])]

    rawheaders = ['oddsdate', 'favorite', 'underdog', 'favscore', 'dogscore', 'fav_moneyline', 'dog_moneyline', 'fav_homeaway', 'fav Basset rank', 'dog Basset rank', 'favorite predictive by others', 'dog predictive by others', 'favorite home by others', 'dog home by others', 'favorite away by others', 'dog away by others', 'favorite home adv by others', 'dog home adv by others', 'favorite schedule strength by others', 'dog schedule strength by others', 'favorite future sos by others', 'dog future sos by others', 'favorite season sos by others', 'dog season sos by others', 'favorite last 5 games by others', 'dog last 5 games by others', 'favorite last 10 games by others', 'dog last 10 games by others', 'favorite luck by others', 'dog luck by others', 'favorite consistency by others', 'dog consistency by others', 'favorite vs 1-10 by others', 'dog vs 1-10 by others', 'fav LAZ', 'dog LAZ', 'fav ARG', 'dog ARG', 'fav MAS', 'dog MAS', 'fav SAG', 'dog SAG', 'fav HOW', 'dog HOW', 'fav BIL', 'dog BIL', 'fav MAR', 'dog MAR', 'fav DOK', 'dog DOK', 'fav DES', 'dog DES', 'fav MOR', 'dog MOR', 'fav BRN', 'dog BRN', 'fav PIG', 'dog PIG', 'fav CGV', 'dog CGV', 'fav BDF', 'dog BDF']
    raw_data = x.rename(columns=dict(zip(list(x), rawheaders)))
    rawvars  = [0,0,0,0,0,0,0,0,'basset', 0, 'predictive', 0, 'home', 0, 'away', 0, 'home_adv', 0, 'sos', 0, 'fut_sos', 0, 'seas_sos', 0 ,'last5', 0, 'last10', 0, 'luck', 0, 'consistency', 0, 'vs_top10', 0, 'LAZ', 0, 'ARG', 0, 'MAS', 0,'SAG',0,'HOW',0,'BIL',0,'MAR',0,'DOK',0,'DES',0,'MOR',0,'BRN',0,'PIG',0,'CGV',0,'BDF',0]
    allinputs = pd.DataFrame()
    allinputs['date'] = x[0]
    allinputs['fav'] = x[1]
    allinputs['dog'] = x[2]
    variablename = 'homeaway'
    variablevalue = x[7]
    allinputs[variablename] = variablevalue
    for z in range(8,len(list(x)),2):
        if z in range(8, 10) or z in range(34, len(rawvars)):
            variablename=None
            variablevalue=None
            variablename = 'diff%s' % (rawvars[z])
            variablevalue = x[z] - x[z+1]
            allinputs[variablename] = variablevalue
            variablename=None
            variablevalue=None
            variablename = 'share%s' % (rawvars[z])
            variablevalue = x[z]/(x[z]+x[z+1])
            allinputs[variablename] = variablevalue
            variablename=None
            variablevalue=None
            variablename = 'total%s' % (rawvars[z])
            variablevalue = x[z]+x[z+1]
            allinputs[variablename] = variablevalue  
        elif z in range(10, 12) or z in range(17, 34):
            variablename=None
            variablevalue=None
            variablename = 'diff%s' % (rawvars[z])
            variablevalue = x[z] - x[z+1]
            allinputs[variablename] = variablevalue
            variablename=None
            variablevalue=None
            variablename = 'total%s' % (rawvars[z])
            variablevalue = x[z]+x[z+1]
            allinputs[variablename] = variablevalue            
    
    homeranks = np.array(x[[12,13]])
    awayranks = np.array(x[[14,15]])
    homefieldranks = np.array(x[[15,17]])
    homeoraway = np.array(x[7])
    
    homeawayrankdiff = []
    homefieldeffect = []
    
    for loc in range(0, len(homeoraway)):
        if homeoraway[loc] == 0:
            diff = None
            field = None
            diff = homeranks[loc][0] - awayranks[loc][1]
            homeawayrankdiff.append(diff)
            field = homefieldranks[loc][0]
            homefieldeffect.append(field)
        elif homeoraway[loc] == 1:
            diff = None
            field = None
            diff = awayranks[loc][0] - homeranks[loc][1]
            homeawayrankdiff.append(diff)
            field = homefieldranks[loc][1]*(-1)
            homefieldeffect.append(field)        
    
    allinputs['homeawaydiff'] = np.array(homeawayrankdiff)
    allinputs['fieldeffect'] = np.array(homefieldeffect)
    
    binary = []
    for game in range(0, len(x[3])):
        if np.array(x[3])[game] > np.array(x[4])[game]: #> np.array(x[5])[game]:
            binary.append(1)
        elif (np.array(x[3])[game] < np.array(x[4])[game]): # < np.array(x[5])[game]:
            binary.append(0)
        else:
            binary.append('in progress or tied')
    
    classset = pd.DataFrame()
    for each in allinputs:
        classset[each] = allinputs[each]
    classset['favml'] = x[5]
    classset['dogml'] = x[6]
    if traintest != 'new':
        classset['y'] = np.array(binary)
    
    if traintest == 'test':
        traintest = 'validation'
    
    classset.to_csv('%s_processed_ml.csv'%(traintest))
    raw_data.to_csv('%s_raw_ml.csv'%(traintest)) 
    return (raw_data, classset)


def process_raw_line(traintest):
    from pulldata_line_test import pulldata_line_test
    from pulldata_line_train import pulldata_line_train
    import pandas as pd
    import numpy as np  
    if traintest == 'train':
        x = pulldata_line_train()
        x = x.dropna(how='any')
    elif traintest == 'test':
        x = pulldata_line_test()
        x = x.dropna(how='any')
    elif traintest == 'new':
        x = pulldata_line_test()
        x = x[pd.isnull(x[3])]

    rawheaders = ['oddsdate', 'favorite', 'underdog', 'favscore', 'dogscore', 'line', 'juice', 'fav_homeaway', 'fav Basset rank', 'dog Basset rank', 'favorite predictive by others', 'dog predictive by others', 'favorite home by others', 'dog home by others', 'favorite away by others', 'dog away by others', 'favorite home adv by others', 'dog home adv by others', 'favorite schedule strength by others', 'dog schedule strength by others', 'favorite future sos by others', 'dog future sos by others', 'favorite season sos by others', 'dog season sos by others', 'favorite last 5 games by others', 'dog last 5 games by others', 'favorite last 10 games by others', 'dog last 10 games by others', 'favorite luck by others', 'dog luck by others', 'favorite consistency by others', 'dog consistency by others', 'favorite vs 1-10 by others', 'dog vs 1-10 by others', 'fav LAZ', 'dog LAZ', 'fav ARG', 'dog ARG', 'fav MAS', 'dog MAS', 'fav SAG', 'dog SAG', 'fav HOW', 'dog HOW', 'fav BIL', 'dog BIL', 'fav MAR', 'dog MAR', 'fav DOK', 'dog DOK', 'fav DES', 'dog DES', 'fav MOR', 'dog MOR', 'fav BRN', 'dog BRN', 'fav PIG', 'dog PIG', 'fav CGV', 'dog CGV', 'fav BDF', 'dog BDF']
    raw_data = x.rename(columns=dict(zip(list(x), rawheaders)))
    rawvars  = [0,0,0,0,0,0,0,0,'basset', 0, 'predictive', 0, 'home', 0, 'away', 0, 'home_adv', 0, 'sos', 0, 'fut_sos', 0, 'seas_sos', 0 ,'last5', 0, 'last10', 0, 'luck', 0, 'consistency', 0, 'vs_top10', 0, 'LAZ', 0, 'ARG', 0, 'MAS', 0,'SAG',0,'HOW',0,'BIL',0,'MAR',0,'DOK',0,'DES',0,'MOR',0,'BRN',0,'PIG',0,'CGV',0,'BDF',0]
    allinputs = pd.DataFrame()
    allinputs['date'] = x[0]
    allinputs['fav'] = x[1]
    allinputs['dog'] = x[2]
    variablename = 'homeaway'
    variablevalue = x[7]
    allinputs[variablename] = variablevalue
    for z in range(8,len(list(x)),2):
        if z in range(8, 10) or z in range(34, len(rawvars)):
            variablename=None
            variablevalue=None
            variablename = 'diff%s' % (rawvars[z])
            variablevalue = x[z] - x[z+1]
            allinputs[variablename] = variablevalue
            variablename=None
            variablevalue=None
            variablename = 'share%s' % (rawvars[z])
            variablevalue = x[z]/(x[z]+x[z+1])
            allinputs[variablename] = variablevalue
            variablename=None
            variablevalue=None
            variablename = 'total%s' % (rawvars[z])
            variablevalue = x[z]+x[z+1]
            allinputs[variablename] = variablevalue  
        elif z in range(10, 12) or z in range(17, 34):
            variablename=None
            variablevalue=None
            variablename = 'diff%s' % (rawvars[z])
            variablevalue = x[z] - x[z+1]
            allinputs[variablename] = variablevalue
            variablename=None
            variablevalue=None
            variablename = 'total%s' % (rawvars[z])
            variablevalue = x[z]+x[z+1]
            allinputs[variablename] = variablevalue            
    
    homeranks = np.array(x[[12,13]])
    awayranks = np.array(x[[14,15]])
    homefieldranks = np.array(x[[15,17]])
    homeoraway = np.array(x[7])
    
    homeawayrankdiff = []
    homefieldeffect = []
    
    for loc in range(0, len(homeoraway)):
        if homeoraway[loc] == 0:
            diff = None
            field = None
            diff = homeranks[loc][0] - awayranks[loc][1]
            homeawayrankdiff.append(diff)
            field = homefieldranks[loc][0]
            homefieldeffect.append(field)
        elif homeoraway[loc] == 1:
            diff = None
            field = None
            diff = awayranks[loc][0] - homeranks[loc][1]
            homeawayrankdiff.append(diff)
            field = homefieldranks[loc][1]*(-1)
            homefieldeffect.append(field)        
    
    allinputs['homeawaydiff'] = np.array(homeawayrankdiff)
    allinputs['fieldeffect'] = np.array(homefieldeffect)
    
    binary = []
    for game in range(0, len(x[3])):
        if (np.array(x[3])[game] - np.array(x[4])[game]) > np.array(x[5])[game]*-1:
            binary.append(1)
        elif (np.array(x[3])[game] - np.array(x[4])[game]) < np.array(x[5])[game]*-1:
            binary.append(0)
        elif (np.array(x[3])[game] - np.array(x[4])[game]) == np.array(x[5])[game]*-1:
            binary.append(None)
    
    classset = pd.DataFrame()
    for each in allinputs:
        classset[each] = allinputs[each]
    classset['juice'] = x[6]
    if traintest != 'new':
        classset['y'] = np.array(binary)
    classset['line'] = x[5]
    classset = classset.dropna(how='any')
    
    if traintest == 'test':
        traintest = 'validation'
    
    classset.to_csv('%s_processed_line.csv'%(traintest))
    raw_data.to_csv('%s_raw_line.csv'%(traintest)) 
    return (raw_data, classset)   


class updatesql():    
    def update_odds():
        passcode = 'ibm1234'
        cnx = mysql.connector.connect(user='root', password=passcode,
                                      host='127.0.0.1',
                                      database='ncaa')    
        cursor = cnx.cursor() 
        cursor.execute('SET SQL_SAFE_UPDATES = 0;')
        cursor.execute('DELETE FROM ncaa.oddsdata WHERE favscore is NULL;')
        cursor.execute('SET SQL_SAFE_UPDATES = 1;')
        cnx.commit()
        offseason = [1,2,3,4,5,6,7]
        start_date = date(2017, 11, 21)
        dates = []
        while 1 != 2:
            new_date = start_date + datetime.timedelta(days=1)
            if new_date != date(2017,11,29):
                urldate = '%s-%s-%s' % (new_date.year, new_date.month, new_date.day)
                if new_date.month not in offseason:
                    dates.append(urldate)
                start_date = new_date
            else:
                break     
        uppercaseteamnames = ['KENT STATE','S MISSISSIPPI','CINCINNATI', 'WASHINGTON', 'MIDDLE TENN','FLORIDA INTL', 'ARIZONA', 'AIR FORCE', 'AKRON', 'ALABAMA', 'APP STATE', 'ARIZONA', 'ARIZONA ST', 'ARKANSAS', 'ARKANSAS ST', 'ARMY', 'AUBURN', 'BYU', 'BALL STATE', 'BAYLOR', 'BOISE STATE', 'BOSTON COL', 'BOWLING GRN', 'BUFFALO', 'CALIFORNIA', 'CENTRAL FL', 'CENTRAL MICH', 'CHARLOTTE', 'CINCINNATI', 'CLEMSON', 'COASTAL CAR', 'COLORADO', 'COLORADO ST', 'CONNECTICUT', 'DUKE', 'E CAROLINA', 'E MICHIGAN', 'FLA ATLANTIC', 'FLORIDA', 'FLORIDA INTL', 'FLORIDA ST', 'FRESNO ST', 'GA SOUTHERN', 'GA TECH', 'GEORGIA', 'GEORGIA STATE', 'HAWAII', 'HOUSTON', 'IDAHO', 'ILLINOIS', 'INDIANA', 'IOWA', 'IOWA STATE', 'KANSAS', 'KANSAS ST', 'KENT STATE', 'KENTUCKY', 'LA LAFAYETTE', 'LA MONROE', 'LA TECH', 'LSU', 'LOUISVILLE', 'MARSHALL', 'MARYLAND',  'U MASS', 'MEMPHIS', 'MIAMI (FL)', 'MIAMI (OH)', 'MICHIGAN', 'MICHIGAN ST', 'MIDDLE TENN', 'MINNESOTA', 'MISS STATE', 'MISSISSIPPI', 'MISSOURI', 'N CAROLINA', 'N ILLINOIS', 'N MEX STATE', 'NC STATE', 'NAVY', 'NEBRASKA', 'NEVADA', 'NEW MEXICO', 'NORTH TEXAS', 'NORTHWESTERN', 'NOTRE DAME', 'OHIO', 'OHIO STATE', 'OKLAHOMA', 'OKLAHOMA ST', 'OLD DOMINION', 'OREGON', 'OREGON ST', 'PENN STATE', 'PITTSBURGH', 'PURDUE', 'RICE', 'RUTGERS', 'SAN DIEGO ST',  'S ALABAMA', 'S CAROLINA', 'S FLORIDA', 'S METHODIST', 'S MISSISSIPPI','SAN JOSE ST', 'STANFORD', 'SYRACUSE', 'TX CHRISTIAN', 'TX EL PASO', 'TX-SAN ANT', 'TEMPLE', 'TENNESSEE', 'TEXAS', 'TEXAS A&M', 'TEXAS STATE', 'TEXAS TECH', 'TOLEDO', 'TROY', 'TULANE', 'TULSA', 'UAB', 'UCLA', 'UNLV', 'USC', 'UTAH', 'UTAH STATE', 'VA TECH', 'VANDERBILT', 'VIRGINIA', 'W KENTUCKY', 'W MICHIGAN', 'W VIRGINIA', 'WAKE FOREST', 'WASH STATE', 'WASHINGTON', 'WISCONSIN', 'WYOMING']
        uppercaseoddsteams = ['KENT', 'SO MISSISSIPPI', 'CINCINNATI U', 'WASHINGTON U', 'MIDDLE TENN ST', 'FLORIDA INTL', 'ARIZONA U', 'AIR FORCE', 'AKRON', 'ALABAMA', 'APPALACHIAN ST', 'ARIZONA', 'ARIZONA STATE', 'ARKANSAS', 'ARKANSAS STATE', 'ARMY', 'AUBURN', 'BYU', 'BALL STATE', 'BAYLOR', 'BOISE STATE', 'BOSTON COLLEGE', 'BOWLING GREEN', 'BUFFALO U', 'CALIFORNIA', 'CENTRAL FLORIDA', 'CENTRAL MICHIGAN', 'CHARLOTTE', 'CINCINNATI', 'CLEMSON', 'COASTAL CAROLINA', 'COLORADO', 'COLORADO STATE', 'CONNECTICUT', 'DUKE', 'EAST CAROLINA', 'EASTERN MICHIGAN', 'FLORIDA ATLANTIC','FLORIDA', 'FLORIDA INTERNATIONAL', 'FLORIDA STATE', 'FRESNO STATE', 'GEORGIA SOUTHERN', 'GEORGIA TECH',  'GEORGIA', 'GEORGIA STATE', 'HAWAII', 'HOUSTON U', 'IDAHO', 'ILLINOIS', 'INDIANA', 'IOWA', 'IOWA STATE', 'KANSAS', 'KANSAS STATE', 'KENT STATE', 'KENTUCKY', 'UL - LAFAYETTE', 'UL - MONROE', 'LOUISIANA TECH', 'LSU', 'LOUISVILLE', 'MARSHALL', 'MARYLAND', 'MASSACHUSETTS', 'MEMPHIS', 'MIAMI FLORIDA', 'MIAMI OHIO', 'MICHIGAN', 'MICHIGAN STATE', 'MID TENNESSEE STATE', 'MINNESOTA', 'MISSISSIPPI STATE', 'MISSISSIPPI', 'MISSOURI', 'NORTH CAROLINA', 'NORTHERN ILLINOIS',   'NEW MEXICO STATE', 'NORTH CAROLINA STATE',  'NAVY', 'NEBRASKA', 'NEVADA', 'NEW MEXICO', 'NORTH TEXAS', 'NORTHWESTERN', 'NOTRE DAME', 'OHIO', 'OHIO STATE', 'OKLAHOMA', 'OKLAHOMA STATE', 'OLD DOMINION', 'OREGON', 'OREGON STATE', 'PENN STATE', 'PITTSBURGH', 'PURDUE', 'RICE', 'RUTGERS', 'SAN DIEGO STATE', 'SOUTH ALABAMA',  'SOUTH CAROLINA', 'SOUTH FLORIDA',  'SMU', 'SOUTHERN MISS', 'SAN JOSE STATE', 'STANFORD', 'SYRACUSE', 'TCU',  'UTEP',  'TEX SAN ANTONIO',  'TEMPLE', 'TENNESSEE U', 'TEXAS', 'TEXAS AM', 'TEXAS STATE', 'TEXAS TECH', 'TOLEDO', 'TROY', 'TULANE', 'TULSA', 'UAB', 'UCLA', 'UNLV', 'USC', 'UTAH', 'UTAH STATE','VIRGINIA TECH', 'VANDERBILT', 'VIRGINIA', 'WESTERN KENTUCKY', 'WESTERN MICHIGAN',  'WEST VIRGINIA', 'WAKE FOREST',  'WASHINGTON STATE', 'WASHINGTON','WISCONSIN','WYOMING']
        oddsteamsdict = {}
        for i in range(0, len(uppercaseteamnames)):
            oddsteamsdict[uppercaseoddsteams[i]] = uppercaseteamnames[i]
        nonfbsteams = []
        all_games = []
        all_errors = []
        favorite_errors = []
        for gameday in dates:
                url = None
                pageContent = None
                tree = None
                day = None
                month = None
                year = None
                if len(gameday.split('-')[2]) == 1:
                    day = '0'+gameday.split('-')[2]
                elif len(gameday.split('-')[2]) == 2:
                    day = gameday.split('-')[2]
                if len(gameday.split('-')[1]) == 1:
                    month = '0'+gameday.split('-')[1]
                elif len(gameday.split('-')[1]) == 2:
                    month = gameday.split('-')[1]
                year = gameday.split('-')[0]
                url = 'http://www.scoresandodds.com/grid_%s%s%s.html' % (year, month, day)
                pageContent=requests.get(url)
                tree = html.fromstring(pageContent.content)    
                for sport in range(1, 10):                    
                    root = '/html/body/div[1]/table/tr/td/div[2]/div[1]/div[1]/table/tr[1]/td[1]/div[%s]' % (sport)
                    sportpath = root+'/div[5]/div[1]/text()'
                    if len(tree.xpath(sportpath)) > 0 and tree.xpath(sportpath)[0] == 'NCAA FB':
                        team1root = '/div[6]/div[1]/table/tbody/tr[@class = "team odd"]'
                        team2root = '/div[6]/div[1]/table/tbody/tr[@class = "team even"]'
                        nameroot = '/td[1]/a/text()'
                        namepath1 = None
                        namepath2 = None
                        team1namelist = None
                        team2namelist = None
                        team1overunderlist = None
                        team2overunderlist = None
                        team1linelist = None
                        team2linelist = None
                        team1moneylinelist = None
                        team2moneylinelist = None
                        team1scorelist = None
                        team2scorelist = None
            
                        namepath1 = root+team1root+nameroot
                        namepath2 = root+team2root+nameroot
                        
                        team1namelist = tree.xpath(namepath1)
                        team2namelist = tree.xpath(namepath2)
            
                        team1linelist = []
                        for l1 in range(1, len(team1namelist)+1):
                            lpath1 = root+'/div[6]/div[1]/table/tbody/tr[@class = "team odd"][position()=%s]/td[4]/text()' % (l1)
                            try:
                                team1linelist.append(tree.xpath(lpath1)[0])
                            except IndexError:
                                if tree.xpath(lpath1) == []:
                                    team1linelist.append(None)
                                                         
                        team2linelist = []
                        for l2 in range(1, len(team2namelist)+1):
                            lpath2 = root+'/div[6]/div[1]/table/tbody/tr[@class = "team even"][position()=%s]/td[4]/text()' % (l2)
                            try:
                                team2linelist.append(tree.xpath(lpath2)[0])
                            except IndexError:
                                if tree.xpath(lpath2) == []:
                                    team2linelist.append(None)            
                        
                        team1overunderlist = []
                        for ou1 in range(1, len(team1namelist)+1):
                            oupath1 = root+'/div[6]/div[1]/table/tbody/tr[@class = "team odd"][position()=%s]/td[4]/text()' % (ou1)
                            try:
                                team1overunderlist.append(tree.xpath(oupath1)[0])
                            except IndexError:
                                if tree.xpath(oupath1) == []:
                                    team1overunderlist.append('Null')
                        
                        team2overunderlist = []
                        for ou2 in range(1, len(team2namelist)+1):
                            oupath2 = root+'/div[6]/div[1]/table/tbody/tr[@class = "team even"][position()=%s]/td[4]/text()' % (ou2)
                            try:
                                team2overunderlist.append(tree.xpath(oupath2)[0])
                            except IndexError:
                                if tree.xpath(oupath2) == []:
                                    team2overunderlist.append('Null')     
                        
                        team1scorelist = []
                        for s1 in range(1, len(team1namelist)+1):
                            spath1 = root+'/div[6]/div[1]/table/tbody/tr[@class = "team odd"][position()=%s]/td[7]/span[1]/text()' % (s1)
                            try:
                                team1scorelist.append(tree.xpath(spath1)[0])
                            except IndexError:
                                if tree.xpath(spath1) == []:
                                    team1scorelist.append('Null')
            
                        team2scorelist = []            
                        for s2 in range(1, len(team2namelist)+1):
                            spath2 = root+'/div[6]/div[1]/table/tbody/tr[@class = "team even"][position()=%s]/td[7]/span[1]/text()' % (s2)
                            try:
                                team2scorelist.append(tree.xpath(spath2)[0])
                            except IndexError:
                                if tree.xpath(spath2) == []:
                                    team2scorelist.append('Null')
                                    
                        team1moneylinelist = []
                        for ml1 in range(1, len(team1namelist)+1):
                            mlpath1 = root+'/div[6]/div[1]/table/tbody/tr[@class = "team odd"][position()=%s]/td[5]/text()' % (ml1)
                            try:
                                team1moneylinelist.append(tree.xpath(mlpath1)[0])
                            except IndexError:
                                team1moneylinelist.append('Null')
                        team2moneylinelist = []
                        for ml2 in range(1, len(team2namelist)+1):
                            mlpath2 = root+'/div[6]/div[1]/table/tbody/tr[@class = "team even"][position()=%s]/td[5]/text()' % (ml2)
                            try:
                                team2moneylinelist.append(tree.xpath(mlpath2)[0])
                            except IndexError:
                                team2moneylinelist.append('Null')
                        if len(team1namelist) == len(team2namelist) == len(team2moneylinelist) == len(team1moneylinelist) == len(team1scorelist) == len(team2scorelist) == len(team1linelist) == len(team2linelist) == len(team1overunderlist) == len(team2overunderlist): 
                            for each in range(0, len(team1namelist)):
                                try:
                                    fbsgame = 'yes'
                                    favorite = None
                                    line = None
                                    team1 = None
                                    team2 = None
                                    moneyline1 = None
                                    moneyline2 = None
                                    score1 = None
                                    score2 = None
                                    linejuice = None
                                    overunder = None
                                    overunderjuice = None
                                    game = []
                                    
                                    try:
                                        team1 = oddsteamsdict[team1namelist[each][4:].upper()]
                                    except KeyError:
                                        fbsgame = 'no'
                                        pass
                                    try:
                                        team2 = oddsteamsdict[team2namelist[each][4:].upper()]
                                    except KeyError:
                                        fbsgame = 'no'
        
                                    if fbsgame == 'yes':                                
                                        try:
                                            moneyline1 = float(team1moneylinelist[each])
                                        except ValueError:
                                            if team1moneylinelist[each] == 'Null':
                                                moneyline1 = team1moneylinelist[each]
                                        try:
                                            moneyline2 = float(team2moneylinelist[each])
                                        except ValueError:
                                            if team2moneylinelist[each] == 'Null':
                                                moneyline2 = team2moneylinelist[each]
                                        try:
                                            score1 = int(team1scorelist[each])
                                        except ValueError:
                                            if team1scorelist[each] == 'Null':
                                                score1 = team1scorelist[each]
                                        try:
                                            score2 = int(team2scorelist[each])
                                        except ValueError:
                                            if team2scorelist[each] == 'Null':
                                                score2 = team2scorelist[each]
                                        favorite1 = False
                                        favorite2 = False
                                        x1 = None
                                        x2 = None
                                        y1 = None
                                        y2 = None
                                        spacesplit1 = None
                                        osplit1 = None
                                        usplit1 = None
                                        
                                        if team1linelist[each] == None and team2linelist[each] == None:
                                            x1 = 'Null'
                                            x2 = 'Null'
                                            y1 = 'Null'
                                            y2 = 'Null'
                                            favorite1 = True
                                        elif team1linelist[each] == None and team2linelist[each] != None:
                                            favorite2 = True
                                            x1 = 'Null'
                                            y1 = 'Null'
                                            x2 = float(team2linelist[each].strip().split(' ')[0])
                                            try:
                                                y2 = float(team2linelist[each].strip().split(' ')[1])
                                            except IndexError:
                                                y2 = 0
                                        elif team2linelist[each] == None and team1linelist[each] != None:
                                            favorite1 = True
                                            x1 = float(team1linelist[each].strip().split(' ')[0])
                                            try:
                                                y1 = float(team1linelist[each].strip().split(' ')[0])
                                            except IndexError:
                                                y1 = 0
                                            x2 = 'Null'
                                            y2 = 'Null'
                                        elif team1linelist[each] != None and team2linelist[each] != None:                           
                                            spacesplit1 = team1linelist[each].strip().split(' ')
                                            osplit1 = team1linelist[each].strip().split('o')
                                            usplit1 = team1linelist[each].strip().split('u')
                                            
                                            if len(spacesplit1) == len(osplit1) == len(usplit1) == 1:
                                                try:
                                                    x1 = float(team1linelist[each].strip())
                                                except ValueError:
                                                    if team1linelist[each].strip() == 'PK':
                                                        x1 = 0
                                                        favorite1 = True
                                                y1 = 0
                                            elif len(spacesplit1) == 1 and len(osplit1) == 1 and len(usplit1) == 2:
                                                x1 = float(team1linelist[each].strip().split('u')[0])
                                                favorite2 = True
                                                y1 = float(team1linelist[each].strip().split('u')[1])
                                            elif len(spacesplit1) == 1 and len(osplit1) == 2 and len(usplit1) == 1:
                                                x1 = float(team1linelist[each].strip().split('o')[0])
                                                favorite2 = True
                                                y1 = float(team1linelist[each].strip().split('o')[1])      
                                            elif len(spacesplit1) == 2 and len(osplit1) == 1 and len(usplit1) == 1:
                                                try:
                                                    x1 = float(team1linelist[each].strip().split(' ')[0])
                                                    favorite1 = True
                                                except ValueError:
                                                    if team1linelist[each].strip().split(' ')[0] == 'PK':
                                                        x1 = 0
                                                        favorite1 = True
                                                try:
                                                    y1 = float(team1linelist[each].strip().split(' ')[1])
                                                except ValueError:
                                                    if team1linelist[each].strip().split(' ')[1] == 'EVEN':
                                                        y1 = 0
                
                                            spacesplit2 = team2linelist[each].strip().split(' ')
                                            osplit2 = team2linelist[each].strip().split('o')
                                            usplit2 = team2linelist[each].strip().split('u')                                        
                                            if len(spacesplit2) == len(osplit2) == len(usplit2) == 1:
                                                try:
                                                    x2 = float(team2linelist[each].strip())
                                                except ValueError:
                                                    if team2linelist[each].strip() == 'PK':
                                                        x2 = 0
                                                        favorite2 = True
                                                y2 = 0
                                            elif len(spacesplit2) == 1 and len(osplit2) == 1 and len(usplit2) == 2:
                                                x2 = float(team2linelist[each].strip().split('u')[0])
                                                favorite1 = True
                                                y2 = float(team2linelist[each].strip().split('u')[1])
                                            elif len(spacesplit2) == 1 and len(osplit2) == 2 and len(usplit2) == 1:
                                                x2 = float(team2linelist[each].strip().split('o')[0])
                                                favorite1 = True
                                                y2 = float(team2linelist[each].strip().split('o')[1])      
                                            elif len(spacesplit2) == 2 and len(osplit2) == 1 and len(usplit2) == 1:
                                                try:
                                                    x2 = float(team2linelist[each].strip().split(' ')[0])
                                                    favorite2 = True
                                                except ValueError:
                                                    if team2linelist[each].strip().split(' ')[0] == 'PK':
                                                        x2 = 0
                                                        favorite2 = True
                                                try:
                                                    y2 = float(team2linelist[each].strip().split(' ')[1])
                                                except ValueError:
                                                    if team2linelist[each].strip().split(' ')[1] == 'EVEN':
                                                        y2 = 0   
        
                                        if favorite1 == favorite2 == True:
                                            pass
                                        elif favorite1 == True and favorite2 == False:
                                            favorite = 1
                                            try:
                                                line = float(x1)
                                            except ValueError:
                                                if x1 == 'Null':
                                                    line = str(x1)
                                            try:
                                                linejuice = float(y1)
                                            except ValueError:
                                                if y1 == 'Null':
                                                    linejuice = str(y1)
                                            try:
                                                overunder = float(x2)
                                            except ValueError:
                                                if x2 == 'Null':
                                                    overunder = str(x2)
                                            try:
                                                overunderjuice = float(y2) 
                                            except ValueError:
                                                if y2 == 'Null':
                                                    overunderjuice = str(y2)                                
                                        elif favorite1 == False and favorite2 == True:
                                            favorite = 2
                                            try:
                                                line = float(x2)
                                            except ValueError:
                                                if x2 == 'Null':
                                                    line = str(x2)
                                            try:
                                                linejuice = float(y2)
                                            except ValueError:
                                                if y2 == 'Null':
                                                    linejuice = str(y2)
                                            try:
                                                overunder = float(x1)
                                            except ValueError:
                                                if x1 == 'Null':
                                                    overunder = str(x1)
                                            try:
                                                overunderjuice = float(y1) 
                                            except ValueError:
                                                if y1 == 'Null':
                                                    overunderjuice = str(y1)
                                        elif favorite1 == favorite2 == False:
                                            if x1 < 0:
                                                favorite = 1
                                                try:
                                                    line = float(x1)
                                                except ValueError:
                                                    if x1 == 'Null':
                                                        line = str(x1)
                                                try:
                                                    linejuice = float(y1)
                                                except ValueError:
                                                    if y1 == 'Null':
                                                        linejuice = str(y1)
                                                try:
                                                    overunder = float(x2)
                                                except ValueError:
                                                    if x2 == 'Null':
                                                        overunder = str(x2)
                                                try:
                                                    overunderjuice = float(y2) 
                                                except ValueError:
                                                    if y2 == 'Null':
                                                        overunderjuice = str(y2)                                   
                                            elif x2 < 0:
                                                favorite = 2
                                                try:
                                                    line = float(x2)
                                                except ValueError:
                                                    if x2 == 'Null':
                                                        line = str(x2)
                                                try:
                                                    linejuice = float(y2)
                                                except ValueError:
                                                    if y2 == 'Null':
                                                        linejuice = str(y2)
                                                try:
                                                    overunder = float(x1)
                                                except ValueError:
                                                    if x1 == 'Null':
                                                        overunder = str(x1)
                                                try:
                                                    overunderjuice = float(y1) 
                                                except ValueError:
                                                    if y1 == 'Null':
                                                        overunderjuice = str(y1) 
                                        if favorite == 1:
                                            game = [gameday, team1, team2, line, linejuice, overunder, overunderjuice, moneyline1, moneyline2, score1, score2, 1]
                                        elif favorite == 2:
                                            game = [gameday, team2, team1, line, linejuice, overunder, overunderjuice, moneyline2, moneyline1, score2, score1, 0]
                                        if favorite == None:
                                            favorite_errors.append(url)
                                        else:
                                            all_games.append(game)
                                            oddsinsert = []
                                            oddsinsertx = None
                                            oddslist = []
                                            initialoddsinsert = None
                                            add_odds = None
                                            oddsinsert.append("('"+game[0]+"', '"+str(game[1])+"', '"+str(game[2])+"', "+str(game[3])+", "+str(game[4])+", "+str(game[5])+", "+str(game[6])+", "+str(game[7])+", "+str(game[8])+", "+str(game[9])+", "+str(game[10])+", "+str(game[11])+")")
                                            oddsinsertx = ','.join(oddsinsert)
                                            oddslist = ['INSERT INTO oddsdata VALUES', oddsinsertx, ';']
                                            initialoddsinsert = ' '.join(oddslist)  
                                            add_odds = initialoddsinsert  
                                            cursor.execute('SET foreign_key_checks = 0;')
                                            cursor.execute(add_odds)
                                            cnx.commit()
                                            cursor.execute('SET foreign_key_checks = 1;')
                                except IntegrityError:
                                    pass
                        else:
                            error = (team1namelist[each][4:], team2namelist[each][4:])
                            nonfbsteams.append(error)
                    else:
                        all_errors.append(url)  
        cursor.close()
        cnx.close()                                 
    
    
    def update_basset():
        passcode = 'ibm1234'
        import requests
        from lxml import html
        import re
        import mysql.connector  
        from mysql.connector import IntegrityError
    
        cnx = mysql.connector.connect(user='root', password=passcode,
                                      host='127.0.0.1',
                                      database='ncaa')    
        cursor = cnx.cursor() 
        teamnames = ['Air Force', 'Akron', 'Alabama', 'App State', 'Arizona', 'Arizona St', 'Arkansas', 'Arkansas St', 'Army', 'Auburn', 'BYU', 'Ball State', 'Baylor', 'Boise State', 'Boston Col', 'Bowling Grn', 'Buffalo', 'California', 'Central FL', 'Central Mich', 'Charlotte', 'Cincinnati', 'Clemson', 'Coastal Car', 'Colorado', 'Colorado St', 'Connecticut', 'Duke', 'E Carolina', 'E Michigan', 'Fla Atlantic', 'Florida', 'Florida Intl', 'Florida St', 'Fresno St', 'GA Southern', 'GA Tech', 'Georgia', 'Georgia State', 'Hawaii', 'Houston', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Iowa State', 'Kansas', 'Kansas St', 'Kent State', 'Kentucky', 'LA Lafayette', 'LA Monroe', 'LA Tech', 'LSU', 'Louisville', 'Marshall', 'Maryland', 'Memphis', 'Miami (FL)', 'Miami (OH)', 'Michigan', 'Michigan St', 'Middle Tenn', 'Minnesota', 'Miss State', 'Mississippi', 'Missouri', 'N Carolina', 'N Illinois', 'N Mex State', 'NC State', 'Navy', 'Nebraska', 'Nevada', 'New Mexico', 'North Texas', 'Northwestern', 'Notre Dame', 'Ohio', 'Ohio State', 'Oklahoma', 'Oklahoma St', 'Old Dominion', 'Oregon', 'Oregon St', 'Penn State', 'Pittsburgh', 'Purdue', 'Rice', 'Rutgers', 'S Alabama', 'S Carolina', 'S Florida', 'S Methodist', 'S Mississippi', 'San Diego St', 'San Jose St', 'Stanford', 'Syracuse', 'TX Christian', 'TX El Paso', 'TX-San Ant', 'Temple', 'Tennessee', 'Texas', 'Texas A&M', 'Texas State', 'Texas Tech', 'Toledo', 'Troy', 'Tulane', 'Tulsa', 'U Mass', 'UAB', 'UCLA', 'UNLV', 'USC', 'Utah', 'Utah State', 'VA Tech', 'Vanderbilt', 'Virginia', 'W Kentucky', 'W Michigan', 'W Virginia', 'Wake Forest', 'Wash State', 'Washington', 'Wisconsin', 'Wyoming']
        bassetnames = ['Air Force', 'Akron', 'Alabama',  'Appalachian State', 'Arizona', 'Arizona State', 'Arkansas', 'Arkansas State', 'Army', 'Auburn', 'Brigham Young', 'Ball State', 'Baylor', 'Boise State', 'Boston College', 'Bowling Green',  'Buffalo', 'California', 'Central Florida', 'Central Michigan',  'North Carolina - Charlot', 'Cincinnati', 'Clemson', 'Coastal Carolina', 'Colorado', 'Colorado State', 'Connecticut', 'Duke', 'East Carolina', 'Eastern Michigan',  'Florida Atlantic',  'Florida','Florida International', 'Florida State', 'Fresno State','Georgia Southern', 'Georgia Tech', 'Georgia', 'Georgia State', 'Hawaii', 'Houston', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Iowa State', 'Kansas', 'Kansas State', 'Kent State', 'Kentucky', 'Louisiana - Lafayette', 'Northeast Louisiana', 'Louisiana Tech',  'Louisiana State', 'Louisville', 'Marshall', 'Maryland', 'Memphis', 'Miami - Florida', 'Miami - Ohio', 'Michigan', 'Michigan State', 'Middle Tennessee State', 'Minnesota', 'Mississippi State',  'Mississippi','Missouri',  'North Carolina', 'Northern Illinois',  'New Mexico State','North Carolina State','Navy', 'Nebraska', 'Nevada - Reno',  'New Mexico', 'North Texas',  'Northwestern', 'Notre Dame', 'Ohio', 'Ohio State', 'Oklahoma', 'Oklahoma State', 'Old Dominion', 'Oregon', 'Oregon State', 'Penn State', 'Pittsburgh', 'Purdue', 'Rice', 'Rutgers', 'South Alabama', 'South Carolina', 'South Florida', 'Southern Methodist', 'Southern Mississippi', 'San Diego State', 'San Jose State','Stanford', 'Syracuse',  'Texas Christian',  'Texas - El Paso', 'Texas - San Antonio', 'Temple', 'Tennessee', 'Texas','Texas A&M','Texas State - San Marcos', 'Texas Tech', 'Toledo', 'Troy State', 'Tulane', 'Tulsa',   'Massachusetts','Alabama - Birmingham','California - Los Angeles','Nevada - Las Vegas','Southern California',  'Utah', 'Utah State', 'Virginia Tech', 'Vanderbilt', 'Virginia', 'Western Kentucky', 'Western Michigan', 'West Virginia', 'Wake Forest', 'Washington State',  'Washington','Wisconsin', 'Wyoming']
        monthdict = {'Jan':1, 'Feb':2, 'Mar':3, 'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12}
        bassetteamsdict = {}
        for i in range(0, len(teamnames)):
            bassetteamsdict[teamnames[i]] = bassetnames[i]
        seasonrange = ['17']
        weekrange = ['08','09','10','11','12','13','14','15','16','17','18','19','20']
        for season in seasonrange:
            nextseason = 0
            for week in weekrange:
                try:
                    bassetinsertx = None
                    bassetinsert = []
                    data = None
                    pageContent = None
                    tree = None
                    invalid = None
                    thisdate = None
                    year = None
                    month = None
                    day = None
                    day = None
                    usedate = None
                    r = None
                    teamratings = []
                    if nextseason == 1:
                        pass
                    else:
                        url = 'http://gmbassett.nfshost.com/football/col_%swk%spred.html' % (season,week)
                        pageContent=requests.get(url)
                        tree = html.fromstring(pageContent.content)
                        invalid = tree.xpath('//html/head/title/text()')
                        if invalid[0] == '404 Not Found':
                            if week == '00':
                                pass
                            else:
                                nextseason = 1
                        else:
                            thisdate = tree.xpath('//html/body/h1/text()')[0].strip().split(' ')
                            bowlcheck = thisdate[1]
                            if bowlcheck == 'Bowl':
                                nextseason = 1
                                pass
                            else:
                                year = thisdate[0]
                                for q in range(0, len(thisdate)):
                                    if thisdate[q] == 'Forecast':
                                        try:
                                            month = str(monthdict[thisdate[q-1][:-1]])
                                            r = q-2
                                        except KeyError:
                                            for w in range(0, q-1):
                                                try:
                                                    month = str(monthdict[thisdate[w][:-1]])
                                                    r = w-1
                                                except KeyError:
                                                    continue                                   
                                        try:
                                            day = str(int(thisdate[r]))
                                        except ValueError:
                                            try:
                                                day = str(int(thisdate[r][1:]))
                                            except ValueError:
                                                day = str(int(thisdate[r].split('-')[1]))     
                                if month == None:
                                    for q in range(0, len(thisdate)):
                                        if thisdate[q] == 'for':
                                            month = str(monthdict[thisdate[q-1][:-1]])
                                            day = str(int(thisdate[q-2][1:]))
                                usedate = year+'-'+month+'-'+day
                                data = str(tree.xpath('/html/body/pre/text()'))
                                for team in teamnames:
                                    emptypass = 0
                                    partialpass = 0
                                    nameindex = None
                                    namematch = None
                                    nameloc = None
                                    rank = None
                                    ratingtuple = None
                                    ratingtuplelist = []
                                    nameindex = bassetteamsdict[team]+'  '
                                    namematch = [m.start() for m in re.finditer(nameindex, data)]
                                    for v in range(0, len(namematch)):
                                        emptypass = 0
                                        partialpass = 0
                                        nameloc = None
                                        rank = None
                                        ratingtuple = None
                                        try:
                                            nameloc = namematch[v]
                                        except IndexError:
                                            emptypass = 1
                                        if emptypass == 0:
                                            try:
                                                int(data[nameloc-2])
                                            except ValueError:
                                                partialpass = 1
                                        if emptypass == 0 and partialpass == 0:
                                            try:
                                                int(data[nameloc-4])
                                                rank = int(data[nameloc-4:nameloc-1])
                                            except ValueError:
                                                rank = int(data[nameloc-3:nameloc-1])
                                            ratingtuple = (team, rank)
                                            ratingtuplelist.append(ratingtuple)
                                    if len(ratingtuplelist) == 1:
                                        teamratings.append(ratingtuplelist[0])
                                    elif len(ratingtuplelist) == 2:
                                        if ratingtuplelist[0] == ratingtuplelist[1]:
                                            teamratings.append(ratingtuplelist[0])
                                        else:
                                            teamratings.append(ratingtuplelist[0])
                                            teamratings.append(ratingtuplelist[1])
                                    elif len(ratingtuplelist) > 2:
                                        for every in ratingtuplelist:
                                            teamratings.append(every)
                                if len(teamratings) > 0:
                                    if week == '15' and season == '13':
                                        usedate = '2013-05-14'
                                    for team in teamratings:
                                            bassetinsert.append("('"+team[0]+"', '"+str(usedate)+"', "+str(team[1])+")")
                                    bassetinsertx = ','.join(bassetinsert)
                                    bassetlist = ['INSERT INTO bassetratings VALUES', bassetinsertx, ';']
                                    initialbassetinsert = ' '.join(bassetlist)  
                                    add_basset = initialbassetinsert  
                                    print(usedate)
                                    cursor.execute('SET foreign_key_checks = 0;')
                                    cursor.execute(add_basset)
                                    cnx.commit()
                                    cursor.execute('SET foreign_key_checks = 1;')
                except IntegrityError:
                    pass
        cursor.close()
        cnx.close()        

    
    
    def update_massey():
        passcode = 'ibm1234'        
        cnx = mysql.connector.connect(user='root', password=passcode,
                              host='127.0.0.1',
                              database='ncaa')    
        cursor = cnx.cursor()    
        teamnames = ['Air Force', 'Akron', 'Alabama', 'App State', 'Arizona', 'Arizona St', 'Arkansas', 'Arkansas St', 'Army', 'Auburn', 'BYU', 'Ball State', 'Baylor', 'Boise State', 'Boston Col', 'Bowling Grn', 'Buffalo', 'California', 'Central FL', 'Central Mich', 'Charlotte', 'Cincinnati', 'Clemson', 'Coastal Car', 'Colorado', 'Colorado St', 'Connecticut', 'Duke', 'E Carolina', 'E Michigan', 'Fla Atlantic', 'Florida', 'Florida Intl', 'Florida St', 'Fresno St', 'GA Southern', 'GA Tech', 'Georgia', 'Georgia State', 'Hawaii', 'Houston', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Iowa State', 'Kansas', 'Kansas St', 'Kent State', 'Kentucky', 'LA Lafayette', 'LA Monroe', 'LA Tech', 'LSU', 'Louisville', 'Marshall', 'Maryland', 'Memphis', 'Miami (FL)', 'Miami (OH)', 'Michigan', 'Michigan St', 'Middle Tenn', 'Minnesota', 'Miss State', 'Mississippi', 'Missouri', 'N Carolina', 'N Illinois', 'N Mex State', 'NC State', 'Navy', 'Nebraska', 'Nevada', 'New Mexico', 'North Texas', 'Northwestern', 'Notre Dame', 'Ohio', 'Ohio State', 'Oklahoma', 'Oklahoma St', 'Old Dominion', 'Oregon', 'Oregon St', 'Penn State', 'Pittsburgh', 'Purdue', 'Rice', 'Rutgers', 'S Alabama', 'S Carolina', 'S Florida', 'S Methodist', 'S Mississippi', 'San Diego St', 'San Jose St', 'Stanford', 'Syracuse', 'TX Christian', 'TX El Paso', 'TX-San Ant', 'Temple', 'Tennessee', 'Texas', 'Texas A&M', 'Texas State', 'Texas Tech', 'Toledo', 'Troy', 'Tulane', 'Tulsa', 'U Mass', 'UAB', 'UCLA', 'UNLV', 'USC', 'Utah', 'Utah State', 'VA Tech', 'Vanderbilt', 'Virginia', 'W Kentucky', 'W Michigan', 'W Virginia', 'Wake Forest', 'Wash State', 'Washington', 'Wisconsin', 'Wyoming', 'W Kentucky', 'Middle Tenn', 'San Jose St', 'LA Lafayette', 'LA Monroe']
        teamlist = ['Air Force', 'Akron', 'Alabama', 'Appalachian St', 'Arizona', 'Arizona St', 'Arkansas', 'Arkansas St', 'Army','Auburn', 'BYU', 'Ball St', 'Baylor', 'Boise St', 'Boston College', 'Bowling Green', 'Buffalo', 'California', 'UCF','C Michigan', 'Charlotte', 'Cincinnati', 'Clemson', 'Coastal Car', 'Colorado', 'Colorado St', 'Connecticut', 'Duke','East Carolina','E Michigan',  'FL Atlantic', 'Florida', 'Florida Intl', 'Florida St', 'Fresno St', 'Ga Southern',  'Georgia Tech','Georgia', 'Georgia St', 'Hawaii', 'Houston', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Iowa St', 'Kansas', 'Kansas St', 'Kent', 'Kentucky', 'ULL', 'ULM', 'Louisiana Tech', 'LSU', 'Louisville', 'Marshall', 'Maryland','Memphis', 'Miami FL', 'Miami OH', 'Michigan', 'Michigan St', 'MTSU', 'Minnesota', 'Mississippi St', 'Mississippi', 'Missouri', 'North Carolina', 'N Illinois', 'New Mexico St', 'NC State', 'Navy', 'Nebraska', 'Nevada', 'New Mexico', 'North Texas', 'Northwestern', 'Notre Dame', 'Ohio', 'Ohio St', 'Oklahoma', 'Oklahoma St', 'Old Dominion', 'Oregon', 'Oregon St', 'Penn St', 'Pittsburgh', 'Purdue', 'Rice', 'Rutgers', 'South Alabama', 'South Carolina', 'South Florida',  'SMU', 'Southern Miss', 'San Diego St', 'San Jose St', 'Stanford', 'Syracuse', 'TCU', 'UTEP', 'UT San Antonio', 'Temple', 'Tennessee', 'Texas', 'Texas A&M', 'Texas St', 'Texas Tech', 'Toledo', 'Troy', 'Tulane', 'Tulsa', 'Massachusetts', 'UAB', 'UCLA', 'UNLV', 'USC', 'Utah', 'Utah St', 'Virginia Tech', 'Vanderbilt', 'Virginia',  'WKU', 'W Michigan', 'West Virginia', 'Wake Forest',  'Washington St', 'Washington',  'Wisconsin', 'Wyoming', 'W Kentucky', 'Middle Tenn St', 'San Jos\xe9 State', 'LA Lafayette', 'LA Monroe']
        espnlist = ['Air Force', 'Akron', 'Alabama', 'Appalachian State', 'Arizona', 'Arizona State', 'Arkansas', 'Arkansas St', 'Army','Auburn', 'BYU', 'Ball State', 'Baylor', 'Boise State', 'Boston College', 'Bowling Green', 'Buffalo', 'California', 'UCF','Central Michigan', 'Charlotte', 'Cincinnati', 'Clemson', 'Coastal Car', 'Colorado', 'Colorado State', 'Connecticut', 'Duke','East Carolina','E Michigan',  'FL Atlantic', 'Florida', 'Florida Intl', 'Florida State', 'Fresno State', 'Ga Southern',  'Georgia Tech','Georgia', 'Georgia St', "Hawai'i", 'Houston', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Iowa State', 'Kansas', 'Kansas State', 'Kent State', 'Kentucky', 'ULL', 'ULM', 'Louisiana Tech', 'LSU', 'Louisville', 'Marshall', 'Maryland','Memphis', 'Miami', 'Miami OH', 'Michigan', 'Michigan State', 'MTSU', 'Minnesota', 'Mississippi State', 'Ole Miss', 'Missouri', 'North Carolina', 'Northern Illinois', 'New Mexico St', 'NC State', 'Navy', 'Nebraska', 'Nevada', 'New Mexico', 'North Texas', 'Northwestern', 'Notre Dame', 'Ohio', 'Ohio State', 'Oklahoma', 'Oklahoma State', 'Old Dominion', 'Oregon', 'Oregon State', 'Penn State', 'Pittsburgh', 'Purdue', 'Rice', 'Rutgers', 'South Alabama', 'South Carolina', 'South Florida',  'SMU', 'Southern Mississippi', 'San Diego State', 'San Jose State', 'Stanford', 'Syracuse', 'TCU', 'UTEP', 'UT San Antonio', 'Temple', 'Tennessee', 'Texas', 'Texas A&M', 'Texas State', 'Texas Tech', 'Toledo', 'Troy', 'Tulane', 'Tulsa', 'Massachusetts', 'UAB', 'UCLA', 'UNLV', 'USC', 'Utah', 'Utah State', 'Virginia Tech', 'Vanderbilt', 'Virginia',  'Western Kentucky', 'Western Michigan', 'West Virginia', 'Wake Forest',  'Washington State', 'Washington',  'Wisconsin', 'Wyoming']
        teamsdict = {}
        for i in range(0, len(teamnames)):
            teamsdict[teamlist[i]] = teamnames[i]
        espndict = {}
        for i in range(0, len(espnlist)):
            espndict[espnlist[i]] = teamnames[i]
        year = '2017'
        for qwerty in range(8, 15):
            try:
                url = None
                pageContent = None
                tree = None
                apteams = None
                uspoll = None
                playoffpoll = None
                bcspoll = None
                url = 'http://www.espn.com/college-football/rankings/_/week/%s/year/%s/seasontype/2'%(qwerty+1, year)
                pageContent=requests.get(url)
                tree = html.fromstring(pageContent.content)
                
                if tree.xpath('//section[@class="col-b"]/div[3]/div[1]/div[1]/h2[1]/text()')[0] == 'AP Top 25':
                    apteams = tree.xpath('//section[@class="col-b"]/div[3]/div[1]/div[1]/div[1]/table/tbody/tr/td[2]/a[2]/span/text()')
                elif tree.xpath('//section[@class="col-b"]/div[3]/div[1]/div[1]/h2[1]/text()')[0] == 'Coaches Poll':
                    uspoll = tree.xpath('//section[@class="col-b"]/div[3]/div[1]/div[2]/div[1]/table/tbody/tr/td[2]/a[2]/span/text()')
                elif tree.xpath('//section[@class="col-b"]/div[3]/div[1]/div[1]/h2[1]/text()')[0] == 'College Football Playoff Rankings':
                    playoffpoll = tree.xpath('//section[@class="col-b"]/div[3]/div[1]/div[1]/div[1]/table/tbody/tr/td[2]/a[2]/span/text()')
                elif tree.xpath('//section[@class="col-b"]/div[3]/div[1]/div[1]/h2[1]/text()')[0] == 'BCS Standings':
                    bcspoll = tree.xpath('//section[@class="col-b"]/div[3]/div[1]/div[1]/div[1]/table/tbody/tr/td[2]/a[2]/span/text()')
            
                if tree.xpath('//section[@class="col-b"]/div[3]/div[1]/div[2]/h2[1]/text()')[0] == 'AP Top 25':
                    apteams = tree.xpath('//section[@class="col-b"]/div[3]/div[1]/div[2]/div[1]/table/tbody/tr/td[2]/a[2]/span/text()')
                elif tree.xpath('//section[@class="col-b"]/div[3]/div[1]/div[2]/h2[1]/text()')[0] == 'Coaches Poll':
                    uspoll = tree.xpath('//section[@class="col-b"]/div[3]/div[1]/div[2]/div[1]/table/tbody/tr/td[2]/a[2]/span/text()')
                elif tree.xpath('//section[@class="col-b"]/div[3]/div[1]/div[2]/h2[1]/text()')[0] == 'BCS Standings':
                    bcspoll = tree.xpath('//section[@class="col-b"]/div[3]/div[1]/div[2]/div[1]/table/tbody/tr/td[2]/a[2]/span/text()')
                
                if apteams == None:
                    if tree.xpath('//section[@class="col-b"]/div[3]/div[2]/div[1]/h2[1]/text()')[0] == 'AP Top 25':
                        apteams = tree.xpath('//section[@class="col-b"]/div[3]/div[2]/div[1]/div[1]/table/tbody/tr/td[2]/a[2]/span/text()')
                if uspoll == None:
                    if tree.xpath('//section[@class="col-b"]/div[3]/div[2]/div[1]/h2[1]/text()')[0] == 'Coaches Poll':
                        uspoll = tree.xpath('//section[@class="col-b"]/div[3]/div[2]/div[1]/div[1]/table/tbody/tr/td[2]/a[2]/span/text()') 
                
                monthdict = {'August':'08', 'September':'09', 'October':'10', 'November':'11', 'December':'12', 'January':'01'}
                url = None
                pageContent = None
                tree = None
                rankingabbrev = None
                headers = None
                rankabbrev = None
                date = None
                month = None
                day = None
                yearx = None
                
                url = 'https://www.masseyratings.com/cf/arch/compare%s-%s.htm'%(year, qwerty)
                pageContent=requests.get(url)
                tree = html.fromstring(pageContent.content)
                try:
                    rankingabbrev = tree.xpath('//html/body/pre/font/text()')[0].split(' ')
                except IndexError:
                    url = 'https://www.masseyratings.com/cf/arch/compare%s-current.htm'%(year)
                    pageContent=requests.get(url)
                    tree = html.fromstring(pageContent.content)
                    rankingabbrev = tree.xpath('//html/body/pre/font/text()')[0].split(' ')            
                headers = []
                rankabbrev = []
                
                if year == '2017':
                    if qwerty != 0:
                        month, day, yearx = tree.xpath('//html/body/table/tr/td/table/tr/td/h4/text()')[0].split(' ')[1:4]
                        month = monthdict[month]
                        day = day[:-1]
                        if len(day) == 1:
                            day = '0'+day
                        date = yearx+'-'+month+'-'+day
                    else:
                        date = '2017-09-01'  
                        
                d = None
                daysahead = None
                d = datetime.date(int(date.split('-')[0]), int(date.split('-')[1]), int(date.split('-')[2]))
                daysahead = 5 - d.weekday()
                if daysahead == 0:
                    daysahead = 7
                if daysahead == -1:
                    daysahead = 6
                date = d + datetime.timedelta(daysahead)
                yearx = str(date.year)
                month = str(date.month)
                if len(month) == 1:
                    month = '0'+month
                day = str(date.day)
                if len(day) == 1:
                    day = '0'+day        
                date = yearx+'-'+month+'-'+day        
                for each in rankingabbrev:
                    if each.split(',')[0] != '' and each.split(',')[0] != 'Team' and each.split(',')[0] != 'Conf' and each.split(',')[0] != 'Record':
                            headers.append(each.split(',')[0])
                            if each.split(',')[0] != 'Rank':
                                rankabbrev.append(each.split(',')[0])
                
                teamsandrankings = None
                rawteams = None
                masseyteams = None
                start = None
                endcontent = None
                startcontent = None
                rawratings = None
                ratings = None         
                teamsandrankings = tree.xpath('//html/body/pre/a/text()')[:len(rankabbrev)]
                rawteams = tree.xpath('//html/body/pre/a/text()')[len(rankabbrev):]
                masseyteams = []
                for each in rawteams:
                    try:
                        masseyteams.append(teamsdict[each])
                    except KeyError:
                        pass
                
                start = [m.start() for m in re.finditer('Mean Median St.Dev', str(pageContent.content))]
                endcontent = [m.start() for m in re.finditer('----------', str(pageContent.content))]
                startcontent = str(pageContent.content)[start[0]:endcontent[0]]
                rawratings = startcontent.split(' ')
                ratings = []
                for every in rawratings:
                    try:
                        every.split('.')[1]
                    except IndexError:
                        try:
                            ratings.append(int(every))
                        except ValueError:
                            try:
                                ratings.append(int(every.split('<')[0]))
                            except ValueError:
                                try:
                                    ratings.append(int(every.split('>')[1]))
                                except IndexError:
                                    pass
                                except ValueError:
                                    try:
                                        ratings.append(int(every.split('>')[1].split('<')[0]))
                                    except ValueError:
                                        pass
                
                playofflist = None
                bcslist = None
                playoffcol = None
                bcscol = None
                playoffspotx = None
                bcsspotx = None
                aplist = None
                apcol = None
                uscol = None
                grid = None
                indices = None
                usspotx = None
                apspotx = None
                dataset = None
                
                aplist = []
                for team in apteams:
                    try:
                        aplist.append(espndict[team])
                    except KeyError:
                        aplist.append(espndict[unicodedata.normalize('NFKD', team).encode('ascii','ignore')])
                uslist = []
                for team in uspoll:
                    try:
                        uslist.append(espndict[team])
                    except KeyError:
                        uslist.append(espndict[unicodedata.normalize('NFKD', team).encode('ascii','ignore')])
                apcol = []
                for team in masseyteams:
                    if team in aplist:
                        apcol.append(aplist.index(team)+1)
                    else:
                        apcol.append(None)
                uscol = []
                for team in masseyteams:
                    if team in uslist:
                        uscol.append(uslist.index(team)+1)
                    else:
                        uscol.append(None)        
                        
                grid = pd.DataFrame(columns = rankabbrev)
                grid['AP'] = np.array(apcol)
                grid['USA'] = np.array(uscol)   
                indices = [i for i, x in enumerate(headers) if x == "Rank"]
                usspotx = headers.index('USA')
                apspotx = headers.index('AP')
                  
                if playoffpoll != None and 'CFP' in headers:    
                    playofflist = []
                    for team in playoffpoll:
                        playofflist.append(espndict[team])         
                    playoffcol = []
                    for team in masseyteams:
                        if team in playofflist:
                            playoffcol.append(playofflist.index(team)+1)
                        else:
                            playoffcol.append(None)   
                    grid['CFP'] = np.array(playoffcol)   
                    playoffspotx = headers.index('CFP')
            
                if bcspoll != None or 'BSC' in headers:    
                    bcslist = []
                    for team in bcspoll:
                        try:
                            bcslist.append(espndict[team])      
                        except KeyError:
                            bcslist.append(espndict[unicodedata.normalize('NFKD', team).encode('ascii','ignore')])
                    bcscol = []
                    for team in masseyteams:
                        if team in bcslist:
                            bcscol.append(bcslist.index(team)+1)
                        else:
                            bcscol.append(None)   
                    if 'BCS' in headers:
                        grid['BCS'] = np.array(bcscol)   
                        bcsspotx = headers.index('BCS')       
            
                dataset = pd.DataFrame()
                stop = 0
                dbrow = 0
                dataspot = 0
                while stop != 1:
                    line = []
                    hcol = 0
                    end = 0
                    while end != 1: 
                        if year == '2017' and qwerty >= 1 and qwerty < 7 and hcol in [headers.index('ENG')]  and dbrow == masseyteams.index('UAB'):
                            line.append(None)
                            hcol += 1 
                        elif year == '2017' and qwerty == 7 and hcol in [headers.index('RME')]  and dbrow == masseyteams.index('Northwestern'):
                            line.append(None)
                            hcol += 1                     
                   
                        elif hcol in indices:
                            if ratings[dataspot] == dbrow+1:
                                    dataspot += 1
                                    hcol += 1
                            else:
                                    end = 1
                                    
                        elif year == '2017' and qwerty in [8,9,11,12] and dbrow == masseyteams.index('Coastal Car') and hcol in [headers.index('KAM'), headers.index('SOR')]:
                            line.append(None)
                            hcol+=1
                        
                        elif year == '2017' and qwerty in [12] and dbrow == masseyteams.index('Ball State') and hcol in [headers.index('NUT')]:
                            line.append(None)
                            hcol+=1
            
                        elif year == '2017' and qwerty >=10 and dbrow == masseyteams.index('Coastal Car') and hcol in [headers.index('KAM')]:
                            line.append(None)
                            hcol+=1
        
                        elif year == '2017' and qwerty in [12] and dbrow in [masseyteams.index('Washington'), masseyteams.index('Ohio')] and hcol in [headers.index('JWN')]:
                            line.append(None)
                            hcol+=1
                            
                        elif year == '2017' and qwerty in [11] and dbrow in [masseyteams.index('Minnesota'), masseyteams.index('Temple'), masseyteams.index('Rice')] and hcol in [headers.index('JWN')]:
                            line.append(None)
                            hcol+=1
                                
                        elif hcol == usspotx:
                                try:
                                    if ratings[dataspot] == grid['USA'][dbrow]:
                                        line.append(ratings[dataspot])
                                        hcol += 1
                                        dataspot += 1
                                    elif grid['USA'][dbrow] == None:
                                        line.append(None)
                                        hcol += 1
                                    else:
                                        try:
                                            if ratings[dataspot] == grid['USA'][dbrow]-1 or ratings[dataspot] == grid['USA'][dbrow]+1:
                                                line.append(ratings[dataspot])
                                                hcol += 1
                                                dataspot += 1
                                        except TypeError:
                                            if grid['USA'][dbrow] == None:
                                                line.append(None)
                                                hcol +=1
                                            else:
                                                end = 1
                                except IndexError:
                                    if len(grid) == len(dataset)+1:
                                        if  len(line)+1 == len(list(grid)) or  len(line)+2 == len(list(grid)):
                                           line.append(None)
                                        elif len(line) + 3 == len(list(grid)):
                                            line.append(None)
                                            hcol += 1
                                           
                                           
                                           
                                           
                        elif hcol == apspotx:
                            try:
                                if ratings[dataspot] == grid['AP'][dbrow]:
                                    line.append(ratings[dataspot])
                                    hcol +=1
                                    dataspot += 1
                                elif grid['AP'][dbrow] == None:
                                    line.append(None)
                                    hcol += 1
                                else:
                                    try:
                                        if ratings[dataspot] == grid['AP'][dbrow]-1 or ratings[dataspot] == grid['AP'][dbrow]+1:
                                            line.append(ratings[dataspot])
                                            hcol += 1
                                            dataspot += 1
                                    except TypeError:
                                        if grid['AP'][dbrow] == None:
                                            line.append(None)
                                            hcol +=1
                                        else:
                                            end = 1
                            except IndexError:
                                if len(grid) == len(dataset)+1:
                                    if len(line)+1 == len(list(grid)) or len(line)+2 == len(list(grid)): 
                                        line.append(None)
                                    elif len(line) + 4 == len(list(grid)):
                                        line.append(None)
                                        hcol += 1
                                    elif len(line) + 5 == len(list(grid)):
                                        line.append(None)
                                        hcol += 1
                                    elif len(line) + 3 == len(list(grid)):
                                        line.append(None)
                                        hcol += 1                                
                                        
                                        
                                        
                        elif playoffspotx != None and hcol == playoffspotx:
                            try:
                                if ratings[dataspot] == grid['CFP'][dbrow]:
                                    line.append(ratings[dataspot])
                                    hcol +=1
                                    dataspot += 1
                                elif grid['CFP'][dbrow] == None:
                                    line.append(None)
                                    hcol += 1
                                else:
                                    try:
                                        if ratings[dataspot] == grid['CFP'][dbrow]-1 or ratings[dataspot] == grid['CFP'][dbrow]+1:
                                            line.append(ratings[dataspot])
                                            hcol += 1
                                            dataspot += 1
                                    except TypeError:
                                        if grid['CFP'][dbrow] == None:
                                            line.append(None)
                                            hcol +=1
                                        else:
                                            end = 1
                            except IndexError:
                                if len(grid) == len(dataset)+1:
                                    if len(line)+1 == len(list(grid)) or len(line)+2 == len(list(grid)): 
                                        line.append(None)     
                                        
                                        
                                        
                                        
                        elif bcsspotx != None and hcol == bcsspotx:
                            if year in ['2010', '2009', '2008'] and qwerty >= 7:
                                if qwerty >= 10:
                                    if grid['BCS'][dbrow] != None:
                                        line.append(None)
                                        hcol += 1
                                        dataspot += 1
                                    elif grid['BCS'][dbrow] == None:
                                        line.append(None)
                                        hcol += 1
                                else:
                                    line.append(None)
                                    hcol += 1
                                    dataspot += 1                            
                            else:
                                try:
                                    if ratings[dataspot] == grid['BCS'][dbrow]:
                                        line.append(ratings[dataspot])
                                        hcol +=1
                                        dataspot += 1
                                    elif grid['BCS'][dbrow] == None:
                                        line.append(None)
                                        hcol += 1
                                    else:
                                        try:
                                            if ratings[dataspot] == grid['BCS'][dbrow]-1 or ratings[dataspot] == grid['BCS'][dbrow]+1:
                                                line.append(ratings[dataspot])
                                                hcol += 1
                                                dataspot += 1
                                        except TypeError:
                                            if grid['BCS'][dbrow] == None:
                                                line.append(None)
                                                hcol +=1
                                            else:
                                                end = 1
                                except IndexError:
                                    if len(grid) == len(dataset)+1:
                                        if len(line)+1 == len(list(grid)) or len(line)+2 == len(list(grid)):
                                            line.append(None)    
                                        elif len(line) + 5  == len(list(grid)):
                                            line.append(None)
                                            hcol += 1
                                        elif len(line) + 4 == len(list(grid)):
                                            line.append(None)
                                            hcol += 1
            
                       
                        else:
                            line.append(ratings[dataspot])
                            dataspot += 1
                            hcol += 1                
                        if len(line) == len(list(grid)):
                            end = 1   
                    rowentries = {}
                    for v in range(0, len(list(grid))):
                        rowentries[rankabbrev[v]] = line[v]
                    dataset = dataset.append(rowentries, ignore_index = True)    
                    dbrow += 1
                    if len(grid) == len(dataset):
                        stop = 1
                sqllabels = None            
                sqlstaging = None 
                sqldb = None
                masseylist = None
                masseyinsert = None
                masseyinsertx = None
                add_massey = None
                initialmasseyinsert = None
                if len(masseyteams) == len(dataset):        
                    sqllabels = ['Team', 'PIR', 'OSC', 'UCC', 'KPK', 'COF', 'LAZ', 'RWP', 'ACU', 'PAY', 'JTR', 'MTN', 'RT', 'DII', 'ASH', 'FMG', 'RUD', 'MGS', 'ARG', 'SOR', 'WLK', 'SEL', 'HEN', 'HAT', 'MAS', 'HKB', 'DOL', 'MvG', 'KEE', 'FAS', 'SAG', 'BIH', 'HOW', 'GRS', 'ENG', 'JRT', 'STH', 'PGH', 'RTH', 'HNL', 'KH', 'EZ', 'WOB', 'ABC', 'ISR', 'JNK', 'AND', 'COL', 'BOW', 'YCM', 'PCP', 'SOL', 'WOL', 'EFI', 'BSS', 'KRA', 'WIL', 'LOG', 'BWE', 'BBT', 'RTP', 'RFL', 'WWP', 'KLK', 'REW', 'DUN', 'KEL', 'DP', 'BIL', 'ONV', 'KNT', 'MCK', 'BMC', 'SP', 'LSW', 'GLD', 'WEL', 'BCM', 'MCL', 'LSD', 'MAR', 'DOI', 'DOK', 'TRP', 'VRN', 'INP', 'MJS', 'CSL', 'DEZ', 'RME', 'DWI', 'DES', 'KEN', 'MOR', 'DCI', 'CTW', 'FPI', 'PPP', 'MRK', 'TFG', 'MDS', 'BAS', 'GRR', 'BRN', 'GBE', 'RSL', 'PIG', 'SFX', 'FEI', 'CGV', 'KAM', 'CFP', 'S&P', 'RBA', 'NOL', 'PFZ', 'MGN', 'TPR', 'BDF', 'D1A', 'ATC', 'CMV', 'MVP', 'NUT', 'RTB']
                    sqlstaging = pd.DataFrame(columns = sqllabels)
                    sqlstaging['Team'] = masseyteams
                    
                    num = 0
                    for each in sqllabels:
                        if each in list(dataset):
                            sqlstaging[each] = dataset[each]
                            num += 1
                    sqlstaging.fillna('Null', inplace = True)                    
                    sqldb = np.array(sqlstaging) 
            
                for team in sqldb:
                    masseyinsert = []
                    masseyinsert.append("('"+team[0]+"', '"+str(date)+"', "+str(team[1])+', '+str(team[2])+', '+str(team[3])+', '+str(team[4])+', '+str(team[5])+', '+str(team[6])+', '+str(team[7])+', '+str(team[8])+', '+str(team[9])+', '+str(team[10])+', '+str(team[11])+', '+str(team[12])+', '+str(team[13])+', '+str(team[14])+', '+str(team[15])+', '+str(team[16])+', '+str(team[17])+', '+str(team[18])+', '+str(team[19])+', '+str(team[20])+', '+str(team[21])+', '+str(team[22])+', '+str(team[23])+', '+str(team[24])+', '+str(team[25])+', '+str(team[26])+', '+str(team[27])+', '+str(team[28])+', '+str(team[29])+', '+str(team[30])+', '+str(team[31])+', '+str(team[32])+', '+str(team[33])+', '+str(team[34])+', '+str(team[35])+', '+str(team[36])+', '+str(team[37])+', '+str(team[38])+', '+str(team[39])+', '+str(team[40])+', '+str(team[41])+', '+str(team[42])+', '+str(team[43])+', '+str(team[44])+', '+str(team[45])+', '+str(team[46])+', '+str(team[47])+', '+str(team[48])+', '+str(team[49])+', '+str(team[50])+', '+str(team[51])+', '+str(team[52])+', '+str(team[53])+', '+str(team[54])+', '+str(team[55])+', '+str(team[56])+', '+str(team[57])+', '+str(team[58])+', '+str(team[59])+', '+str(team[60])+', '+str(team[61])+', '+str(team[62])+', '+str(team[63])+', '+str(team[64])+', '+str(team[65])+', '+str(team[66])+', '+str(team[67])+', '+str(team[68])+', '+str(team[69])+', '+str(team[70])+', '+str(team[71])+', '+str(team[72])+', '+str(team[73])+', '+str(team[74])+', '+str(team[75])+', '+str(team[76])+', '+str(team[77])+', '+str(team[78])+', '+str(team[79])+', '+str(team[80])+', '+str(team[81])+', '+str(team[82])+', '+str(team[83])+', '+str(team[84])+', '+str(team[85])+', '+str(team[86])+', '+str(team[87])+', '+str(team[88])+', '+str(team[89])+', '+str(team[90])+', '+str(team[91])+', '+str(team[92])+', '+str(team[93])+', '+str(team[94])+', '+str(team[95])+', '+str(team[96])+', '+str(team[97])+', '+str(team[98])+', '+str(team[99])+', '+str(team[100])+', '+str(team[101])+', '+str(team[102])+', '+str(team[103])+', '+str(team[104])+', '+str(team[105])+', '+str(team[106])+', '+str(team[107])+', '+str(team[108])+', '+str(team[109])+', '+str(team[110])+', '+str(team[111])+', '+str(team[112])+', '+str(team[113])+', '+str(team[114])+', '+str(team[115])+', '+str(team[116])+', '+str(team[117])+', '+str(team[118])+', '+str(team[119])+', '+str(team[120])+', '+str(team[121])+', '+str(team[122])+', '+str(team[123])+', '+str(team[124])+")")
                    masseyinsertx = ','.join(masseyinsert)
                    masseylist = ['INSERT INTO masseyratings VALUES', masseyinsertx, ';']
                    initialmasseyinsert = ' '.join(masseylist)  
                    add_massey = initialmasseyinsert  
                    cursor.execute('SET foreign_key_checks = 0;')
                    cursor.execute(add_massey)
                cnx.commit()
                cursor.execute('SET foreign_key_checks = 1;')
                print(date)
            except IntegrityError:
                pass
            
        cursor.close()
        cnx.close()     
        
        
class line():
    
    def update_web_db():
        linevalidation = pd.read_csv('line_validation.csv')
        cols = ['date', 'favorite', 'underdog', 'line', 'juice', 'confidence','juice_adj_conf', 'pred','covered']
        linevalidation = linevalidation[cols]
        
        linepreds = pd.read_csv('future_line_predictions.csv')
        cols = ['date', 'favorite', 'underdog', 'line', 'juice', 'confidence','juice_adj_conf','pred']
        linepreds = linepreds[cols]
        
        cnx = mysql.connector.connect(user='b7bd657ff1e565', password='1c64aeb2',
                                      host='us-cdbr-sl-dfw-01.cleardb.net',
                                      database='ibmx_e6e52fcfe7c7709')    
        cursor = cnx.cursor()
        cursor.execute('call ibmx_e6e52fcfe7c7709.reset_line();')
        for data in np.array(linevalidation):
            insert = []
            insert.append("('"+str(data[0])+"', '"+str(data[1])+"', '"+str(data[2])+"', "+str(data[3])+", "+str(data[4])+", "+str(data[5])+", "+str(data[6])+", '"+str(data[7])+"', '"+str(data[8])+"');")
            insertx = ','.join(insert)
            inlist = ['INSERT INTO line_validation VALUES', insertx]
            inlist = ' '.join(inlist)
            cursor.execute(inlist)
        cnx.commit()
        cursor.close()
        cnx.close()
         
        cnx = mysql.connector.connect(user='b7bd657ff1e565', password='1c64aeb2',
                                      host='us-cdbr-sl-dfw-01.cleardb.net',
                                      database='ibmx_e6e52fcfe7c7709')    
        cursor = cnx.cursor() 
        for data in np.array(linepreds):
            insert = []
            insert.append("('"+str(data[0])+"', '"+str(data[1])+"', '"+str(data[2])+"', "+str(data[3])+", "+str(data[4])+", "+str(data[5])+", "+str(data[6])+", '"+str(data[7])+"');")
            insertx = ','.join(insert)
            inlist = ['INSERT INTO line_prediction VALUES', insertx]
            inlist = ' '.join(inlist)
            cursor.execute(inlist)
        cnx.commit()
        cursor.close()
        cnx.close()
        
    def update_excel():
        process_raw_line('test')
        process_raw_line('new')
        traindata = pd.read_csv('train_processed_line.csv')
        testdata = pd.read_csv('validation_processed_line.csv')
        newdata = pd.read_csv('new_processed_line.csv')
        alldata = testdata.append(traindata)
        
        
        features = list(traindata)[5:]
        x_feat = features[:-2]
        train_x = traindata[x_feat]
        train_y = traindata['y']
        test_x = testdata[x_feat]
        test_y = testdata['y']
        test_juice = testdata['juice']
        all_x = alldata[x_feat]
        all_y = alldata['y']
        new_x = newdata[x_feat]
        new_juice = newdata['juice']
        
        model = tuned_line_models.tuned_ensemble()
        model.fit(train_x, train_y)
        
        cols = ['date', 'favorite', 'underdog', 'line', 'juice', 'confidence','juice_adj_conf', 'pred','covered']
        test = pd.DataFrame(columns = cols)
        test['date'] = np.array(testdata['date'])
        test['favorite'] = np.array(testdata['fav'])
        test['underdog'] = np.array(testdata['dog'])
        test['line'] = np.array(testdata['line'])
        winner = []
        for i, team in enumerate(np.array(testdata['y'])):
            if team == 0:
                winner.append(testdata['dog'][i])
            elif team == 1:
                winner.append(testdata['fav'][i])
            else:
                winner.append(team)
        
        test['covered'] = winner
        pred = []
        conf = []
        juiceconf = []
        for x,y,juice in zip(np.array(test_x), test_y, test_juice):
            odds = model.predict_proba(x.reshape(1,-1))
            if odds[0][0] > .5 - float(juice)/200:
                juiceconf.append(odds[0][0]+float(juice)/200)
            elif odds[0][1] > .5 + float(juice)/200:
                juiceconf.append(odds[0][1]-float(juice)/200)
                
                
        for i, (x, y) in enumerate(zip(np.array(test_x), np.array(test_y))):
            prediction = model.predict(x.reshape(1,-1))
            odds = model.predict_proba(x.reshape(1,-1))
            if odds[0][0] > odds[0][1]:
                conf.append(odds[0][0])
                pred.append(testdata['dog'][i])
            elif odds[0][0] < odds[0][1]:
                conf.append(odds[0][1])
                pred.append(testdata['fav'][i])
                
        test['juice_adj_conf'] = juiceconf   
        test['pred'] = pred
        test['confidence'] = conf
        test['juice']  = test_juice
        
        test.to_csv('line_validation.csv')
        
        model = tuned_line_models.tuned_ensemble()
        model.fit(all_x, all_y)
        cols = ['date', 'favorite', 'underdog', 'line', 'juice', 'confidence','juice_adj_conf','pred']
        new = pd.DataFrame(columns = cols)
        new['date'] = np.array(newdata['date'])
        new['favorite'] = np.array(newdata['fav'])
        new['underdog'] = np.array(newdata['dog'])
        new['line'] = np.array(newdata['line'])
        pred = []
        conf = []
        
        for i, x in enumerate(np.array(new_x)):
            prediction = model.predict(x.reshape(1,-1))
            odds = model.predict_proba(x.reshape(1,-1))
            if prediction == 0:
                pred.append(np.array(newdata['dog'])[i])
            elif prediction == 1:
                pred.append(np.array(newdata['fav'])[i])        
            if odds[0][0] > odds[0][1]:
                conf.append(odds[0][0])
            elif odds[0][0] < odds[0][1]:
                conf.append(odds[0][1])
        
        juiceconf = []
        for x,juice in zip(np.array(new_x), new_juice):
            odds = model.predict_proba(x.reshape(1,-1))
            if odds[0][0] > .5 - float(juice)/200:
                juiceconf.append(odds[0][0]+float(juice)/200)
            elif odds[0][1] > .5 + float(juice)/200:
                juiceconf.append(odds[0][1]-float(juice)/200)
                
              
        new['pred'] = pred
        new['confidence'] = conf
        new['juice']  = new_juice
        new['juice_adj_conf'] = juiceconf
        new.to_csv('future_line_predictions.csv')
        
    def graph():
        traindata = pd.read_csv('train_processed_line.csv')
        testdata = pd.read_csv('validation_processed_line.csv')
                
        features = list(traindata)[5:]
        x_feat = features[:-2]
        train_x = traindata[x_feat]
        train_y = traindata['y']
        test_x = testdata[x_feat]
        test_y = testdata['y']
        test_juice = testdata['juice']
        
        model = tuned_line_models.tuned_ensemble()
        model.fit(train_x, train_y)
        
        cols = ['date', 'favorite', 'underdog', 'line', 'juice', 'confidence','juice_adj_conf', 'pred','covered']
        test = pd.DataFrame(columns = cols)
        test['date'] = np.array(testdata['date'])
        test['favorite'] = np.array(testdata['fav'])
        test['underdog'] = np.array(testdata['dog'])
        test['line'] = np.array(testdata['line'])
        winner = []
        for i, team in enumerate(np.array(testdata['y'])):
            if team == 0:
                winner.append(testdata['dog'][i])
            elif team == 1:
                winner.append(testdata['fav'][i])
            else:
                winner.append(team)
        
        test['covered'] = winner
        pred = []
        conf = []
        juiceconf = []
        for x,y,juice in zip(np.array(test_x), test_y, test_juice):
            odds = model.predict_proba(x.reshape(1,-1))
            if odds[0][0] > .5 - float(juice)/200:
                juiceconf.append(odds[0][0]+float(juice)/200)
            elif odds[0][1] > .5 + float(juice)/200:
                juiceconf.append(odds[0][1]-float(juice)/200)
                
                
        for i, (x, y) in enumerate(zip(np.array(test_x), np.array(test_y))):
            prediction = model.predict(x.reshape(1,-1))
            odds = model.predict_proba(x.reshape(1,-1))
            if odds[0][0] > odds[0][1]:
                conf.append(odds[0][0])
                pred.append(testdata['dog'][i])
            elif odds[0][0] < odds[0][1]:
                conf.append(odds[0][1])
                pred.append(testdata['fav'][i])
                
        test['juice_adj_conf'] = juiceconf   
        test['pred'] = pred
        test['confidence'] = conf
        test['juice']  = test_juice

        plt.figure(figsize=(15, 8))
        plt.title("Net Funds Over Season With Different Juice Adjusted Confidence Thresholds For Making Bet",
                  fontsize=16)
        plt.xlabel("game number")
        plt.ylabel("funds")
        ax = plt.axes()
        ax.axhline(y=1000, color = 'k', linestyle = ':')
        for confidence_threshold in [.5,.51,.52,.53,.54,.55,.56,.57,.58,.59]:
            bank = 1000
            bankhistory = []
            for x,y,juice in zip(np.array(test_x), test_y, test_juice):
                pred = model.predict_proba(x.reshape(1,-1))
                if pred[0][0] > confidence_threshold - float(juice)/200:
                    prediction = 0
                    risk = 110 - juice
                    realwinner = y
                    if realwinner == prediction:
                        bank += 100
                    elif realwinner != prediction:
                        bank -= risk
                    bankhistory.append(bank)
                elif pred[0][1] > confidence_threshold + float(juice)/200:
                    prediction = 1
                    risk = 110 + juice
                    realwinner = y
                    if realwinner == prediction:
                        bank += 100
                    elif realwinner != prediction:
                        bank -= risk
                    bankhistory.append(bank)
                else:
                    bankhistory.append(bank)
            ax.plot(bankhistory, label=confidence_threshold)      
        plt.legend(loc="best")
        plt.grid('off')
        plt.show()
        
        
class moneyline():
        
    def update_web_db():            
        mlvalidation = pd.read_csv('ml_validation.csv')
        cols = ['date', 'favorite', 'underdog', 'favml', 'dogml', 'implied fav odds', 'implied dog odds', 'model fav odds', 'model dog odds', 'odds diff', 'prediction', 'winner']
        mlvalidation = mlvalidation[cols]
        
        mlpreds = pd.read_csv('future_ml_predictions.csv')
        cols = ['date', 'favorite', 'underdog', 'favml', 'dogml', 'implied fav odds', 'implied dog odds', 'model fav odds', 'model dog odds', 'odds diff', 'prediction']
        mlpreds = mlpreds[cols]
        
        cnx = mysql.connector.connect(user='b7bd657ff1e565', password='1c64aeb2',
                                      host='us-cdbr-sl-dfw-01.cleardb.net',
                                      database='ibmx_e6e52fcfe7c7709')    
        cursor = cnx.cursor() 
        cursor.execute('call ibmx_e6e52fcfe7c7709.reset_ml();')
        for data in np.array(mlvalidation):
            insert = []
            insert.append("('"+str(data[0])+"', '"+str(data[1])+"', '"+str(data[2])+"', "+str(data[3])+", "+str(data[4])+", "+str(data[5])+", "+str(data[6])+", "+str(data[7])+", "+str(data[8])+", "+str(data[9])+", '"+str(data[10])+"', '"+str(data[11])+"');")
            insertx = ','.join(insert)
            inlist = ['INSERT INTO ml_validation VALUES', insertx]
            inlist = ' '.join(inlist)
            cursor.execute(inlist)
        cnx.commit()
        cursor.close()
        cnx.close()
         
        cnx = mysql.connector.connect(user='b7bd657ff1e565', password='1c64aeb2',
                                      host='us-cdbr-sl-dfw-01.cleardb.net',
                                      database='ibmx_e6e52fcfe7c7709')    
        cursor = cnx.cursor() 
        for data in np.array(mlpreds):
            insert = []
            insert.append("('"+str(data[0])+"', '"+str(data[1])+"', '"+str(data[2])+"', "+str(data[3])+", "+str(data[4])+", "+str(data[5])+", "+str(data[6])+", "+str(data[7])+", "+str(data[8])+", "+str(data[9])+", '"+str(data[10])+"');")
            insertx = ','.join(insert)
            inlist = ['INSERT INTO ml_prediction VALUES', insertx]
            inlist = ' '.join(inlist)
            cursor.execute(inlist)
        cnx.commit()
        cursor.close()
        cnx.close()
        
    def update_excel():
        process_raw_ml('test')
        process_raw_ml('new')
        
        traindata = pd.read_csv('train_processed_ml.csv')
        testdata = pd.read_csv('validation_processed_ml.csv')
        newdata = pd.read_csv('new_processed_ml.csv')
        newdata = newdata.dropna(how = 'any')
        
        df_majority_train = traindata[traindata.y==1]
        df_minority_train = traindata[traindata.y==0]
        df_minority_upsampled_train = resample(df_minority_train, 
                                         replace=True,     # sample with replacement
                                         n_samples=len(df_majority_train.y),    # to match majority class
                                         random_state=123) # reproducible results
        traindata = pd.concat([df_majority_train, df_minority_upsampled_train])
        df_majority_test = testdata[testdata.y=='1']
        df_minority_test = testdata[testdata.y=='0']
        df_minority_upsampled_test = resample(df_minority_test, 
                                         replace=True,     # sample with replacement
                                         n_samples=len(df_majority_test.y),    # to match majority class
                                         random_state=123) # reproducible results
        testdata_upsampled = pd.concat([df_majority_test, df_minority_upsampled_test])
        alldata = traindata.append(testdata_upsampled)
        
        features = list(traindata)[5:]
        x_feat = features[:-3]
        train_x = traindata[x_feat]
        train_y = traindata['y']
        test_x = testdata[x_feat]
        test_y = testdata['y']
        all_x = alldata[x_feat]
        all_y = alldata['y'].astype(int)
        new_x = newdata[x_feat]
        
        model = tuned_ml_models.tuned_ensemble()
        model.fit(train_x, train_y)
        cols = ['date', 'favorite', 'underdog', 'favml', 'dogml', 'implied fav odds', 'implied dog odds', 'model fav odds', 'model dog odds', 'odds diff', 'prediction', 'winner']
        test = pd.DataFrame(columns = cols)
        test['date'] = np.array(testdata['date'])
        test['favorite'] = np.array(testdata['fav'])
        test['underdog'] = np.array(testdata['dog'])
        winner = []
        lineodds = []
        for i, team in enumerate(np.array(testdata['y'])):
            if team == '0':
                winner.append(testdata['dog'][i])
            elif team == '1':
                winner.append(testdata['fav'][i])
            else:
                winner.append(team)
        test['winner'] = winner
        test['favml'] = np.array(testdata['favml'])
        test['dogml'] = np.array(testdata['dogml'])
        pred = []
        implied_dog = []
        implied_fav = []
        model_dog = []
        model_fav = []
        modelpred = []
        modeldiff = []
        
        for fav, dog in zip(np.array(testdata['favml']), np.array(testdata['dogml'])):
            if dog > 0:
                implied_dog.append((100/(dog+100)))
            elif dog < 0:
                implied_dog.append((-1*(dog))/((-1*(dog))+100))
            if fav < 0:
                implied_fav.append((-1*(fav))/((-1*(fav))+100))
            elif fav > 0:
                implied_fav.append((100/(dog+100)))
        for i, (x, y) in enumerate(zip(np.array(test_x), np.array(test_y))):
            prediction = model.predict(x.reshape(1,-1))
            odds = model.predict_proba(x.reshape(1,-1))
            pred.append(prediction[0])
            model_dog.append(odds[0][0])
            model_fav.append(odds[0][1])
            if odds[0][0] > implied_dog[i]:
                modeldiff.append(odds[0][0] - implied_dog[i])
                modelpred.append(np.array(testdata['dog'])[i])
                lineodds.append(testdata['dogml'][i])
            elif odds[0][1] > implied_fav[i]:
                modeldiff.append(odds[0][1]- implied_fav[i])
                modelpred.append(np.array(testdata['fav'])[i])
                lineodds.append(testdata['favml'][i])
            else:
                modeldiff.append(0)
                modelpred.append('No Odds Advantage, No Prediction Made')
                lineodds.append(0)
        
        
        test['model fav odds'] = model_fav
        test['model dog odds'] = model_dog  
        test['implied fav odds'] = implied_fav
        test['implied dog odds'] = implied_dog
        test['prediction'] = modelpred
        test['odds diff'] = modeldiff
        
        
        test.to_csv('ml_validation.csv')
        
        model = tuned_ml_models.tuned_ensemble()
        model.fit(all_x, all_y)
        cols = ['date', 'favorite', 'underdog', 'favml', 'dogml', 'implied fav odds', 'implied dog odds', 'model fav odds', 'model dog odds', 'odds diff', 'prediction']
        new = pd.DataFrame(columns = cols)
        new['date'] = np.array(newdata['date'])
        new['favorite'] = np.array(newdata['fav'])
        new['underdog'] = np.array(newdata['dog'])
        lineodds = []
        
        new['favml'] = np.array(newdata['favml'])
        new['dogml'] = np.array(newdata['dogml'])
        pred = []
        implied_dog = []
        implied_fav = []
        model_dog = []
        model_fav = []
        modelpred = []
        modeldiff = []
        
        for fav, dog in zip(np.array(newdata['favml']), np.array(newdata['dogml'])):
            if dog > 0:
                implied_dog.append((100/(dog+100)))
            elif dog < 0:
                implied_dog.append((-1*(dog))/((-1*(dog))+100))
            if fav < 0:
                implied_fav.append((-1*(fav))/((-1*(fav))+100))
            elif fav > 0:
                implied_fav.append((100/(dog+100)))
        for i, (x) in enumerate(np.array(new_x)):
            prediction = model.predict(x.reshape(1,-1))
            odds = model.predict_proba(x.reshape(1,-1))
            pred.append(prediction[0])
            model_dog.append(odds[0][0])
            model_fav.append(odds[0][1])
            if odds[0][0] > implied_dog[i]:
                modeldiff.append(odds[0][0] - implied_dog[i])
                modelpred.append(np.array(testdata['dog'])[i])
                lineodds.append(testdata['dogml'][i])
            elif odds[0][1] > implied_fav[i]:
                modeldiff.append(odds[0][1]- implied_fav[i])
                modelpred.append(np.array(testdata['fav'])[i])
                lineodds.append(testdata['favml'][i])
            else:
                modeldiff.append(0)
                modelpred.append('No Odds Advantage, No Prediction Made')
                lineodds.append(0)
        new['model fav odds'] = model_fav
        new['model dog odds'] = model_dog  
        new['implied fav odds'] = implied_fav
        new['implied dog odds'] = implied_dog
        new['prediction'] = modelpred
        new['odds diff'] = modeldiff
        
        new.to_csv('future_ml_predictions.csv')
        
    def graph():           
        traindata = pd.read_csv('train_processed_ml.csv')
        testdata = pd.read_csv('validation_processed_ml.csv')
        
        df_majority_train = traindata[traindata.y==1]
        df_minority_train = traindata[traindata.y==0]
        df_minority_upsampled_train = resample(df_minority_train, 
                                         replace=True,     # sample with replacement
                                         n_samples=len(df_majority_train.y),    # to match majority class
                                         random_state=123) # reproducible results
        traindata = pd.concat([df_majority_train, df_minority_upsampled_train])            
        
        features = list(traindata)[5:]
        x_feat = features[:-3]
        train_x = traindata[x_feat]
        train_y = traindata['y']
        test_x = testdata[x_feat]
        test_y = testdata['y']
        
        model = tuned_ml_models.tuned_ensemble()
        model.fit(train_x, train_y)
        
        cols = ['date', 'favorite', 'underdog', 'favml', 'dogml', 'implied fav odds', 'implied dog odds', 'model fav odds', 'model dog odds', 'odds diff', 'prediction', 'winner']
        test = pd.DataFrame(columns = cols)
        test['date'] = np.array(testdata['date'])
        test['favorite'] = np.array(testdata['fav'])
        test['underdog'] = np.array(testdata['dog'])
        winner = []
        lineodds = []
        for i, team in enumerate(np.array(testdata['y'])):
            if team == '0':
                winner.append(testdata['dog'][i])
            elif team == '1':
                winner.append(testdata['fav'][i])
            else:
                winner.append(team)
        
        test['winner'] = winner
        test['favml'] = np.array(testdata['favml'])
        test['dogml'] = np.array(testdata['dogml'])
        pred = []
        implied_dog = []
        implied_fav = []
        model_dog = []
        model_fav = []
        modelpred = []
        modeldiff = []
        
        for fav, dog in zip(np.array(testdata['favml']), np.array(testdata['dogml'])):
            if dog > 0:
                implied_dog.append((100/(dog+100)))
            elif dog < 0:
                implied_dog.append((-1*(dog))/((-1*(dog))+100))
            if fav < 0:
                implied_fav.append((-1*(fav))/((-1*(fav))+100))
            elif fav > 0:
                implied_fav.append((100/(dog+100)))
        for i, (x, y) in enumerate(zip(np.array(test_x), np.array(test_y))):
            prediction = model.predict(x.reshape(1,-1))
            odds = model.predict_proba(x.reshape(1,-1))
            pred.append(prediction[0])
            model_dog.append(odds[0][0])
            model_fav.append(odds[0][1])
            if odds[0][0] > implied_dog[i]:
                modeldiff.append(odds[0][0] - implied_dog[i])
                modelpred.append(np.array(testdata['dog'])[i])
                lineodds.append(testdata['dogml'][i])
            elif odds[0][1] > implied_fav[i]:
                modeldiff.append(odds[0][1]- implied_fav[i])
                modelpred.append(np.array(testdata['fav'])[i])
                lineodds.append(testdata['favml'][i])
            else:
                modeldiff.append(0)
                modelpred.append('No Odds Advantage, No Prediction Made')
                lineodds.append(0)
        
        
        test['model fav odds'] = model_fav
        test['model dog odds'] = model_dog  
        test['implied fav odds'] = implied_fav
        test['implied dog odds'] = implied_dog
        test['prediction'] = modelpred
        test['odds diff'] = modeldiff

        plt.figure(figsize=(15, 8))
        plt.title("Net Funds Over Season With Different Implied Odds vs Confidence Difference Thresholds For Making Bet",
                  fontsize=16)
        plt.xlabel("game number")
        plt.ylabel("funds")
        ax = plt.axes()
        ax.axhline(y = 1000, color = 'k', linestyle = ':')
        for confidence_threshold in [0,.02,.04,.06,.08,.1]:
            bank = 1000
            bankhistory = []
            for x,y,pred,odds,vegas in zip(np.array(test_x), np.array(test['winner']), modelpred, modeldiff, lineodds):
                if odds > confidence_threshold:
                    if vegas > 0:
                        payout = vegas
                    elif vegas < 0:
                        payout = (100/(-1*vegas))*100
                    if pred == y:
                        bank+=payout
                    elif pred != y:
                        bank -= 100
                bankhistory.append(bank)
            ax.plot(bankhistory, label=confidence_threshold)    
        plt.legend(loc="best")
        plt.grid('off')
        plt.show()  