#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 --------------------------------------------------------------------------------
 SPADE - Support for Provenance Auditing in Distributed Environments.
 Copyright (C) 2015 SRI International
 This program is free software: you can redistribute it and/or
 modify it under the terms of the GNU General Public License as
 published by the Free Software Foundation, either version 3 of the
 License, or (at your option) any later version.
 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 General Public License for more details.
 You should have received a copy of the GNU General Public License
 along with this program. If not, see <http://www.gnu.org/licenses/>.
 --------------------------------------------------------------------------------

"""
import os
from web3 import Web3
import csv
import pandas as pd

#sm_file = 'Smart_Contract_Addresses.list'
sm_file = 'sm_add_nponzi.csv'
# path = '/Users/charles/charles/university/Master Project/go-ethereum/analysis_tool_python/SmartPonziDetection/'
path = '../dataset/'
# database_bcode = path + 'dataset/'
database_bcode = '../dataset/'

web3 = Web3(Web3.HTTPProvider('https://mainnet.infura.io/TO1X2JTG8k9PiaYd0iQr'))


# with open(path + sm_file, 'rt') as f:
#     truc = csv.reader(f)
#     add = list(truc)
#
#
# addresses = [pk[:42] for pklist in add for pk in pklist]
with open(database_bcode + 'non_ponzi_collection.csv') as f:
# with open(database_bcode + 'ponzi_collection.csv') as f:
    truc = csv.reader(f)
    csv_file = list(truc)


# addresses = [line[0].split('(')[0].strip() for line in csv_file if line[0] != 'addr']

ponzi_df = pd.read_csv('../dataset/ponzi_collection.csv')
addresses = ponzi_df['Address'].tolist()

i = 0
for ad in addresses:
    # code = repr(web3.eth.getCode(web3.toChecksumAddress(ad)))[12:-2]
    # print(ad + ', progress: ' + str(round(i / len(addresses) * 100, 2)) + '%')
    # while True:
    #     try:
    #         if code:
    #             print('code: ', code)
    #             i += 1
    #             with open(database_bcode + 'non_ponzi_bcode/' + ad + '.json', 'w') as f:
                #     print('writing...')
                # with open(database_bcode + 'ponzi_bcode/' + ad + '.json', 'w') as f:
                #     f.write(code)
                # f.close()
                # print('file closed')
            # else:
            #     print('code == false', code, ad)
            # break
        # except Exception as e:
        #     print('Error: ')
        #     print(e)
    #Disasemble
    print(ad)
    # os.system('cat /Users/e31989/Documents/sm_database/bytecode/' + ad +'.json | evmdis > /Users/e31989/Documents/features/' + ad + '.json' )
    # print('evmdis disasembling...')
    i += 1
    try:
        os.system(
    #         'cat /Users/Jinyue/Documents/ML-jsong32/final_project/dataset/non_ponzi_bcode/' + ad + '.json | evmdis > /Users/Jinyue/Documents/ML-jsong32/final_project/dataset/non_ponzi_features/' + ad + '.json')
        'cat ../ponzi_bcode/' + ad + '.json | evmdis > ../ponzi_features/' + ad + '.json')
    # print('evmdis work done\n')
    except Exception as e:
        print('Error: ')
        print(e)
# for /r %i in (*.json); do cat "%i" | evmdis > "/Users/e31989/Documents/features/$~ni.json"; done
# for %i in (*.json); do cat "%i" | evmdis > "/Users/e31989/Documents/features/$~ni.json"; done