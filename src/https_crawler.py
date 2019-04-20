from requests_html import HTMLSession
import json
# session = HTMLSession()
# url = 'https://etherscan.io/api?module=opcode&action=getopcode&address=0x109c4f2ccc82c4d77bde15f306707320294aea3f'
# r = session.get(url)
# print(r.html.text)

import csv

sm_file = 'sm_add_nponzi.csv'
# path = '/Users/yifanzhou/Desktop/Study/MachineLearning/finalproject/CS690Ponzi/'
database_opcode = '../dataset/'
url = 'https://etherscan.io/api?module=opcode&action=getopcode&address={0}'

#
# with open(database_opcode + 'non_ponzi_collection.csv') as f:
#     truc = csv.reader(f)
#     csv_file = list(truc)
#
# addresses = [line[0].split('(')[0].strip() for line in csv_file if line[0] != 'addr']
#
# i = 0
# for ad in addresses:
#     session = HTMLSession()
#     code = session.get(url.format(ad)).html.text
#     print('non ponzi')
#     print(ad + ', progress: ' + str(round(i / len(addresses) * 100, 2)) + '%')
#     if code:
#         i += 1
#         # print(str(i) + ": " + ad)
#         with open(database_opcode + 'non_ponzi_official_opcode/' + ad + '.json', 'w') as f:
#             f.write(code)
#         f.close()
#     else:
#         print(ad)
#
# with open(database_opcode + 'ponzi_collection.csv') as f:
#     truc = csv.reader(f)
#     csv_file = list(truc)
#
# addresses = [line[2] for line in csv_file if line[0] != 'Id']
#
# # print('ponzi addrs: ', addresses)
#
# i = 0
# for ad in addresses:
#     session = HTMLSession()
#     code = session.get(url.format(ad)).html.text
#     print('ponzi old')
#     print(ad + ', progress: ' + str(round(i / len(addresses) * 100, 2)) + '%')
#     if code:
#         i += 1
#         with open(database_opcode + 'ponzi_official_opcode/' + ad + '.json', 'w') as f:
#             f.write(code)
#         f.close()
#     else:
#         print(ad)

with open(database_opcode + 'ponzi_supplement_2019_Apr_9.csv') as f:
    truc = csv.reader(f)
    csv_file = list(truc)

addresses = [line[2] for line in csv_file if line[0] != 'Id']

i = 0
for ad in addresses:
    session = HTMLSession()
    code = session.get(url.format(ad)).html.text
    print('ponzi new')
    print(ad + ', progress: ' + str(round(i / len(addresses) * 100, 2)) + '%')
    if code:
        i += 1
        with open(database_opcode + 'ponzi_official_opcode/' + ad + '.json', 'w') as f:
            f.write(code)
        f.close()
    else:
        print(ad)