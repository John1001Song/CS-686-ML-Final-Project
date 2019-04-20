from bs4 import BeautifulSoup
from lxml import html
import requests
import pandas as pd

class BytecodeCrawler:
    def __init__(self):
        self.ponzi_addresses = []
        self.non_ponzi_addresses = []
        self.paths = dict()
        # 0: ponzi; 1: non ponzi
        self.un_downloaded_address = [[], []]

    def start(self):
        self.define_paths()
        self.get_addresses()
        self.get_all_bcode()

    def define_paths(self):
        self.paths['ponzi_collection'] = '../dataset/ponzi_collection.csv'
        self.paths['ponzi_collection_2019'] = '../dataset/ponzi_supplement_2019_Apr_9.csv'
        self.paths['non_ponzi_collection'] = '../dataset/non_ponzi_collection.csv'
        self.paths['ponzi_bcode_output'] = '../dataset/ponzi/bcode/'
        self.paths['non_ponzi_bcode_output'] = '../dataset/non_ponzi/bcode/'

    def get_addresses(self):
        self.ponzi_addresses = pd.read_csv(self.paths['ponzi_collection'])['Address'].tolist()
        self.ponzi_addresses += pd.read_csv(self.paths['ponzi_collection_2019'])['Address'].tolist()
        np_raw_addresses = pd.read_csv(self.paths['non_ponzi_collection'])['addr']
        self.non_ponzi_addresses = [addr.split(' ')[0] for addr in np_raw_addresses]

    def get_all_bcode(self):
        bcode_paths = [self.ponzi_addresses, self.non_ponzi_addresses]
        for i in range(2):
            count = 0
            for ad in bcode_paths[i]:
                count += 1
                self.get_one_bcode(ad, i)
                print(ad + ', progress: ' + str(round(count / len(bcode_paths[i]) * 100, 2)) + '%')

    def get_one_bcode(self, address, index):
        url = 'https://etherscan.io/address/{0}#code'
        if index == 0:
            output_path = self.paths['ponzi_bcode_output']
        else:
            output_path = self.paths['non_ponzi_bcode_output']

        resp = requests.get(url.format(address))

        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, 'html.parser')
            divs = soup.findAll("div", {"id": "verifiedbytecode2"})
            for d in divs:
                # print(d)
                # print(type(d))
                # print(d.string)
                if d.string == None or d.string == '':
                    self.un_downloaded_address[index].append(address)
                    print('Empty bcode')
                else:
                    with open(output_path + address + '.json', 'w') as f:
                        f.write(d.string)
                    f.close()
        else:
            print("Error")
            self.un_downloaded_address[index].append(address)

if __name__ == '__main__':
    bc = BytecodeCrawler()
    bc.start()
    print(bc.un_downloaded_address)