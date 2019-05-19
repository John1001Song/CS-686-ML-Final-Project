import pandas as pd
import os
import json
import numpy as np
from ast import literal_eval
from scipy import stats


OPCODES = ['STOP', 'ADD', 'MUL', 'SUB', 'DIV', 'SDIV', 'MOD', 'SMOD', 'ADDMOD', 'MULMOD', 'EXP', 'SIGNEXTEND',

             'SUICIDE', 'DELEGATE_CALL', 'CREATE2',

             'LT', 'GT', 'SLT', 'SGT', 'EQ', 'ISZERO', 'AND', 'OR', 'XOR', 'NOT', 'BYTE', 'SHA3', 'ADDRESS', 'BALANCE',
             'ORIGIN', 'CALLER', 'CALLVALUE', 'CALLDATALOAD', 'CALLDATASIZE', 'CALLDATACOPY', 'CODESIZE', 'CODECOPY',
             'GASPRICE', 'EXTCODESIZE', 'EXTCODECOPY',

             'RETURNDATASIZE', 'RETURNDATACOPY',

             'BLOCKHASH', 'COINBASE', 'TIMESTAMP', 'NUMBER', 'DIFFICULTY', 'GASLIMIT', 'POP', 'MLOAD', 'MSTORE',
             'MSTORE8', 'SLOAD', 'SSTORE', 'JUMP', 'JUMPI', 'PC', 'MSIZE', 'GAS', 'JUMPDEST',
             'PUSH1', 'PUSH2', 'PUSH3', 'PUSH4', 'PUSH5', 'PUSH6', 'PUSH7', 'PUSH8', 'PUSH9', 'PUSH10', 'PUSH11',
             'PUSH12', 'PUSH13', 'PUSH14', 'PUSH15', 'PUSH16', 'PUSH17', 'PUSH18', 'PUSH19', 'PUSH20', 'PUSH21',
             'PUSH22', 'PUSH23', 'PUSH24', 'PUSH25', 'PUSH26', 'PUSH27', 'PUSH28', 'PUSH29','PUSH30', 'PUSH31', 'PUSH32',
             'DUP1', 'DUP2', 'DUP3', 'DUP4', 'DUP5', 'DUP6', 'DUP7', 'DUP8', 'DUP9',
             'DUP10', 'DUP11', 'DUP12', 'DUP13', 'DUP14', 'DUP15', 'DUP16',
             'SWAP1', 'SWAP2', 'SWAP3', 'SWAP4', 'SWAP5', 'SWAP6', 'SWAP7', 'SWAP8', 'SWAP9', 'SWAP10', 'SWAP11',
             'SWAP12', 'SWAP13', 'SWAP14', 'SWAP15', 'SWAP16',
             'LOG0', 'LOG1', 'LOG2', 'LOG3', 'LOG4',

             'PUSH', 'DUP', 'SWAP',

             'CREATE', 'CALL', 'CALLCODE', 'RETURN', 'DELEGATECALL', 'STATICCALL', 'REVERT', 'SELFDESTRUCT',

            '29', '0d', 'ec', 'd9', 'a9', '46', 'b3', '2a', 'd2', 'c9', '22', '21', '1c', 'ea', 'c7', 'ee', 'd5', 'e5',
            'ad', 'ac', '25', 'ca', 'be', 'e8', 'aa', 'b1', '1e', '49', 'e6', 'eb', 'a7', 'cf', '2e', '0c', '5d', 'ef',
            'c8', '24', 'fe', 'e3', '4d', '2d', '3f', 'c0', 'd1', '23', 'b6', 'd7', 'cd', 'e1', 'e4', 'd6', 'af', 'f8',
            'e0', 'e9', 'da', 'b2', '4e', '1d', 'cb', 'dc', 'df', 'c6', 'fb', 'b9', 'f9', 'd4', '28', 'b4', 'c5', 'f7',
            '3e', 'a8', 'a6', '2f', 'b7', '47', 'fc', 'bf', '5c', '2c', '1b', '3d', 'c4', 'ce', '4b', 'bb', 'd8', 'bc',
            '4c', 'f6', 'bd', 'db', '2b', '26', '48', '4f', 'c3', 'de', 'd3', 'e2', '4a', 'ab', 'd0', 'b5', 'dd', '0f',
            '5e', 'c1', 'ed', '27', 'a5', 'b8', 'c2', 'e7', '0e', '5f', 'b0', 'ae', '1f', 'ba'
            ]

OPCODES_without_REVERT = ['STOP', 'ADD', 'MUL', 'SUB', 'DIV', 'SDIV', 'MOD', 'SMOD', 'ADDMOD', 'MULMOD', 'EXP', 'SIGNEXTEND',

             'SUICIDE', 'DELEGATE_CALL', 'CREATE2',

             'LT', 'GT', 'SLT', 'SGT', 'EQ', 'ISZERO', 'AND', 'OR', 'XOR', 'NOT', 'BYTE', 'SHA3', 'ADDRESS', 'BALANCE',
             'ORIGIN', 'CALLER', 'CALLVALUE', 'CALLDATALOAD', 'CALLDATASIZE', 'CALLDATACOPY', 'CODESIZE', 'CODECOPY',
             'GASPRICE', 'EXTCODESIZE', 'EXTCODECOPY',

             'RETURNDATASIZE', 'RETURNDATACOPY',

             'BLOCKHASH', 'COINBASE', 'TIMESTAMP', 'NUMBER', 'DIFFICULTY', 'GASLIMIT', 'POP', 'MLOAD', 'MSTORE',
             'MSTORE8', 'SLOAD', 'SSTORE', 'JUMP', 'JUMPI', 'PC', 'MSIZE', 'GAS', 'JUMPDEST',
             'PUSH1', 'PUSH2', 'PUSH3', 'PUSH4', 'PUSH5', 'PUSH6', 'PUSH7', 'PUSH8', 'PUSH9', 'PUSH10', 'PUSH11',
             'PUSH12', 'PUSH13', 'PUSH14', 'PUSH15', 'PUSH16', 'PUSH17', 'PUSH18', 'PUSH19', 'PUSH20', 'PUSH21',
             'PUSH22', 'PUSH23', 'PUSH24', 'PUSH25', 'PUSH26', 'PUSH27', 'PUSH28', 'PUSH29','PUSH30', 'PUSH31', 'PUSH32',
             'DUP1', 'DUP2', 'DUP3', 'DUP4', 'DUP5', 'DUP6', 'DUP7', 'DUP8', 'DUP9',
             'DUP10', 'DUP11', 'DUP12', 'DUP13', 'DUP14', 'DUP15', 'DUP16',
             'SWAP1', 'SWAP2', 'SWAP3', 'SWAP4', 'SWAP5', 'SWAP6', 'SWAP7', 'SWAP8', 'SWAP9', 'SWAP10', 'SWAP11',
             'SWAP12', 'SWAP13', 'SWAP14', 'SWAP15', 'SWAP16',
             'LOG0', 'LOG1', 'LOG2', 'LOG3', 'LOG4',

             'PUSH', 'DUP', 'SWAP',

             'CREATE', 'CALL', 'CALLCODE', 'RETURN', 'DELEGATECALL', 'STATICCALL', 'SELFDESTRUCT',

            '29', '0d', 'ec', 'd9', 'a9', '46', 'b3', '2a', 'd2', 'c9', '22', '21', '1c', 'ea', 'c7', 'ee', 'd5', 'e5',
            'ad', 'ac', '25', 'ca', 'be', 'e8', 'aa', 'b1', '1e', '49', 'e6', 'eb', 'a7', 'cf', '2e', '0c', '5d', 'ef',
            'c8', '24', 'fe', 'e3', '4d', '2d', '3f', 'c0', 'd1', '23', 'b6', 'd7', 'cd', 'e1', 'e4', 'd6', 'af', 'f8',
            'e0', 'e9', 'da', 'b2', '4e', '1d', 'cb', 'dc', 'df', 'c6', 'fb', 'b9', 'f9', 'd4', '28', 'b4', 'c5', 'f7',
            '3e', 'a8', 'a6', '2f', 'b7', '47', 'fc', 'bf', '5c', '2c', '1b', '3d', 'c4', 'ce', '4b', 'bb', 'd8', 'bc',
            '4c', 'f6', 'bd', 'db', '2b', '26', '48', '4f', 'c3', 'de', 'd3', 'e2', '4a', 'ab', 'd0', 'b5', 'dd', '0f',
            '5e', 'c1', 'ed', '27', 'a5', 'b8', 'c2', 'e7', '0e', '5f', 'b0', 'ae', '1f', 'ba'
            ]

class PreprocessData:
    '''
    Convert all contracts in opcodes to integers to represent.
    '''
    def __init__(self):
        self.opcodes = OPCODES
        self.opcodes_without_revert = OPCODES_without_REVERT
        self.paths = dict()
        # self.op_equal_size = [[], []] # 0: ponzi, 1:non_ponzi
        self.op_equal_size = [] # [{'address':0x123asd, 'sequence':[1,2,3,...,0], 'label':0}, {}, ..., {}]
        self.longest_op_list = 0

    def define_path(self):
        self.paths['db'] = '../dataset/'
        # input paths
        self.paths['ponzi_op'] = self.paths['db'] + 'ponzi_official_opcode/'
        self.paths['non_ponzi_op'] = self.paths['db'] + 'non_ponzi_official_opcode/'
        # output paths
        self.paths['ponzi_op_int'] = self.paths['db'] + 'ponzi_official_opcode_int/'
        self.paths['non_ponzi_op_int'] = self.paths['db'] + 'non_ponzi_official_opcode_int/'

        # output paths without REVERT
        self.paths['ponzi_op_int_wo_revert'] = self.paths['db'] + 'ponzi_official_opcode_int_wo_revert/'
        self.paths['non_ponzi_op_int_wo_revert'] = self.paths['db'] + 'non_ponzi_official_opcode_int_wo_revert/'

        # output paths for equal size
        self.paths['op_int_equal_dataframe'] = self.paths['db'] + 'op_int_equal_dataframe/'

    def start(self):
        self.define_path()
        # self.check_all_opcodes()
        # self.get_opcode_int()
        self.get_opcode_int_without_revert()
        self.add_0_when_no_data()
        # self.load_all_op_int()

    def check_all_opcodes(self):
        '''Go through all json files and check un-doc opcodes'''
        extra_opcode = set()
        opcode_paths = [self.paths['ponzi_op'], self.paths['non_ponzi_op']]

        for i in range(2):
            count = 0
            for filename in os.listdir(opcode_paths[i]):
                if not filename.endswith('.json'):
                    continue

                print(filename + ', progress: ' + str(round(count / len(os.listdir(opcode_paths[i])) * 100, 2)) + '%')

                with open(opcode_paths[i] + filename) as f_opcode:
                    while True:
                        line = f_opcode.readline()
                        # print(line)
                        if not line:
                            break
                        if line.__contains__('(Unknown Opcode)'):
                            # '29'(Unknown Opcode)
                            unknown_op = line[1:3]
                            if not unknown_op in self.opcodes:
                                extra_opcode.add(unknown_op)
                        elif line.__contains__(' '):
                            temp_op = line.split(' ')[0]
                            if temp_op not in self.opcodes:
                                extra_opcode.add(temp_op)
                        else:
                            temp_op = line.split('\n')[0]
                            if temp_op not in self.opcodes:
                                extra_opcode.add(temp_op)
                count += 1
        print('extra opcode: ', extra_opcode)

    def get_opcode_int_without_revert(self):
        opcode_paths = [self.paths['ponzi_op'],
                        self.paths['non_ponzi_op']]
        output_path = [self.paths['ponzi_op_int_wo_revert'], self.paths['non_ponzi_op_int_wo_revert']]

        for i in range(2):
            count = 0
            for filename in os.listdir(opcode_paths[i]):
                if not filename.endswith('.json'):
                    continue
                print(filename + ', progress: ' + str(round(count / len(os.listdir(opcode_paths[i])) * 100, 2)) + '%')
                self.get_one_opcode_int_without_revert(opcode_paths[i], filename, output_path[i])
                count += 1

    def get_one_opcode_int_without_revert(self, data_path, hex_addr, output_path):
        op_list = []
        with open(data_path + hex_addr) as f_opcode:
            while True:
                line = f_opcode.readline()
                # print(line)
                if not line:
                    break
                if line.__contains__('(Unknown Opcode)'):
                    op_list.append(line[1:3])
                    # print(line[1:3])
                elif line.__contains__('REVERT'):
                    continue
                elif line.__contains__(' '):
                    op_list.append(line.split(' ')[0])
                else:
                    op_list.append(line.split('\n')[0])
        # print('op int list: ', op_int_list)
        # print(len(op_int_list))
        op_int_list = []
        for op in op_list:
            op_int_list.append(self.opcodes_without_revert.index(op)+1)

        with open(output_path+hex_addr.split('.json')[0]+'.txt', 'w') as f_op_int:
            for item in op_int_list:
                f_op_int.write("%s\n" % item)

    def get_opcode_int(self):
        opcode_paths = [self.paths['ponzi_op'],
                        self.paths['non_ponzi_op']]
        output_path = [self.paths['ponzi_op_int'], self.paths['non_ponzi_op_int']]

        for i in range(2):
            count = 0
            for filename in os.listdir(opcode_paths[i]):
                if not filename.endswith('.json'):
                    continue
                print(filename + ', progress: ' + str(round(count / len(os.listdir(opcode_paths[i])) * 100, 2)) + '%')
                self.get_one_opcode_int(opcode_paths[i], filename, output_path[i])
                count += 1

    def get_one_opcode_int(self, data_path, hex_addr, output_path):
        op_list = []
        with open(data_path + hex_addr) as f_opcode:
            while True:
                line = f_opcode.readline()
                # print(line)
                if not line:
                    break
                if line.__contains__('(Unknown Opcode)'):
                    # print('!!!!!!!!!', line)
                    op_list.append(line[1:3])
                    # print(line[1:3])
                elif line.__contains__(' '):
                    op_list.append(line.split(' ')[0])
                else:
                    op_list.append(line.split('\n')[0])
        # print('op int list: ', op_int_list)
        # print(len(op_int_list))
        op_int_list = []
        for op in op_list:
            op_int_list.append(self.opcodes.index(op) + 1)

        with open(output_path+hex_addr.split('.json')[0]+'.txt', 'w') as f_op_int:
            for item in op_int_list:
                f_op_int.write("%s\n" % item)

    def load_one_opcode_int(self, data_path, hex_addr):
        op_list = []
        with open(data_path + hex_addr) as f_opcode:
            while True:
                line = f_opcode.readline()
                if not line:
                    break
                else:
                    op_list.append(int(line.split('\n')[0]))
        f_opcode.close()
        return op_list, len(op_list)

    def add_0_when_no_data(self):
        '''
        This func makes all contracts have a same length in their op lists
        For example:
        [1, 2, 3, 4, 0, 0, 0]
        [4, 3, 2, 1, 1, 1, 1]
        ...
        '''
        # input
        opcode_int_paths = [self.paths['ponzi_op_int_wo_revert'], self.paths['non_ponzi_op_int_wo_revert']]

        # load all ponzi and non ponzi into lists
        for i in range(2):
            for filename in os.listdir(opcode_int_paths[i]):
                if not filename.endswith('.txt'):
                    continue
                # print(filename)
                cur_op_list, cur_op_list_size = self.load_one_opcode_int(opcode_int_paths[i], filename)
                # print(cur_op_list)
                # print(cur_op_list_size)
                self.op_equal_size.append({'address': filename.split('.txt')[0], 'sequence': cur_op_list, 'label': i})
                if cur_op_list_size > self.longest_op_list:
                    self.longest_op_list = cur_op_list_size

        # make all lists have the same length
        for op_list_set in self.op_equal_size:
            cur_length = len(op_list_set['sequence'])
            op_list_set['sequence'] += [0] * (self.longest_op_list - cur_length)
            # op_list_set['sequence'] = np.asarray(op_list_set['sequence'])

        # build a dataframe
        df = pd.DataFrame(self.op_equal_size)
        # df.to_numpy()
        # print(df.iloc[0, :])
        # print(df.iloc[6716, :])

        # for index, row in df.iterrows():
        #     temp_list = []
        #     for op_int in row['sequence']:
        #         temp_list.append(int(op_int))
        #     row['sequence'] = temp_list

        # dump the dataframe
        columns = ['address', 'label']
        for i in range(self.longest_op_list):
            columns.append(f'op_seq_{i}')
        with open(self.paths['op_int_equal_dataframe'] + 'op_int_equal_new.csv', 'w') as f_out:
            f_out.write('address')
            columns.remove('address')
            for c_name in columns:
                f_out.write(',' + c_name)
            f_out.write('\n')
            f_out.close()
        with open(self.paths['op_int_equal_dataframe'] + 'op_int_equal_new.csv', 'a') as f_out:
            for index, row in df.iterrows():
                f_out.write(row['address'] + ',' + str(row['label']))
                for op_int in row['sequence']:
                    f_out.write(',' + str(op_int))
                f_out.write('\n')


        # df.to_csv(self.paths['op_int_equal_dataframe']+'op_int_equal_pd.csv', index=False)
        # use json dump, then use json load; otherwise, df cannot read it
        # with open(self.paths['op_int_equal_dataframe']+'op_int_equal.csv', 'w') as f_out:
        #     json.dump(self.op_equal_size, f_out)

    def load_all_op_int(self):
        # input
        opcode_int_paths = [self.paths['ponzi_op_int_wo_revert'], self.paths['non_ponzi_op_int_wo_revert']]

        sizes = []

        # load all ponzi and non ponzi into lists
        for i in range(2):
            for filename in os.listdir(opcode_int_paths[i]):
                if not filename.endswith('.txt'):
                    continue
                # print(filename)
                cur_op_list, cur_op_list_size = self.load_one_opcode_int(opcode_int_paths[i], filename)
                # print(cur_op_list)
                # print(cur_op_list_size)
                self.op_equal_size.append({'address': filename.split('.txt')[0], 'sequence': cur_op_list, 'label': i})
                sizes.append(cur_op_list_size)
                if cur_op_list_size > self.longest_op_list:
                    self.longest_op_list = cur_op_list_size

        print('mean of all sizes: ', np.mean(sizes))
        print('median of all sizes: ', np.median(sizes))
        print('mode of all sizes: ', stats.mode(sizes))
        print('min of all sizes: ', min(sizes))
        print('max of all sizes: ', max(sizes))

if __name__ == '__main__':
    pp = PreprocessData()
    pp.start()