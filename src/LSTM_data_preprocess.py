import pandas as pd
import os

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

    def start(self):
        self.define_path()
        # self.check_all_opcodes()
        # self.get_opcode_int()
        self.get_opcode_int_without_revert()

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
                    # print('!!!!!!!!!', line)
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
            op_int_list.append(self.opcodes.index(op))

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
            op_int_list.append(self.opcodes.index(op))

        with open(output_path+hex_addr.split('.json')[0]+'.txt', 'w') as f_op_int:
            for item in op_int_list:
                f_op_int.write("%s\n" % item)


if __name__ == '__main__':
    pp = PreprocessData()
    pp.start()
