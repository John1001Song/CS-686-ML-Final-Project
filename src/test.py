import pandas as pd
import torch
import numpy as np


# df = pd.read_csv('../dataset/op_int_equal_dataframe/op_int_equal.csv')
# df = pd.read_csv('../dataset/op_int_equal_dataframe/op_int_equal_pd.csv')
# print(df.head())
# print(df.columns)
# df = pd.read_csv('../dataset/op_int_equal_dataframe/op_int_equal_new.csv')
# print(df.iloc[100, :])
# print(len(df.iloc[100, :]))
# print(len(df.iloc[10, :]))
# print(len(df.iloc[6000, :]))
# print(type(df.iloc[100, :]))
# for index, row in df.iterrows():
#     temp_list = []
#     row['sequence'] = row['sequence'][1:-1]
#     for op_int in row['sequence']:
#         if op_int == ',' or ' ':
#             continue
#         temp_list.append(int(op_int))
#     row['sequence'] = temp_list

# dictionary of lists

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

print(len(OPCODES))