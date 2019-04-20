import os
import ast
import json
import csv
from web3 import Web3
import pandas as pd
import pprint
import urllib3

# import urllib.request, json
#
# with urllib.request.urlopen("https://etherscan.io/api?module=opcode&action=getopcode&address=0x0b230b071008bbb145b5bff27db01c9248f486b9") as url:
#     data = json.loads(url.read().decode())
#     print(data)

http = urllib3.PoolManager()

heroes = http.request('GET', 'https://etherscan.io/api?module=opcode&action=getopcode&address=0x0b230b071008bbb145b5bff27db01c9248f486b9')

print(heroes.data.decode('UTF-8'))

heroes_dict = json.loads(heroes.data.decode('UTF-8'))
print(heroes_dict)

print(heroes_dict['result'])

op_code = heroes_dict['result'].split('<br>')

print(op_code)

str_list = list(filter(None, op_code)) # fastest

print(str_list)

OPCODES = ['SWAP8', 'DUP11', 'DUP14', 'SWAP10', 'DUP15', 'LOG2', 'INVALID', 'SWAP9', 'SWAP5', 'SWAP12', 'SWAP16',
           'DUP9', 'LOG1', 'DUP12', 'SWAP11', 'SWAP2', 'MSTORE8', 'SWAP14', 'DUP13', 'POP', 'DUP8','DUP7',
           'DUP3', 'DUP4', 'MSTORE', 'SWAP3', 'CODECOPY', 'JUMP', 'DUP5', 'SWAP13', 'STOP', 'CALLDATACOPY', 'SWAP7',
           'SWAP1', 'SWAP6', 'RETURN', 'DUP6', 'SWAP4', 'REVERT', 'SELFDESTRUCT', 'DUP10', 'DUP16', 'JUMPI',
           'SSTORE', 'PUSH', 'LOG3', 'LOG4', 'Missing', 'SWAP15', 'DUP1&2']


OPCODES_2 = ['STOP', 'ADD', 'MUL', 'SUB', 'DIV', 'SDIV', 'MOD', 'SMOD', 'ADDMOD', 'MULMOD', 'EXP', 'SIGNEXTEND',

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

             'CREATE', 'CALL', 'CALLCODE', 'RETURN', 'DELEGATECALL', 'STATICCALL', 'REVERT', 'SELFDESTRUCT']



op_to_int = []

for op in str_list:
    if op not in OPCODES:
        op_to_int.append(-1)
        print('un_doc op: ', op)
    else:
        op_to_int.append(OPCODES.index(op))

print(op_to_int)