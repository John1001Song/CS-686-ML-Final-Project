import json
import os
from src.open_data import OPCODES


PATH_NEW = '../dataset/'
PATH_OLD = '../Marion_files/sm_database/'
PATH_NEW_OP_P = PATH_NEW + 'ponzi/op_count/'
PATH_NEW_OP_NP = PATH_NEW + 'non_ponzi/op_count/'
PATH_OLD_OP_P = PATH_OLD + 'opcode/opcodes_count/'
PATH_OLD_OP_NP = PATH_OLD + 'opcode_np/opcode_count/bytecode_np/'


def compare_two_op_freq(op_old, op_new):
    for i in range(len(OPCODES)):
        if op_old != op_new:
            return False
    return True


def start():
    op_old = [
        [fname.split('.json')[0] for fname in os.listdir(PATH_OLD_OP_P) if fname.endswith('.json')],
        [fname.split('.json')[0] for fname in os.listdir(PATH_OLD_OP_NP) if fname.endswith('.json')]
    ]
    print(f"op_old: {len(op_old[0])}, {len(op_old[1])}")
    op_new = [
        [fname.split('.csv')[0] for fname in os.listdir(PATH_NEW_OP_P) if fname.endswith('.csv')],
        [fname.split('.csv')[0] for fname in os.listdir(PATH_NEW_OP_NP) if fname.endswith('.csv')]
    ]
    print(f"op_new: {len(op_new[0])}, {len(op_new[1])}")
    with open(PATH_OLD + 'op_freq.json', 'rb', ) as f:
        op_freq_old = json.loads(f.read())
    print(f"op_freq_old: {len(op_freq_old[0])}, {len(op_freq_old[1])}")
    with open(PATH_NEW + 'op_freq.json', 'rb', ) as f:
        op_freq_new = json.loads(f.read())
    print(f"op_freq_new: {len(op_freq_new[0])}, {len(op_freq_new[1])}")
    for op_index in range(2):
        for i in range(len(op_old[op_index])):
            contract_hash = op_old[op_index][i]
            if contract_hash not in op_new[op_index]:
                continue
            index_op_new = op_new[op_index].index(contract_hash)
            if not compare_two_op_freq(op_freq_old[op_index][i], op_freq_new[op_index][index_op_new]):
                print(f"{op_index}, contract_hash: {contract_hash}, {op_new[op_index][index_op_new]}")


if __name__ == '__main__':
    start()
