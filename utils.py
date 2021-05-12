import json
import numpy as np
import torch
import pandas as pd
from fuzzy_match import match


def load_json(fname):
    with open(fname, 'r', encoding = 'utf-8') as json_file:
        return json.load(json_file)


def read_lexicon(fname):
    with open(fname, 'r', encoding = 'utf-8') as f:
        return set(f.read().split('\n'))


def all_norms(word, morph):
    result = set([var.normal_form for var in morph.parse(word)])
    return result


def load_vectors(fname):
    pad_token = 'PAD'
    vocab = dict()
    
    embeddings = list()
    embedding_dim = 300
    
    vocab[pad_token] = len(vocab)
    embeddings.append(np.zeros(embedding_dim))
    
    with open(fname, 'r', encoding = 'utf-8') as f:
        a = f.readline()
        for line in f:
            parts = line.strip().split()
            token = ' '.join(parts[:-embedding_dim])
            if token in vocab:
                continue
            word_vector = np.array(list(map(float, parts[-embedding_dim:]))) 
            vocab[token] = len(vocab)
            embeddings.append(word_vector)
    
    embeddings = np.stack(embeddings)
    embeddings = torch.tensor(embeddings).float()
    return vocab, embeddings


def recovery(name, dmd_recovery_dict):
    rec_name, score = match.extract(name, list(dmd_recovery_dict.keys()), limit=1)[0]
    if score >= 0.15:
        return dmd_recovery_dict[rec_name]
    else:
        return name

def prepare_dict(cur_df, dmd_recovery_dict):
    intake_dmd = cur_df.loc[(cur_df['label'] != 'O')&(cur_df['dmd_tag'] != 'O'), ['token', 'dmd_tag']]
    intake_dmd = intake_dmd.reset_index().to_numpy()
    intake_dict = {}
    cur_dict = {'dmd_names': [], 'inds':[]}
    cur_name = []

    for i, row in enumerate(intake_dmd):
        cur_dict['inds'].append(row[0])
        cur_name.append(row[1])
        try:
            if intake_dmd[i+1][2] == 'B-DMD':
                cur_name = ' '.join(cur_name)
                cur_dict['dmd_names'].append(cur_name)
                intake_dict[len(intake_dict)] = cur_dict
                cur_dict = {'dmd_names': [], 'inds':[]}
                cur_name = []
        except IndexError:
            cur_name = ' '.join(cur_name)
            cur_dict['dmd_names'].append(cur_name)
            intake_dict[len(intake_dict)] = cur_dict
            
    for i in range(len(intake_dict)):
        intake_dict[i]['full_dmd_name'] = recovery(intake_dict[i]['dmd_names'][0], dmd_recovery_dict)        
    
    return intake_dict

def prepare_all_dmds(cur_df, dmd_recovery_dict):
    all_dmd = cur_df.loc[cur_df['dmd_tag'] != 'O', ['token', 'dmd_tag']]
    all_dmd = all_dmd.reset_index().to_numpy()
    list_of_all_dmds = []
    cur_dict = {'dmd_names': '', 'inds':[]}
    cur_name = []
    for i, row in enumerate(all_dmd):
        cur_dict['inds'].append(row[0])
        cur_name.append(row[1])
        try:
            if all_dmd[i+1][2] == 'B-DMD':
                cur_name = ' '.join(cur_name)
                cur_dict['dmd_names'] = cur_name
                list_of_all_dmds.append(cur_dict)
                cur_dict = {'dmd_names': '', 'inds':[]}
                cur_name = []
        except IndexError:
            cur_name = ' '.join(cur_name)
            cur_dict['dmd_names'] = cur_name
            list_of_all_dmds.append(cur_dict)
            
    for i, dmd in enumerate(list_of_all_dmds):
        list_of_all_dmds[i]['full_dmd_name'] = recovery(list_of_all_dmds[i]['dmd_names'], dmd_recovery_dict)
    
    return list_of_all_dmds

def add_dmd_inds(intake_dict, list_of_all_dmds):
    for i in range(len(intake_dict)):
        for dmd in list_of_all_dmds:
            if set(dmd['inds']) & set(intake_dict[i]['inds']):
                continue
            else:
                if intake_dict[i]['full_dmd_name'] == dmd['full_dmd_name']:
                    intake_dict[i]['inds'].extend(dmd['inds'])
                    intake_dict[i]['dmd_names'].append(dmd['dmd_names'])
                    
def remove_same(intake_dict):
    for i in range(len(intake_dict)):
        intake_dict[i]['inds'].sort()
        intake_dict[i]['dmd_names'].sort()
    clean_intake_dict = {}
    for i in range(len(intake_dict)):
        if not intake_dict[i] in clean_intake_dict.values():
            clean_intake_dict[len(clean_intake_dict)] = intake_dict[i]
    return clean_intake_dict

def create_dmd_json(cur_df, dmd_recovery_dict):
    intake_dict = prepare_dict(cur_df, dmd_recovery_dict)
    list_of_all_dmds = prepare_all_dmds(cur_df, dmd_recovery_dict)
    add_dmd_inds(intake_dict, list_of_all_dmds)
    clean_intake_dict = remove_same(intake_dict)
    return clean_intake_dict


def prepare_all_adr_tokens(inds, df):
    cur_df = df.iloc[inds][['token', 'adr_label']].copy()
    cur_df = cur_df.reset_index().to_numpy()
    list_of_all_adrs = []
    cur_adr = {'class': '', 'tokens':[], 'inds':[]}
    for i, row in enumerate(cur_df):
        cur_adr['class'] = row[2].split('-')[-1]
        cur_adr['inds'].append(row[0])
        cur_adr['tokens'].append(row[1])
        try:
            if cur_df[i+1][2][0] == 'B': #cur_df[i+1][2].split('-')[-1] != cur_adr['class'] or 
                list_of_all_adrs.append(cur_adr)
                cur_adr = {'class': '', 'tokens':[], 'inds':[]}
        except IndexError:
            list_of_all_adrs.append(cur_adr)
    return list_of_all_adrs