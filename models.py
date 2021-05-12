import torch
import torch.nn as nn
import torch.nn.functional as F

class DMDtagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, char_vocab_size, tagset_size, max_len=12):
        super().__init__()

        self.char_embeddings = nn.Embedding(char_vocab_size, embedding_dim)       
        self.char_lstm = nn.LSTM(embedding_dim*max_len, hidden_dim, num_layers = 3, dropout = 0.4, batch_first=True, bidirectional=True)        
        self.char_linear = nn.Linear(hidden_dim*2, tagset_size)

    def forward(self, sentence):
        char_embeds = [self.char_embeddings(word) for word in sentence]
        char_embeds = torch.stack(char_embeds)
        char_embeds = torch.flatten(char_embeds, start_dim=2)
        lstm_out, _ = self.char_lstm(char_embeds)
        tag_space = self.char_linear(lstm_out)
        return tag_space


class Intaketagger(nn.Module):

    def __init__(self, embeddings, embedding_dim, hidden_dim, tagset_size):
        super().__init__()
        self.dmd_embeddings = nn.Embedding(3, 2)    
        self.word_embeddings = nn.Embedding.from_pretrained(embeddings, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim+2, hidden_dim, num_layers = 5, dropout = 0.4, batch_first=True, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim*2+2, tagset_size)

    def forward(self, sentence, dmd_tags):
        
        embeds = self.word_embeddings(sentence)    
        dmd_embeds = self.dmd_embeddings(dmd_tags)
        res = torch.cat((embeds, dmd_embeds), 2)
        lstm_out, _ = self.lstm(res)
        lstm_out = torch.cat((lstm_out, dmd_embeds), 2)
        tag_space = self.hidden2tag(lstm_out)
        return tag_space


class ADRtagger(nn.Module):

    def __init__(self, embeddings, embedding_dim, hidden_dim, tagset_size): 
        super().__init__()
        self.word_embeddings = nn.Embedding.from_pretrained(embeddings, padding_idx=0)
        self.tag_embeddings = nn.Embedding(5, 3)      
        self.dropout = nn.Dropout(0.4)
        self.lstm = nn.LSTM(embedding_dim+3, hidden_dim, num_layers = 5, dropout = 0.4, batch_first=True, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim*2+3, tagset_size)

    def forward(self, sentence, tags):
        embeds = self.word_embeddings(sentence)
        tag_embeds = self.tag_embeddings(tags)    
        embeds = torch.cat((embeds, tag_embeds), 2)
        embeds = self.dropout(embeds)   
        lstm_out, _ = self.lstm(embeds)
        res = torch.cat((lstm_out, tag_embeds), 2)
        tag_space = self.hidden2tag(res)
        return tag_space


class Linktagger(nn.Module):

    def __init__(self, embeddings, embedding_dim, hidden_dim, tagset_size):
        super().__init__()  
        self.word_embeddings = nn.Embedding.from_pretrained(embeddings, padding_idx=0)
        self.adr_embeddings = nn.Embedding(2, 2)
        self.drug_embeddings = nn.Embedding(3, 3)        
        self.dropout = nn.Dropout(0.4)
        self.lstm = nn.LSTM(embedding_dim+5, hidden_dim, num_layers = 5, dropout = 0.4, batch_first=True, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim*2+5, tagset_size)

    def forward(self, sentence, adr, drug):
        embeds = self.word_embeddings(sentence)
        adr_embeds = self.adr_embeddings(adr)
        drug_embeds = self.drug_embeddings(drug)       
        embeds = torch.cat((embeds, adr_embeds, drug_embeds), 2)
        embeds = self.dropout(embeds)       
        lstm_out, _ = self.lstm(embeds)
        res = torch.cat((lstm_out, adr_embeds, drug_embeds), 2)
        tag_space = self.hidden2tag(res)
        return tag_space