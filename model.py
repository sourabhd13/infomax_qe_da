import torch
from torch import nn
from transformers import BertModel, BertConfig
from modules.encoders import CPC, MMILB, SubNet

class MMIM(nn.Module):
    def __init__(self, hp, bert_config1, bert_config2, bert_config3):
        """
        Construct MultiModal InfoMax model.
        Args: 
            hp (dict): a dict stores training and model configurations
            bert_config1, bert_config2, bert_config3 (dict): configurations for the three BERT models
        """
        super().__init__()
        self.hp = hp
        hp.d_tout = hp.d_tin

        # Text Encoders (three BERT models)
        self.text_enc1 = BertModel(BertConfig(**bert_config1))
        self.text_enc2 = BertModel(BertConfig(**bert_config2))
        self.text_enc3 = BertModel(BertConfig(**bert_config3))

        # For MI maximization
        self.mi_tt1 = MMILB(
            x_size=hp.d_tout,
            y_size=hp.d_tout,
            mid_activation=hp.mmilb_mid_activation,
            last_activation=hp.mmilb_last_activation
        )
        self.mi_tt2 = MMILB(
            x_size=hp.d_tout,
            y_size=hp.d_tout,
            mid_activation=hp.mmilb_mid_activation,
            last_activation=hp.mmilb_last_activation
        )
        self.mi_tt3 = MMILB(
            x_size=hp.d_tout,
            y_size=hp.d_tout,
            mid_activation=hp.mmilb_mid_activation,
            last_activation=hp.mmilb_last_activation
        )

        dim_sum = 3 * hp.d_tout

        # CPC MI bound
        self.cpc_zt1 = CPC(
            x_size=hp.d_tout,
            y_size=hp.d_prjh,
            n_layers=hp.cpc_layers,
            activation=hp.cpc_activation
        )
        self.cpc_zt2 = CPC(
            x_size=hp.d_tout,
            y_size=hp.d_prjh,
            n_layers=hp.cpc_layers,
            activation=hp.cpc_activation
        )
        self.cpc_zt3 = CPC(
            x_size=hp.d_tout,
            y_size=hp.d_prjh,
            n_layers=hp.cpc_layers,
            activation=hp.cpc_activation
        )

        # Fusion Settings
        self.fusion_prj = SubNet(
            in_size=dim_sum,
            hidden_size=hp.d_prjh,
            n_class=hp.n_class,
            dropout=hp.dropout_prj
        )

    def forward(self, sentences1, sentences2, sentences3, y=None, mem=None):
        """
        sentences1, sentences2, sentences3 should have dimension [batch_size, seq_len]
        """
        # Text encoding with three different BERT models
        enc_word1 = self.text_enc1(sentences1)['last_hidden_state']  # (batch_size, seq_len, emb_size)
        enc_word2 = self.text_enc2(sentences2)['last_hidden_state']  # (batch_size, seq_len, emb_size)
        enc_word3 = self.text_enc3(sentences3)['last_hidden_state']  # (batch_size, seq_len, emb_size)

        text1 = enc_word1[:, 0, :]  # (batch_size, emb_size)
        text2 = enc_word2[:, 0, :]  # (batch_size, emb_size)
        text3 = enc_word3[:, 0, :]  # (batch_size, emb_size)

        # Mutual Information Maximization between text modalities
        if y is not None:
            lld_tt1, tt1_pn, H_tt1 = self.mi_tt1(x=text1, y=text2, labels=y, mem=mem['tt1'])
            lld_tt2, tt2_pn, H_tt2 = self.mi_tt2(x=text1, y=text3, labels=y, mem=mem['tt2'])
            lld_tt3, tt3_pn, H_tt3 = self.mi_tt3(x=text2, y=text3, labels=y, mem=mem['tt3'])
        else:
            lld_tt1, tt1_pn, H_tt1 = self.mi_tt1(x=text1, y=text2)
            lld_tt2, tt2_pn, H_tt2 = self.mi_tt2(x=text1, y=text3)
            lld_tt3, tt3_pn, H_tt3 = self.mi_tt3(x=text2, y=text3)

        # Concatenate the encoded texts
        text_concat = torch.cat([text1, text2, text3], dim=1)  # (batch_size, 3*emb_size)

        # CPC computation between text modalities and fusion
        nce_zt1 = self.cpc_zt1(text1, self.fusion_prj(text_concat))
        nce_zt2 = self.cpc_zt2(text2, self.fusion_prj(text_concat))
        nce_zt3 = self.cpc_zt3(text3, self.fusion_prj(text_concat))
        
        nce = nce_zt1 + nce_zt2 + nce_zt3

        pn_dic = {'tt1': tt1_pn, 'tt2': tt2_pn, 'tt3': tt3_pn}
        lld = lld_tt1 + lld_tt2 + lld_tt3
        H = H_tt1 + H_tt2 + H_tt3
        

        return lld, nce, preds, pn_dic, H

