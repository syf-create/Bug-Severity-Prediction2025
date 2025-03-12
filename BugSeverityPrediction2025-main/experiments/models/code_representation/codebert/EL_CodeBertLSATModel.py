import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel

torch.backends.cudnn.deterministic = True
import torch.nn.functional as F
from transformers.modeling_outputs import SequenceClassifierOutput


class TextRNNAtten(nn.Module):
    def __init__(self, config):
        super(TextRNNAtten, self).__init__()
        hidden_size = config.hidden_size  # 隐藏层数量
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        num_classes = config.num_labels  # 类别数
        num_layers = 2  # 双层LSTM

        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers,
                            bidirectional=True, batch_first=True, dropout=classifier_dropout)
        self.tanh = nn.Tanh()
        self.w = nn.Parameter(torch.zeros(hidden_size * 2))
        self.dense = nn.Linear(hidden_size * 2, hidden_size)
        self.residual_dense = nn.Linear(hidden_size * 3, hidden_size)  # 新增，用于处理拼接后的张量
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, cls_embedding):
        # x = [batch size, 12, hidden_size]
        x = self.dropout(x)
        # x = [batch size, 12, hidden_size]
        output, (hidden, cell) = self.lstm(x)
        # output = [batch size, 12, num_directions * hidden_size]
        M = self.tanh(output)
        # M = [batch size, 12, num_directions * hidden_size]
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)
        # alpha = [batch size, 12, 1]
        out = output * alpha
        # out = [batch size, 12, num_directions * hidden_size]
        out = torch.sum(out, 1)
        # out = [batch size, num_directions * hidden_size].
        # 将 out 和 cls_embedding 连接起来
        out = torch.cat((out, cls_embedding),
                        dim=1)  # 拼接后的 out 形状为 [batch size, num_directions * hidden_size + hidden_size]

        out = F.gelu(out)
        # out = [batch size, num_directions * hidden_size + hidden_size]
        out = self.residual_dense(out)  # 通过全连接层将维度还原到 hidden_size
        # out = [batch size, hidden_size]
        out = self.dropout(out)
        # out = [batch size, hidden_size]
        out = self.fc(out)
        # out = [batch size, num_classes]
        return out, alpha


class TextRNNAtten_GRU(nn.Module):
    def __init__(self, config):
        super(TextRNNAtten_GRU, self).__init__()
        hidden_size = config.hidden_size
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        num_classes = config.num_labels

        # 单层且单向 GRU，设置 num_layers=1 且 bidirectional=False
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=1,
                          bidirectional=False, batch_first=True, dropout=0)  # 单层且单向 GRU
        self.tanh = nn.Tanh()
        self.w = nn.Parameter(torch.zeros(hidden_size))  # 单向 GRU 输出维度为 hidden_size
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.residual_dense = nn.Linear(hidden_size * 2, hidden_size)  # 用于拼接后还原维度
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, cls_embedding):
        # x = [batch size, 12, hidden_size]
        x = self.dropout(x)
        # x = [batch size, 12, hidden_size]
        output, hidden = self.gru(x)
        # output = [batch size, 12, hidden_size] (单层单向 GRU 输出维度为 hidden_size)

        M = self.tanh(output)
        # M = [batch size, 12, hidden_size]

        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)
        # alpha = [batch size, 12, 1]

        out = output * alpha
        # out = [batch size, 12, hidden_size]

        out = torch.sum(out, 1)
        # out = [batch size, hidden_size]

        # 将 out 和 cls_embedding 连接起来
        out = torch.cat((out, cls_embedding), dim=1)
        # 拼接后的 out 形状为 [batch size, hidden_size + hidden_size]

        out = F.gelu(out)
        out = self.residual_dense(out)  # 通过全连接层将维度还原到 hidden_size
        out = self.dropout(out)
        out = self.fc(out)
        # out = [batch size, num_classes]

        return out, alpha


class TextRNNAtten_LSTM(nn.Module):
    def __init__(self, config):
        super(TextRNNAtten_LSTM, self).__init__()
        hidden_size = config.hidden_size  # 隐藏层数量
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        num_classes = config.num_labels  # 类别数
        num_layers = 1  # 单层 LSTM

        # 使用单层单向 LSTM
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers,
                            bidirectional=False, batch_first=True, dropout=classifier_dropout)
        self.tanh = nn.Tanh()
        self.w = nn.Parameter(torch.zeros(hidden_size))  # 注意单向时维度为 hidden_size
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.residual_dense = nn.Linear(hidden_size * 2, hidden_size)  # 新增，用于处理拼接后的张量
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, cls_embedding):
        # x = [batch size, 12, hidden_size]
        x = self.dropout(x)
        # x = [batch size, 12, hidden_size]
        output, (hidden, cell) = self.lstm(x)
        # output = [batch size, 12, hidden_size]  # 单向时没有 num_directions
        M = self.tanh(output)
        # M = [batch size, 12, hidden_size]
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)
        # alpha = [batch size, 12, 1]
        out = output * alpha
        # out = [batch size, 12, hidden_size]
        out = torch.sum(out, 1)
        # out = [batch size, hidden_size]
        # 将 out 和 cls_embedding 连接起来
        out = torch.cat((out, cls_embedding),
                        dim=1)  # 拼接后的 out 形状为 [batch size, hidden_size + hidden_size]

        out = F.gelu(out)
        # out = [batch size, hidden_size + hidden_size]
        out = self.residual_dense(out)  # 通过全连接层将维度还原到 hidden_size
        # out = [batch size, hidden_size]
        out = self.dropout(out)
        # out = [batch size, hidden_size]
        out = self.fc(out)
        # out = [batch size, num_classes]
        return out, alpha


class EL_CodeBertLSATModel(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super().__init__()
        self.encoder = encoder
        self.num_labels = config.num_labels
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

        # self.classifier_lstm = TextRNNAtten(config)
        #self.classifier_lstm = TextRNNAtten_GRU(config)  # 使用单层单向 GRU
        self.classifier_lstm = TextRNNAtten_LSTM(config)  # 使用单层单向 LSTM

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_layer = nn.Linear(config.hidden_size, config.num_labels)


        # 类别权重初始化
        self.class_weights = None
    def set_class_weights(self, class_weight):
        """设置类权重"""
        self.class_weights = torch.FloatTensor([class_weight[0], class_weight[1], class_weight[2], class_weight[3]]).to(self.args.device)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            num_features=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            output_attentions=True,
            output_hidden_states=True,
        )

        hidden_states = outputs.hidden_states  # 13*[bs, seq_len, hidden] 第一层是embedding层不需要
        cls_embeddings = hidden_states[1][:, 0, :].unsqueeze(1)  # [bs, 1, hidden]
        # 将每一层的第一个token(cls向量)提取出来，拼在一起当作LSTM的输入
        for i in range(2, 13):
            cls_embeddings = torch.cat((cls_embeddings, hidden_states[i][:, 0, :].unsqueeze(1)), dim=1)
        # cls_embeddings: [bs, encode_layer=12, hidden]

        # 获取第12层的cls向量
        cls_embedding_12th = hidden_states[12][:, 0, :]  # [bs, hidden]

        logits, alpha = self.classifier_lstm(cls_embeddings, cls_embedding_12th)
        prob = torch.softmax(logits, -1)

        # 计算损失，并应用类别权重
        loss = None
        if labels is not None:
            # 确保类权重已经设置
            if self.class_weights is not None:
                loss_fct = CrossEntropyLoss(weight=self.class_weights)
            else:
                loss_fct = CrossEntropyLoss()  # 无权重时使用标准交叉熵损失
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, prob
        else:
            return prob








