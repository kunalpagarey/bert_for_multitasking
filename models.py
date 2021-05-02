
"""
  This code is an implementation of bert for multi-task training.
  1. It can be trained as a Sequence classifier by not providing regression output size or providing it as zero.
  2. It can be trained in multi-task setup i.e. Sequence classifier and predicting valence-arousal (va) values for the sequence.
  3. NextEmotionPredictor task is to predict next emotion for a bot with which it responds to user response. It needs one-hot encoding of current emotion 
     of the user to concatenate with output embedding of bert and furether fed to a linear classifier layer.
"""


from typing import Optional, Tuple
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertPreTrainedModel, BertModel
from transformers.file_utils import ModelOutput


class SequenceClassifierOutput(ModelOutput):
    cls_loss: Optional[torch.FloatTensor] = None
    cls_logits: torch.FloatTensor = None
    va_loss: Optional[torch.FloatTensor] = None
    va_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class BertForMultiTaskClassification(BertPreTrainedModel):
    def __init__(self, config, va=0):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.finetuning_task = config.finetuning_task
        self.va = va
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if self.finetuning_task == "NextEmotionPredictor":
            self.nep_linear = nn.Linear(config.hidden_size + config.num_labels, 512)
            self.classifier = nn.Linear(512, config.num_labels)
        else:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        if self.va != 0:
            self.va_linear1 = nn.Linear(config.hidden_size, 256)
            self.va_linear2 = nn.Linear(256, self.va)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        va_labels=None,
        current_emotion=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        
        if self.finetuning_task == "NextEmotionPredictor":
            if current_emotion is not None:
                nep_input = torch.cat((pooled_output, current_emotion), dim=1)
                nep_input = self.nep_linear(nep_input)
                nep_input = self.dropout(nep_input)
                cls_logits = self.classifier(nep_input)
            else:
                raise ValueError("Need current emotions (one-hot encoded) detected from emotion detector.")
        else:
            cls_logits = self.classifier(pooled_output)
        
        if self.va != 0:
            va_input = self.va_linear1(pooled_output)
            va_input = self.dropout(va_input)
            va_logits = self.va_linear2(va_input)  # size = batch, va_coordinates
        else:
            va_logits = None

        cls_loss = None
        va_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            cls_loss = loss_fct(cls_logits.view(-1, self.num_labels), labels.view(-1))
        
        if va_labels is not None:
                #  We are doing regression
                loss_fct = MSELoss()
                va_loss = loss_fct(va_logits.view(-1), va_labels.view(-1))

        return SequenceClassifierOutput(
            va_loss=va_loss,
            va_logits=va_logits,
            cls_loss=cls_loss,
            cls_logits=cls_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
