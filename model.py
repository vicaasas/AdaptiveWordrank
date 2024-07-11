import torch
from torch import nn
import torch.nn.functional as F
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
# from modeling_bart_origin import BartForConditionalGeneration
from modeling_bart_SCAN import BartForConditionalGeneration
# from modeling_pegasus_origin import PegasusForConditionalGeneration
from modeling_pegasus_SCAN import PegasusForConditionalGeneration


class AdaptiveWordrank(nn.Module):
    
    def __init__(self, mname, pad_token_id,config, is_pegasus=False):
        super(AdaptiveWordrank, self).__init__()
        if is_pegasus:
            self.model = PegasusForConditionalGeneration.from_pretrained(mname, cache_dir="./local_cache")
        else:
            # self.model = BartScorer.from_pretrained(mname, cache_dir="./local_cache")
            # self.model = BartForConditionalGeneration.from_pretrained(mname, cache_dir="./local_cache")
            self.model = BartForConditionalGeneration.from_pretrained(mname)
            # self.model = BartForConditionalGeneration(config=config)
        self.pad_token_id = pad_token_id
    def get_encoder(self):
        return self.model.get_encoder()
    def forward(self, text_id,candidate_id,keyword_ids=None,keyword_position=None,src_word_position=None,labels=None, normalize=True, score_mode="base", length_penalty=1, require_gold=True, adding=0):
        
        batch_size = text_id.size(0)
        
        input_mask = text_id != self.pad_token_id
        
        cand_mask = candidate_id != self.pad_token_id
        # print(cand_mask[:, :, 0])
        cand_mask[:, 0] = 1
        # print(cand_mask.shape)
        output = self.model(
            input_ids=text_id, 
            # labels=labels,
            # keyword_ids=keyword_ids,
            # src_word_position=src_word_position,
            attention_mask=input_mask,
            decoder_input_ids=candidate_id, 
            decoder_attention_mask=cand_mask,
            output_hidden_states=True,
            keyword_position=keyword_position,
        )
        # encoder_last_hidden_state=output["encoder_last_hidden_state"]
        word_energy=output["word_energy"]
        # CK_loss = output['CK_loss']
        output = output['logits']  # [bz x cand_num, seq_len, word_dim]
        
        probs = output.view(batch_size, output.size(1), output.size(2))

        if require_gold:
            output = {
                # 'score': scores[:, 1:],
                # "summary_score": scores[:, 0], 
                "probs": probs,
                # "decoder_hidden_states":decoder_hidden_states
                # "encoder_last_hidden_state":encoder_last_hidden_state,
                "word_energy":word_energy,
                # "masked_lm_loss":masked_lm_loss
                # "CK_loss":CK_loss
            }
        # else:
        #     output = {'score': scores, "probs": probs}
        return output
    def get_encoder_logits(self, input_ids, attention_mask=None):


        encoder_outputs = self.model.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # logits and loss for the encoder
        encoder_last_hidden_state = encoder_outputs['last_hidden_state']  # last hidden state
        encoder_logits = self.model.energy_net(encoder_last_hidden_state)

        return encoder_outputs, encoder_logits
    def scoring_mode(self):
        self.model.model.scoring_mode()

    def generation_mode(self):
        self.model.model.generation_mode()



    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask=None,
        no_repeat_ngram_size=None,
        length_penalty=None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        early_stopping: Optional[bool] = None,
        num_beams: Optional[int] = None,
        src_word_position=None,
    ):

        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # segment_ids=segment_ids,
            max_length=max_length,
            min_length=min_length,
            early_stopping=early_stopping,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            length_penalty=length_penalty,
            src_word_position=src_word_position,
            )