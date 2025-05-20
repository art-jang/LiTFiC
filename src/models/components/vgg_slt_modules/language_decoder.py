import torch
import torch.nn as nn
import random

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils.data_utils import get_unique_bg_words, drop_words


class LanguageDecoder(nn.Module):
    def __init__(self,
                 pretrained_llm,
                 tokenizer_config,
                 decoder_config,
                 lora_config,
                 gradient_checkpointing_enable=False,
                 freeze_decoder=False,
                 precision='float32',
                 lora = False,
                 use_pl_w_feats = False,
                 mix_in_pls_prob = 0.5,
                 bg_desc = False,
                 use_rec_prev = False,
                 mix_in_prev_prob = 0.5,
                 mix_in_bg_prob = 0.5,
                 use_bg_words = False,
                 drop_bg_sw = False,
                 use_gt_prev = False,
                 drop_bgw_pct = 0.0,
                 drop_pl_pct = 0.0,
                 use_spottings = False,
                 mix_in_spottings = 0.0,
                 dropout=0.0,
                 **kwargs):
        super(LanguageDecoder, self).__init__()
        if precision == "bf16-mixed":
            torch_dtype = torch.bfloat16
        elif precision == "16-mixed":
            torch_dtype = torch.float16
        elif precision == "32-true":
            torch_dtype = torch.float32
            decoder_config.attn_implementation = None
        else:
            raise ValueError(f"Invalid precision: {precision}")
        
        self.torch_dtype = torch_dtype
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_llm, **tokenizer_config)
        self.add_eos_token = True 

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.decoder = AutoModelForCausalLM.from_pretrained(pretrained_llm, 
                                                            torch_dtype=torch_dtype,
                                                            **decoder_config)
        self.torch_dtype = torch_dtype

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.embed_tokens = self.decoder.model.embed_tokens
        self.lora = lora
        if lora:
            lora_config = LoraConfig(**lora_config)
            self.decoder = get_peft_model(self.decoder, lora_config)

        if gradient_checkpointing_enable:
            self.decoder.gradient_checkpointing_enable()
        if freeze_decoder and not lora:
            for param in self.decoder.parameters():
                param.requires_grad = False
        

        self.use_pl_w_feats = use_pl_w_feats
        self.mix_in_pls_prob = mix_in_pls_prob
        self.bg_desc = bg_desc
        self.use_rec_prev = use_rec_prev
        self.mix_in_prev_prob = mix_in_prev_prob
        self.mix_in_bg_prob = mix_in_bg_prob

        self.use_bg_words = use_bg_words
        self.drop_bg_sw = drop_bg_sw

        self.use_gt_prev = use_gt_prev

        self.drop_bgw_pct = drop_bgw_pct
        self.drop_pl_pct = drop_pl_pct

        self.use_spottings = use_spottings
        self.mix_in_spottings = mix_in_spottings
        
        assert (self.use_gt_prev and self.use_rec_prev) == False, "use_gt_prev and use_rec_prev cannot be used together. Please set one of them to False."

    def _tokenize(self, text, device='cpu'):
        input_ids = self.tokenizer(text, return_tensors='pt')['input_ids'][0].to(device)
        if self.add_eos_token:
            input_ids = torch.cat([input_ids, torch.LongTensor([self.tokenizer.eos_token_id]).to(device)], dim=0)
        return input_ids
    
    def _process(self, x, video_masks, subtitles, questions=None, previous_contexts=None, device='cpu', ignore_idx=-100, pls=None, background_description=None, spottings=None):

        final_q = None

        # Initial prompt setup for questions
        # questions = ['You are an AI assistant designed to interpret a video of a sign language signing sequence and translate it into English.' for _ in questions]
        
        # Add previous contexts if applicable
        if previous_contexts is not None and self.use_rec_prev and (random.random() <= self.mix_in_prev_prob or not self.training):
            questions = [q + f' The previous context is the following: {c}' if c else q for q, c in zip(questions, previous_contexts)]
        
        if self.use_spottings and (random.random() <= self.mix_in_spottings or not self.training):
            questions = [q + ' The following are some possible words present in the sentence: ' + ", ".join(s) if len(s) > 0 else "Not available" + '.' for q, s in zip(questions, spottings)]
        
        if self.use_pl_w_feats and (random.random() <= self.mix_in_pls_prob or not self.training):
            questions = [q + f' The following are some possible words present in the sentence: {", ".join(drop_words(pl, self.drop_pl_pct)) if len(pl) > 0 else "Not available"}.' for q, pl in zip(questions, pls)]
            
        # Add background description if applicable
        if self.bg_desc and (random.random() <= self.mix_in_bg_prob or not self.training):
            if self.use_bg_words:
                bg_desc = [drop_words(get_unique_bg_words(b, drop_sw=self.drop_bg_sw), self.drop_bgw_pct) for b in background_description]
                questions = [q + f' Description of the background is: {", ".join(b) if len(b) > 0 else "Not available"}.' for q, b in zip(questions, bg_desc)]
            else:
                questions = [q + f' Description of the background is: {b}' for q, b in zip(questions, background_description)]
            
        
        # Finalize question prompt for video tokens
        questions = [q + " The following are the video tokens: " for q in questions]
        final_q = [".\nGenerated Sentence: " for _ in questions]


        max_len = 0
        labels = []
        inputs_embeds = []
        for i in range(x.shape[0]):
            cur_v_embed = x[i, :int(video_masks[i].sum())].to(device) # [num_v_tokens, C]
            cur_s = self._tokenize(subtitles[i], device=device)[1:] # remove bos token
            
            if questions is not None:
                cur_q = self._tokenize(questions[i], device=device)[:-1] # remove eos token
                cur_q_embed = self.embed_tokens(cur_q)


                if final_q is not None:
                    cur_final_q = self._tokenize(final_q[i], device=device)[1:-1]
                    cur_final_q_embed = self.embed_tokens(cur_final_q)
                
            cur_s_embed = self.embed_tokens(cur_s)

        
            if questions is not None:
                if final_q is not None:
                    cur_embed = torch.cat([cur_q_embed, cur_v_embed, cur_final_q_embed, cur_s_embed], dim=0)
                    cur_label = cur_label = torch.cat([torch.LongTensor([ignore_idx]*(len(cur_q)+len(cur_v_embed)+len(cur_final_q_embed))).to(device), cur_s], dim=0)
                else:
                    cur_embed = torch.cat([cur_q_embed, cur_v_embed, cur_s_embed], dim=0) # [num_q_tokens + num_v_tokens + num_s_tokens, C]
                    cur_label = torch.cat([torch.LongTensor([ignore_idx]*(len(cur_q)+len(cur_v_embed))).to(device), cur_s], dim=0) # [num_q_tokens + num_v_tokens + num_s_tokens]
            else:
                # add bos token embedding to the seqymces and the bos token to the labels
                cur_v_embed = torch.cat([self.embed_tokens(torch.LongTensor([self.tokenizer.bos_token_id]).to(device)), cur_v_embed], dim=0)
                cur_embed = torch.cat([cur_v_embed, cur_s_embed], dim=0)
                cur_label = torch.cat([torch.LongTensor([ignore_idx]*len(cur_v_embed)).to(device), cur_s], dim=0)

            inputs_embeds.append(cur_embed)
            labels.append(cur_label)

            if len(cur_label) > max_len:
                max_len = len(cur_label)

        pad_embed = self.embed_tokens(torch.LongTensor([self.tokenizer.pad_token_id]).to(device))
        attn_masks = []
        for i in range(x.shape[0]):
            padded_len = max_len - len(labels[i])
            if self.tokenizer.padding_side == "left":
                attn_mask = torch.cat([torch.zeros(1, padded_len), torch.ones(1, len(labels[i]))], dim=1).to(device)
                inputs_embeds[i] = torch.cat([pad_embed.repeat(padded_len, 1), inputs_embeds[i]], dim=0)
                labels[i] = torch.cat([torch.LongTensor([ignore_idx]*padded_len).to(device), labels[i]], dim=0)
            elif self.tokenizer.padding_side == "right":
                attn_mask = torch.cat([torch.ones(1, len(labels[i])), torch.zeros(1, padded_len)], dim=1).to(device)
                inputs_embeds[i] = torch.cat([inputs_embeds[i], pad_embed.repeat(padded_len, 1)], dim=0)
                labels[i] = torch.cat([labels[i], torch.LongTensor([ignore_idx]*padded_len).to(device)], dim=0)
            attn_masks.append(attn_mask)
            assert len(inputs_embeds[i]) == len(labels[i])
        inputs_embeds = torch.stack(inputs_embeds, dim=0)
        attn_masks = torch.cat(attn_masks, dim=0)
        labels = torch.stack(labels, dim=0)
        position_ids = None
        if self.tokenizer.padding_side == "left":
            position_ids = attn_masks.cumsum(-1)-1
            position_ids.masked_fill_(attn_masks == 0, 1)
            position_ids = position_ids.to(device)
            position_ids = position_ids[:, :-1]

        
        return inputs_embeds[:, :-1], attn_masks[:, :-1], labels[:, 1:], position_ids
    
    def _process_predict(self, x, video_masks, subtitles, questions=None, previous_contexts=None, device='cpu', pls=None, background_description = None, rec_prev=None, spottings=None):

        final_q = None

        # Handling rec_prev if applicable
        if self.use_rec_prev and not self.use_gt_prev:
            if len(rec_prev) > 0:
                rec_prev = [". ".join(r) + "." if len(r) > 0 else '' for r in rec_prev]
                previous_contexts = rec_prev
            else:
                previous_contexts = ['' for _ in range(len(subtitles))]

        # Initial prompt setup for questions
        questions = ['You are an AI assistant designed to interpret a video of a sign language signing sequence and translate it into English.' for _ in questions]

        
        # Add previous contexts if applicable
        if previous_contexts is not None and self.use_rec_prev:
            questions = [q + f' The previous context is the following: {c}' if c else q for q, c in zip(questions, previous_contexts)]
        
        if self.use_spottings:
            questions = [q + ' The following are some possible words present in the sentence: ' + ", ".join(s) if len(s) > 0 else "Not available" + '.' for q, s in zip(questions, spottings)]
        
        if self.use_pl_w_feats:
            questions = [q + f' The following are some possible words present in the sentence: {", ".join(pl) if len(pl) > 0 else "Not available"}.' for q, pl in zip(questions, pls)]

        # Add background description if applicable
        if self.bg_desc:
            if self.use_bg_words:
                bg_desc = [get_unique_bg_words(b, drop_sw=self.drop_bg_sw) for b in background_description]
                questions = [q + f' Description of the background is: {", ".join(b) if len(b) > 0 else "Not available"}.' for q, b in zip(questions, bg_desc)]
            else:
                questions = [q + f' Description of the background is: {b}' for q, b in zip(questions, background_description)]
        
        questions = [q + " The following are the video tokens: " for q in questions]
        final_q = [".\nGenerated Sentence: " for _ in questions]


        max_len = 0
        inputs_embeds = []
        for i in range(x.shape[0]):
            cur_v_embed = x[i, :int(video_masks[i].sum())].to(device) # [num_v_tokens, C]

            if questions is not None:
                cur_q = self._tokenize(questions[i], device=device)[:-1] # remove eos token
                cur_q_embed = self.embed_tokens(cur_q)
                

                if final_q is not None:
                    cur_final_q = self._tokenize(final_q[i], device=device)[1:-1]
                    cur_final_q_embed = self.embed_tokens(cur_final_q)

            if questions is not None:
                if final_q is not None:
                    cur_embed = torch.cat([cur_q_embed, cur_v_embed, cur_final_q_embed], dim=0)
                else:
                    cur_embed = torch.cat([cur_q_embed, cur_v_embed], dim=0) # [num_q_tokens + num_v_tokens + num_s_tokens, C]            
            else:
                cur_v_embed = torch.cat([self.embed_tokens(torch.LongTensor([self.tokenizer.bos_token_id]).to(device)), cur_v_embed], dim=0)
                cur_embed = cur_v_embed
            
            inputs_embeds.append(cur_embed)

            if len(cur_embed) > max_len:
                max_len = len(cur_embed)
        
        pad_embed = self.embed_tokens(torch.LongTensor([self.tokenizer.pad_token_id]).to(device))
        attn_masks = []
        for i in range(x.shape[0]):
            padded_len = max_len - len(inputs_embeds[i])
            if self.tokenizer.padding_side == "left":
                attn_mask = torch.cat([torch.zeros(1, padded_len), torch.ones(1, len(inputs_embeds[i]))], dim=1).to(device)
                inputs_embeds[i] = torch.cat([pad_embed.repeat(padded_len, 1), inputs_embeds[i]], dim=0)
            elif self.tokenizer.padding_side == "right":
                attn_mask = torch.cat([torch.ones(1, len(inputs_embeds[i])), torch.zeros(1, padded_len)], dim=1).to(device)
                inputs_embeds[i] = torch.cat([inputs_embeds[i], pad_embed.repeat(padded_len, 1)], dim=0)
            attn_masks.append(attn_mask)

        inputs_embeds = torch.stack(inputs_embeds, dim=0).to(device, dtype=next(self.decoder.parameters()).dtype)
        attn_masks = torch.cat(attn_masks, dim=0).to(device, dtype=next(self.decoder.parameters()).dtype)
        position_ids = None
        if self.tokenizer.padding_side == "left":
            position_ids = attn_masks.cumsum(-1)-1
            position_ids.masked_fill_(attn_masks == 0, 1)
            position_ids = position_ids.to(device)
        

        return inputs_embeds, attn_masks, position_ids

    def forward(self, x, video_masks, subtitles, questions=None, previous_contexts=None, pls=None, background_description=None, rec_prev=None, spottings=None):
   
        outputs, labels, gen_sentences = self._forward(x, video_masks, subtitles, questions, previous_contexts, pls=pls, background_description=background_description, rec_prev=rec_prev, spottings=spottings)    
        return outputs, labels, gen_sentences


    def _forward(self, x, video_masks, subtitles, questions=None, previous_contexts=None, pls=None, background_description=None, rec_prev=None, spottings=None):
        inputs_embeds, attn_masks, labels, position_ids = self._process(x, video_masks, subtitles, questions, previous_contexts, device=x.device, pls=pls, background_description=background_description, spottings=spottings)
        inputs_embeds = self.dropout(inputs_embeds)
        outputs = self.decoder(inputs_embeds=inputs_embeds, attention_mask=attn_masks, position_ids=position_ids, return_dict=True)
        if not self.training:
            gen_sentences = self._predict(x, video_masks, subtitles, questions, previous_contexts, pls=pls, background_description=background_description, rec_prev=rec_prev, spottings=spottings)
        else:
            gen_sentences = None
            
        return outputs, labels, gen_sentences
    
    def _predict(self, x, video_masks, subtitles, questions=None, previous_contexts=None, pls=None, background_description=None, rec_prev=None, spottings=None):
        inputs_embeds, attn_masks, _ = self._process_predict(x, video_masks, subtitles, questions, previous_contexts, device=x.device, pls=pls, background_description=background_description, rec_prev=rec_prev, spottings=spottings)
        outputs = self.decoder.generate(inputs_embeds=inputs_embeds, attention_mask=attn_masks, max_new_tokens=50, pad_token_id=self.tokenizer.eos_token_id, return_dict_in_generate=True, output_scores=True)
        
        return outputs['sequences']