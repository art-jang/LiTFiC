import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import ipdb
from peft import LoraConfig, get_peft_model
import random

from src.utils.score_utils import average_exp_values
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
                 oracle=False,
                 sub_sub = False,
                 lora = False,
                 use_pl_probs = False,
                 use_pl_w_feats = False,
                 mix_in_pls = False,
                 mix_in_pls_prob = 0.5,
                 bg_desc = False,
                 use_bg_w_sub = False,
                 use_rec_prev = False,
                 mix_in_prev_prob = 0.5,
                 mix_in_bg_prob = 0.5,
                 min_prev_conf = 0.0,
                 use_man_gloss = False,
                 use_bg_words = False,
                 drop_bg_sw = False,
                 ret_sent = False,
                 mix_in_ret_prob = 0.0,
                 use_lip_feats = False,
                 mix_in_lip_prob = 0.0,
                 use_prev_pls = False,
                 use_prev_pls_probs = False,
                 mix_in_prev_pls = 0.0,
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
        
        self.oracle = oracle
        self.sub_sub = sub_sub
        self.use_pl_probs = use_pl_probs
        self.use_pl_w_feats = use_pl_w_feats
        self.mix_in_pls = mix_in_pls
        self.mix_in_pls_prob = mix_in_pls_prob
        self.bg_desc = bg_desc
        self.use_bg_w_sub = use_bg_w_sub
        self.use_rec_prev = use_rec_prev
        self.mix_in_prev_prob = mix_in_prev_prob
        self.mix_in_bg_prob = mix_in_bg_prob
        self.min_prev_conf = min_prev_conf

        self.use_man_gloss = use_man_gloss

        self.use_bg_words = use_bg_words
        self.drop_bg_sw = drop_bg_sw

        self.ret_sent = ret_sent
        self.mix_in_ret_prob = mix_in_ret_prob

        self.use_lip_feats = use_lip_feats
        self.mix_in_lip_prob = mix_in_lip_prob

        self.use_prev_pls = use_prev_pls
        self.use_prev_pls_probs = use_prev_pls_probs
        self.mix_in_prev_pls = mix_in_prev_pls

        self.use_gt_prev = use_gt_prev

        self.drop_bgw_pct = drop_bgw_pct
        self.drop_pl_pct = drop_pl_pct

        self.use_spottings = use_spottings
        self.mix_in_spottings = mix_in_spottings

    def _tokenize(self, text, device='cpu'):
        input_ids = self.tokenizer(text, return_tensors='pt')['input_ids'][0].to(device)
        if self.add_eos_token:
            input_ids = torch.cat([input_ids, torch.LongTensor([self.tokenizer.eos_token_id]).to(device)], dim=0)
        return input_ids
    
    def _process(self, x, video_masks, subtitles, questions=None, previous_contexts=None, device='cpu', ignore_idx=-100, pls=None, sub_gt=None, probs=None, background_description=None, man_gloss=None, ret_sent=None, lip_feats=None, lip_masks=None, prev_pls=None, prev_pls_probs=None, spottings=None):

        final_q = None
        inter_q = None
        pl_switch = False

        # Initial prompt setup for questions
        questions = ['You are an AI assistant designed to interpret a video of a sign language signing sequence and translate it into English.' for _ in questions]

        # Process for non-oracle case
        if not self.oracle:
            # Add previous contexts if applicable
            if previous_contexts is not None and self.use_rec_prev and (random.random() <= self.mix_in_prev_prob or not self.training):
                questions = [q + f' The previous context is the following: {c}' if c else q for q, c in zip(questions, previous_contexts)]
            
            if self.use_prev_pls and (random.random() <= self.mix_in_prev_pls or not self.training):
                if self.use_prev_pls_probs:
                    questions = [q + ' The previous possible words with their confidences: ' + ", ".join([f"{word}({prob:.2f})" for word, prob in zip(pl, p)]) if len(pl) > 0 else "Not Available" + '.' for q, pl, p in zip(questions, prev_pls, prev_pls_probs)]
                else:
                    questions = [q + ' The previous possible words are: ' + ", ".join(pl) if len(pl) > 0 else "Not Available" + '.' for q, pl in zip(questions, prev_pls)]


            if self.use_spottings and (random.random() <= self.mix_in_spottings or not self.training):
                questions = [q + ' The following are some possible words present in the sentence: ' + ", ".join(s) if len(s) > 0 else "Not available" + '.' for q, s in zip(questions, spottings)]
            
            # Use manual gloss if applicable
            pls = man_gloss if self.use_man_gloss and man_gloss is not None else pls

            # Incorporate possible words (PL) and probabilities in the format pl_word(pl_prob)
            if self.use_pl_probs and (random.random() <= self.mix_in_pls_prob or not self.training):
                questions = [q + ' The following are the possible words with their confidences: ' + ", ".join([f"{word}({prob:.2f})" for word, prob in zip(pl, p)]) + '.' for q, pl, p in zip(questions, pls, probs)]
            elif self.use_pl_w_feats and (random.random() <= self.mix_in_pls_prob or not self.training):
                questions = [q + f' The following are some possible words present in the sentence: {", ".join(drop_words(pl, self.drop_pl_pct)) if len(pl) > 0 else "Not available"}.' for q, pl in zip(questions, pls)]
                
            # Add background description if applicable
            if self.bg_desc and (random.random() <= self.mix_in_bg_prob or not self.training):
                if self.use_bg_words:
                    bg_desc = [drop_words(get_unique_bg_words(b, drop_sw=self.drop_bg_sw), self.drop_bgw_pct) for b in background_description]
                    questions = [q + f' Description of the background is: {", ".join(b) if len(b) > 0 else "Not available"}.' for q, b in zip(questions, bg_desc)]
                else:
                    questions = [q + f' Description of the background is: {b}' for q, b in zip(questions, background_description)]
                
            
            if self.ret_sent and (random.random() <= self.mix_in_ret_prob or not self.training):
                questions = [q + f' Most similar sentence retrieved from a text_corpus is: {r}' for q, r in zip(questions, ret_sent) if r != ""]


            # Finalize question prompt for video tokens
            if self.use_lip_feats and (random.random() <= self.mix_in_lip_prob or not self.training):
                questions = [q + f' The following are the lip features: ' for q in questions]
                inter_q = [". The following are the video tokens: " for _ in questions]
                final_q = [".\nGenerated Sentence: " for _ in questions]
            
            else:
                # Finalize question prompt for video tokens
                questions = [q + " The following are the video tokens: " for q in questions]
                final_q = [".\nGenerated Sentence: " for _ in questions]

        # Process for oracle case
        else:
            if self.sub_sub:
                if self.training and self.mix_in_pls and random.random() > self.mix_in_pls_prob:
                    pls = sub_gt
                else:
                    self.sub_sub = False
                    pl_switch = True
            else:
                pls = sub_gt

            # Set up system prompt for oracle case
            system_prompt = (
                "You are a helpful assistant designed to generate a sentence based on the list of words entered by the user. "
                "You need to strictly follow these rules:\n"
                "1. The user will only give the list of English words separated by a space, you just need to generate a meaningful sentence from them.\n"
                "2. Only provide a response containing the generated sentence. If you cannot create an English sentence then respond with ”No Translation.\n"
            )
            if self.use_pl_probs and not self.sub_sub:
                system_prompt += "3. The user may provide a list of probabilities for each word. You can use these probabilities to generate the sentence.\n"

            # Apply system prompt to all questions
            questions = [system_prompt for _ in questions]

            # Add previous contexts if applicable
            if previous_contexts is not None and (random.random() <= random.random() * self.mix_in_prev_prob or not self.training) and self.use_rec_prev:
                questions = [q + f' The previous context is the following: {c}' if c else q for q, c in zip(questions, previous_contexts)]

            # Add word list with probabilities in the format pl_word(pl_prob)
            if not self.sub_sub and self.use_pl_probs:
                questions = [q + ' The word list with their confidences: ' + " ".join([f"{word}({prob:.2f})" for word, prob in zip(pl, p)]) + '.' for q, pl, p in zip(questions, pls, probs)]
            else:
                questions = [q + ' The word list is: ' + " ".join(pl) + '.' for q, pl in zip(questions, pls)]
            

            # Add background description if applicable
            if self.bg_desc and (self.sub_sub and self.use_bg_w_sub or random.random() <= random.random() * self.mix_in_bg_prob or not self.training):
                questions = [q + f' Description of the background is: {b}' for q, b in zip(questions, background_description)]
            
            if self.ret_sent and (random.random() <= self.mix_in_ret_prob or not self.training):
                questions = [q + f' Most similar sentence retrieved from a text_corpus is: {r}' for q, r in zip(questions, ret_sent) if r != ""]

            # Finalize oracle prompt for generated sentence
            questions = [q + '\nGenerated Sentence: ' for q in questions]
        

        max_len = 0
        labels = []
        inputs_embeds = []
        for i in range(x.shape[0]):
            cur_v_embed = x[i, :int(video_masks[i].sum())].to(device) # [num_v_tokens, C]
            cur_s = self._tokenize(subtitles[i], device=device)[1:] # remove bos token
            
            if questions is not None:
                cur_q = self._tokenize(questions[i], device=device)[:-1] # remove eos token
                cur_q_embed = self.embed_tokens(cur_q)

                if inter_q is not None:
                    cur_l_embed = lip_feats[i, :int(lip_masks[i].sum())].to(device) # [num_l_tokens, C]
                    cur_inter_q = self._tokenize(inter_q[i], device=device)[1:-1]
                    cur_inter_q_embed = self.embed_tokens(cur_inter_q)

                if final_q is not None:
                    cur_final_q = self._tokenize(final_q[i], device=device)[1:-1]
                    cur_final_q_embed = self.embed_tokens(cur_final_q)
                
            cur_s_embed = self.embed_tokens(cur_s)

            if  questions is not None and self.oracle:
                cur_embed = torch.cat([cur_q_embed, cur_s_embed], dim=0) # [num_q_tokens + num_v_tokens + num_s_tokens, C]
                cur_label = torch.cat([torch.LongTensor([ignore_idx]*(len(cur_q))).to(device), cur_s], dim=0) # [num_q_tokens + num_v_tokens + num_s_tokens]
            elif questions is not None:
                if inter_q is not None:
                    cur_embed = torch.cat([cur_q_embed, cur_l_embed, cur_inter_q_embed, cur_v_embed, cur_final_q_embed], dim=0)
                    cur_label = torch.cat([torch.LongTensor([ignore_idx]*(len(cur_q)+len(cur_l_embed)+len(cur_inter_q_embed) + len(cur_v_embed) + len(cur_final_q_embed))).to(device), cur_s], dim=0)
                elif final_q is not None:
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

        if pl_switch:
            self.sub_sub = True
        
        return inputs_embeds[:, :-1], attn_masks[:, :-1], labels[:, 1:], position_ids
    
    def _process_predict(self, x, video_masks, subtitles, questions=None, previous_contexts=None, device='cpu', ignore_idx=-100, pls=None, sub_gt=None, probs = None, background_description = None, rec_prev=None, rec_prev_conf=None, man_gloss=None, ret_sent=None, lip_feats=None, lip_masks=None, prev_pls=None, prev_pls_probs=None, spottings=None):

        final_q = None
        pl_switch = False
        inter_q = None

        # Handling rec_prev if applicable
        if self.use_rec_prev and not self.sub_sub and not self.use_gt_prev:
            if len(rec_prev) > 0:
                rec_prev = [". ".join(r) + "." if len(r) > 0 and c[0] >= self.min_prev_conf else '' for r, c in zip(rec_prev, rec_prev_conf)]
                previous_contexts = rec_prev
            else:
                previous_contexts = ['' for _ in range(len(subtitles))]

        # Initial prompt setup for questions
        questions = ['You are an AI assistant designed to interpret a video of a sign language signing sequence and translate it into English.' for _ in questions]

        # Process for non-oracle case
        if not self.oracle:
            # Add previous contexts if applicable
            if previous_contexts is not None and self.use_rec_prev:
                questions = [q + f' The previous context is the following: {c}' if c else q for q, c in zip(questions, previous_contexts)]
            
            if self.use_prev_pls:
                if self.use_prev_pls_probs:
                    questions = [q + ' The previous possible words with their confidences: ' + ", ".join([f"{word}({prob:.2f})" for word, prob in zip(pl, p)]) if len(pl) > 0 else "Not Available" + '.' for q, pl, p in zip(questions, prev_pls, prev_pls_probs)]
                else:
                    questions = [q + ' The previous possible words are: ' + ", ".join(pl) if len(pl) > 0 else "Not Available" + '.' for q, pl in zip(questions, prev_pls)]

            if self.use_spottings:
                questions = [q + ' The following are some possible words present in the sentence: ' + ", ".join(s) if len(s) > 0 else "Not available" + '.' for q, s in zip(questions, spottings)]
            
            # Use manual gloss if applicable
            pls = man_gloss if self.use_man_gloss and man_gloss is not None else pls

            # Incorporate possible words (PL) and probabilities in the format pl_word(pl_prob)
            if self.use_pl_probs:
                questions = [q + ' The following are the possible words with their confidences: ' + ", ".join([f"{word}({prob:.2f})" for word, prob in zip(pl, p)]) + '.' for q, pl, p in zip(questions, pls, probs)]
            elif self.use_pl_w_feats:
                questions = [q + f' The following are some possible words present in the sentence: {", ".join(pl) if len(pl) > 0 else "Not available"}.' for q, pl in zip(questions, pls)]

            # Add background description if applicable
            if self.bg_desc:
                if self.use_bg_words:
                    bg_desc = [get_unique_bg_words(b, drop_sw=self.drop_bg_sw) for b in background_description]
                    questions = [q + f' Description of the background is: {", ".join(b) if len(b) > 0 else "Not available"}.' for q, b in zip(questions, bg_desc)]
                else:
                    questions = [q + f' Description of the background is: {b}' for q, b in zip(questions, background_description)]
            
            if self.ret_sent:
                questions = [q + f' Most similar sentence retrieved from a text_corpus is: {r}' for q, r in zip(questions, ret_sent) if r != ""]

            
            if self.use_lip_feats:
                questions = [q + f' The following are the lip features: ' for q in questions]
                inter_q = [". The following are the video tokens: " for _ in questions]
                final_q = [".\nGenerated Sentence: " for _ in questions]
            
            else:
                # Finalize question prompt for video tokens
                questions = [q + " The following are the video tokens: " for q in questions]
                final_q = [".\nGenerated Sentence: " for _ in questions]

        # Process for oracle case
        else:
            if self.sub_sub:
                pls = sub_gt

            # Set up system prompt for oracle case
            system_prompt = (
                "You are a helpful assistant designed to generate a sentence based on the list of words entered by the user. "
                "You need to strictly follow these rules:\n"
                "1. The user will only give the list of English words separated by a space, you just need to generate a meaningful sentence from them.\n"
                "2. Only provide a response containing the generated sentence. If you cannot create an English sentence then respond with ”No Translation.\n"
            )
            if self.use_pl_probs and not self.sub_sub:
                system_prompt += "3. The user may provide a list of probabilities for each word. You can use these probabilities to generate the sentence.\n"

            # Apply system prompt to all questions
            questions = [system_prompt for _ in questions]

            # Add previous contexts if applicable
            if previous_contexts is not None and self.use_rec_prev:
                questions = [q + f' The previous context is the following: {c}' if c else q for q, c in zip(questions, previous_contexts)]

            # Add word list with probabilities in the format pl_word(pl_prob)
            if not self.sub_sub and self.use_pl_probs:
                questions = [q + ' The word list with their confidences: ' + " ".join([f"{word}({prob:.2f})" for word, prob in zip(pl, p)]) + '.' for q, pl, p in zip(questions, pls, probs)]
            else:
                questions = [q + ' The word list is: ' + " ".join(pl) + '.' for q, pl in zip(questions, pls)]

            # Add background description if applicable
            if self.bg_desc:
                questions = [q + f' Description of the background is: {b}' for q, b in zip(questions, background_description)]
            
            if self.ret_sent:
                questions = [q + f' Most similar sentence retrieved from a text_corpus is: {r}' for q, r in zip(questions, ret_sent) if r != ""]

            # Finalize oracle prompt for generated sentence
            questions = [q + '\nGenerated Sentence: ' for q in questions]

    

        max_len = 0
        inputs_embeds = []
        for i in range(x.shape[0]):
            cur_v_embed = x[i, :int(video_masks[i].sum())].to(device) # [num_v_tokens, C]

            if questions is not None:
                cur_q = self._tokenize(questions[i], device=device)[:-1] # remove eos token
                cur_q_embed = self.embed_tokens(cur_q)
                
                if self.use_lip_feats:
                    cur_l_embed = lip_feats[i, :int(lip_masks[i].sum())].to(device) # [num_l_tokens, C]
                    cur_inter_q = self._tokenize(inter_q[i], device=device)[1:-1]
                    cur_inter_q_embed = self.embed_tokens(cur_inter_q)

                if final_q is not None:
                    cur_final_q = self._tokenize(final_q[i], device=device)[1:-1]
                    cur_final_q_embed = self.embed_tokens(cur_final_q)

            if questions is not None:
                if inter_q is not None:
                    cur_embed = torch.cat([cur_q_embed, cur_l_embed, cur_inter_q_embed, cur_v_embed, cur_final_q_embed], dim=0)
                elif final_q is not None:
                    cur_embed = torch.cat([cur_q_embed, cur_v_embed, cur_final_q_embed], dim=0)
                else:
                    cur_embed = torch.cat([cur_q_embed, cur_v_embed], dim=0) # [num_q_tokens + num_v_tokens + num_s_tokens, C]            
            else:
                cur_v_embed = torch.cat([self.embed_tokens(torch.LongTensor([self.tokenizer.bos_token_id]).to(device)), cur_v_embed], dim=0)
                cur_embed = cur_v_embed
            
            if self.oracle:
                cur_embed = cur_q_embed

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

    def forward(self, x, video_masks, subtitles, questions=None, previous_contexts=None, pls=None, sub_gt=None, probs=None, ret = False, background_description=None, rec_prev=None, rec_prev_conf=None, man_gloss=None, ret_sent=None, lip_feats=None, lip_masks=None, prev_pls=None, prev_pls_probs=None, spottings=None):
        outputs_list = []
        gen_sentences_list = None
        labels_list = []
        avg_conf_list = []

        outputs, labels, gen_sentences, avg_conf = self._forward(x, video_masks, subtitles, questions, previous_contexts, pls=pls, sub_gt=sub_gt, probs=probs, ret=ret, background_description=background_description, rec_prev=rec_prev, rec_prev_conf=rec_prev_conf, man_gloss=man_gloss, ret_sent=ret_sent, lip_feats=lip_feats, lip_masks=lip_masks, prev_pls=prev_pls, prev_pls_probs=prev_pls_probs, spottings=spottings)    
        outputs_list.append(outputs)
        labels_list.append(labels)

        if gen_sentences is not None:
            gen_sentences_list = [gen_sentences]
            avg_conf_list.append(avg_conf)
            
        if self.oracle and self.sub_sub and not self.training:
            self.sub_sub = False
            outputs, labels, gen_sentences, avg_conf = self._forward(x, video_masks, subtitles, questions, previous_contexts, pls=pls, sub_gt=sub_gt, probs=probs, ret=ret, background_description=background_description, rec_prev=rec_prev, rec_prev_conf=rec_prev_conf, man_gloss=man_gloss, ret_sent=ret_sent, lip_feats=lip_feats, lip_masks=lip_masks, prev_pls=prev_pls, prev_pls_probs=prev_pls_probs, spottings=spottings)
            self.sub_sub = True
            outputs_list.append(outputs)
            labels_list.append(labels)
            if gen_sentences_list is not None:
                gen_sentences_list.append(gen_sentences)
                avg_conf_list.append(avg_conf)

        
        return outputs_list, labels_list, gen_sentences_list, avg_conf_list


    def _forward(self, x, video_masks, subtitles, questions=None, previous_contexts=None, pls=None, sub_gt=None, probs=None, ret = False, background_description=None, rec_prev=None, rec_prev_conf=None, man_gloss=None, ret_sent=None, lip_feats=None, lip_masks=None, prev_pls=None, prev_pls_probs=None, spottings=None):
        inputs_embeds, attn_masks, labels, position_ids = self._process(x, video_masks, subtitles, questions, previous_contexts, device=x.device, pls=pls, sub_gt=sub_gt, probs=probs, background_description=background_description, man_gloss=man_gloss, ret_sent=ret_sent, lip_feats=lip_feats, lip_masks=lip_masks, prev_pls=prev_pls, prev_pls_probs=prev_pls_probs, spottings=spottings)
        inputs_embeds = self.dropout(inputs_embeds)
        outputs = self.decoder(inputs_embeds=inputs_embeds, attention_mask=attn_masks, position_ids=position_ids, return_dict=True)
        if not self.training and not ret:
            gen_sentences, avg_conf = self._predict(x, video_masks, subtitles, questions, previous_contexts, pls=pls, sub_gt=sub_gt, probs=probs, background_description=background_description, rec_prev=rec_prev, rec_prev_conf=rec_prev_conf, man_gloss=man_gloss, ret_sent=ret_sent, lip_feats=lip_feats, lip_masks=lip_masks, prev_pls=prev_pls, prev_pls_probs=prev_pls_probs, spottings=spottings)
        else:
            gen_sentences = None
            avg_conf = None
            
        return outputs, labels, gen_sentences, avg_conf
    
    def _predict(self, x, video_masks, subtitles, questions=None, previous_contexts=None, pls=None, sub_gt=None, probs=None, background_description=None, rec_prev=None, rec_prev_conf=None, man_gloss=None, ret_sent=None, lip_feats=None, lip_masks=None, prev_pls=None, prev_pls_probs=None, spottings=None):
        inputs_embeds, attn_masks, position_ids = self._process_predict(x, video_masks, subtitles, questions, previous_contexts, device=x.device, pls=pls, sub_gt=sub_gt, probs=probs, background_description=background_description, rec_prev=rec_prev, rec_prev_conf=rec_prev_conf, man_gloss=man_gloss, ret_sent=ret_sent, lip_feats=lip_feats, lip_masks=lip_masks, prev_pls=prev_pls, prev_pls_probs=prev_pls_probs, spottings=spottings)
        outputs = self.decoder.generate(inputs_embeds=inputs_embeds, attention_mask=attn_masks, max_new_tokens=50, pad_token_id=self.tokenizer.eos_token_id, return_dict_in_generate=True, output_scores=True)

        scores = self.decoder.compute_transition_scores(outputs['sequences'], outputs['scores'], normalize_logits=True)
        avg_conf = average_exp_values(scores)
        
        return outputs['sequences'], avg_conf
    
