import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import ipdb
from peft import LoraConfig, get_peft_model
import random


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
        self.add_eos_token = True if os.path.basename(pretrained_llm) in ['Meta-Llama-3-8B', 'Meta-Llama-3-8B-Instruct'] else False

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.decoder = AutoModelForCausalLM.from_pretrained(pretrained_llm, 
                                                            torch_dtype=torch_dtype,
                                                            **decoder_config)
        self.torch_dtype = torch_dtype

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

    def _tokenize(self, text, device='cpu'):
        input_ids = self.tokenizer(text, return_tensors='pt')['input_ids'][0].to(device)
        if self.add_eos_token:
            input_ids = torch.cat([input_ids, torch.LongTensor([self.tokenizer.eos_token_id]).to(device)], dim=0)
        return input_ids
    
    def _process(self, x, video_masks, subtitles, questions=None, previous_contexts=None, device='cpu', ignore_idx=-100, pls=None, sub_gt=None, probs=None, background_description=None):

        final_q = None
        pl_switch = False
        if not self.oracle:
            
            if previous_contexts is not None:
                questions = [q + ' The previous context is the following: ' + c + ' And the given word list is as follows: ' if c != '' \
                            else q for q, c in zip(questions, previous_contexts)]
            
            if self.use_pl_w_feats:
                questions = [q + 'The following are some possible words present in the sentence: ' + " ".join(pl) for q, pl in zip(questions, pls)]
                if self.use_pl_probs:
                    questions = [q + 'The confidences for the previous words are: ' + ", ".join([f"{p:.2f}" for p in pl]) for q, pl in zip(questions, probs)]
            
            if self.bg_desc:
                questions = [q + '. Description of the background is: ' + b for q, b in zip(questions, background_description)]
            
            questions = [q + ". The following are the video tokens: " for q in questions]
            final_q = ["\nGenerated Sentence: " for _ in questions]
        
        else:
            if self.sub_sub:
                if self.training and self.mix_in_pls:
                    rand = random.random()
                    if rand > self.mix_in_pls_prob:
                        pls = sub_gt
                    else: 
                        self.sub_sub = False
                        pl_switch = True
                else:
                    pls = sub_gt

            system_prompt = "You are a helpful assistant designed to generate a sentence based on the list of words entered by the user. You need to strictly follow these rules:\n" + \
                            "1. The user will only give the list of English words separated by a space, you just need to generate a meaningful sentence from them.\n" + \
                            "2. Only provide a response containing the generated sentence. If you cannot create an English sentence then respond with ”No Translation.\n"
            if self.use_pl_probs and not self.sub_sub:
                system_prompt +=  "3. The user may provide a list probabilities for each word. You can use these probabilities to generate the sentence.\n"

            questions = [system_prompt for q in questions]

            if previous_contexts is not None:
                questions = [q + 'The previous context is the following: ' + c for q, c in zip(questions, previous_contexts)]

            questions = [q + 'The word list is: ' + " ".join(pl) for q, pl in zip(questions, pls)]
        
            if self.use_pl_probs and not self.sub_sub:
                questions = [q + '. The probabilities are: ' + ", ".join([f"{p:.2f}" for p in pl]) for q, pl in zip(questions, probs)]

            if self.bg_desc:
                if self.sub_sub:
                    if self.use_bg_w_sub:
                        questions = [q + '. Description of the background is: ' + b for q, b in zip(questions, background_description)]
                else:
                    questions = [q + '. Description of the background is: ' + b for q, b in zip(questions, background_description)]
            
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

                if final_q is not None:
                    cur_final_q = self._tokenize(final_q[i], device=device)[:-1]
                    cur_final_q_embed = self.embed_tokens(cur_final_q)
                
            cur_s_embed = self.embed_tokens(cur_s)

            if  questions is not None and self.oracle:
                cur_embed = torch.cat([cur_q_embed, cur_s_embed], dim=0) # [num_q_tokens + num_v_tokens + num_s_tokens, C]
                cur_label = torch.cat([torch.LongTensor([ignore_idx]*(len(cur_q))).to(device), cur_s], dim=0) # [num_q_tokens + num_v_tokens + num_s_tokens]
            elif questions is not None:
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

        if pl_switch:
            self.sub_sub = True
        
        return inputs_embeds[:, :-1], attn_masks[:, :-1], labels[:, 1:], position_ids
    
    def _process_predict(self, x, video_masks, subtitles, questions=None, previous_contexts=None, device='cpu', ignore_idx=-100, pls=None, sub_gt=None, probs = None, background_description = None):

        final_q = None
        if not self.oracle:

            if previous_contexts is not None:
                questions = [q + ' The previous context is the following: ' + c + ' And the given word list is as follows: ' if c != '' \
                            else q for q, c in zip(questions, previous_contexts)]
            
            if self.use_pl_w_feats:
                questions = [q + 'The following are some possible words present in the sentence: ' + ", ".join(pl) for q, pl in zip(questions, pls)]
                if self.use_pl_probs:
                    questions = [q + '. The confidences for the previous words are: ' + ", ".join([f"{p:.2f}" for p in pl]) for q, pl in zip(questions, probs)]
            
            if self.bg_desc:
                questions = [q + '. Description of the background is: ' + b for q, b in zip(questions, background_description)]
            
            questions = [q + ". The following are the video tokens: " for q in questions]
            final_q = ["\nGenerated Sentence: " for _ in questions]
        
        else:
            if self.sub_sub:
                pls = sub_gt
                
            system_prompt = "You are a helpful assistant designed to generate a sentence based on the list of words entered by the user. You need to strictly follow these rules:\n" + \
                            "1. The user will only give the list of English words separated by a space, you just need to generate a meaningful sentence from them.\n" + \
                            "2. Only provide a response containing the generated sentence. If you cannot create an English sentence then respond with ”No Translation.\n"
            if self.use_pl_probs and not self.sub_sub:
                system_prompt +=  "3. The user may provide a list probabilities for each word. You can use these probabilities to generate the sentence.\n"

            questions = [system_prompt for q in questions]

            if previous_contexts is not None:
                questions = [q + ' The previous context is the following: ' + c for q, c in zip(questions, previous_contexts)]
        
            questions = [q + 'The word list is: ' + " ".join(pl) for q, pl in zip(questions, pls)]
        
            if self.use_pl_probs and not self.sub_sub:
                questions = [q + 'The probabilities are: ' + ", ".join([f"{p:.2f}" for p in pl]) for q, pl in zip(questions, probs)]
            
            if self.bg_desc:
                if self.sub_sub:
                    if self.use_bg_w_sub:
                        questions = [q + '. Description of the background is: ' + b for q, b in zip(questions, background_description)]
                else:
                    questions = [q + '. Description of the background is: ' + b for q, b in zip(questions, background_description)]
            
            questions = [q + '\nGenerated Sentence: ' for q in questions]
        

        max_len = 0
        inputs_embeds = []
        for i in range(x.shape[0]):
            cur_v_embed = x[i, :int(video_masks[i].sum())].to(device) # [num_v_tokens, C]

            if questions is not None:
                cur_q = self._tokenize(questions[i], device=device)[:-1] # remove eos token
                cur_q_embed = self.embed_tokens(cur_q)

                if final_q is not None:
                    cur_final_q = self._tokenize(final_q[i], device=device)[:-1]
                    cur_final_q_embed = self.embed_tokens(cur_final_q)

            if questions is not None:
                if final_q is not None:
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

    def forward(self, x, video_masks, subtitles, questions=None, previous_contexts=None, pls=None, sub_gt=None, probs=None, ret = False, background_description=None):
        outputs_list = []
        gen_sentences_list = None
        labels_list = []

        outputs, labels, gen_sentences = self._forward(x, video_masks, subtitles, questions, previous_contexts, pls=pls, sub_gt=sub_gt, probs=probs, ret=ret, background_description=background_description)    
        outputs_list.append(outputs)
        labels_list.append(labels)

        if gen_sentences is not None:
            gen_sentences_list = [gen_sentences]
            
        if self.oracle and self.sub_sub and not self.training:
            self.sub_sub = False
            outputs, labels, gen_sentences = self._forward(x, video_masks, subtitles, questions, previous_contexts, pls=pls, sub_gt=sub_gt, probs=probs, ret=ret, background_description=background_description)
            self.sub_sub = True
            outputs_list.append(outputs)
            labels_list.append(labels)
            if gen_sentences_list is not None:
                gen_sentences_list.append(gen_sentences)

        
        return outputs_list, labels_list, gen_sentences_list


    def _forward(self, x, video_masks, subtitles, questions=None, previous_contexts=None, pls=None, sub_gt=None, probs=None, ret = False, background_description=None):
        inputs_embeds, attn_masks, labels, position_ids = self._process(x, video_masks, subtitles, questions, previous_contexts, device=x.device, pls=pls, sub_gt=sub_gt, probs=probs, background_description=background_description)
        outputs = self.decoder(inputs_embeds=inputs_embeds, attention_mask=attn_masks, position_ids=position_ids, return_dict=True)
        if not self.training and not ret:
            gen_sentences = self._predict(x, video_masks, subtitles, questions, previous_contexts, pls=pls, sub_gt=sub_gt, probs=probs, background_description=background_description)
        else:
            gen_sentences = None
            
        return outputs, labels, gen_sentences
    
    def _predict(self, x, video_masks, subtitles, questions=None, previous_contexts=None, pls=None, sub_gt=None, probs=None, background_description=None):
        inputs_embeds, attn_masks, position_ids = self._process_predict(x, video_masks, subtitles, questions, previous_contexts, device=x.device, pls=pls, sub_gt=sub_gt, probs=probs, background_description=background_description)
        outputs = self.decoder.generate(inputs_embeds=inputs_embeds, attention_mask=attn_masks, max_new_tokens=50, pad_token_id=self.tokenizer.eos_token_id)
        
        return outputs
    
