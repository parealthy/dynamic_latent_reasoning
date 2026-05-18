from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel


class ReasoningNetBase(torch.nn.Module):
    def __init__(self, latent_trajectory_length=128, hidden_size=1024):
        super(ReasoningNetBase, self).__init__()

        self.latent_trajectory = nn.Parameter(
            torch.randn(latent_trajectory_length, hidden_size),
            requires_grad=True
        )

        self.latent_trajectory_length = latent_trajectory_length
        self.hidden_size = hidden_size


class TransformerReasoningNet(ReasoningNetBase, torch.nn.Module):
    def __init__(self, model_name_or_path, latent_trajectory_length=128, hidden_size=1024):
        super(TransformerReasoningNet, self).__init__(latent_trajectory_length, hidden_size)

        self.reasoning_network = AutoModel.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        self.reasoning_network.embed_tokens = None

        if self.reasoning_network.config.hidden_size != self.hidden_size:
            self.transform_layer = torch.nn.Linear(self.hidden_size, self.reasoning_network.config.hidden_size)
            self.reverse_transform_layer = torch.nn.Linear(self.reasoning_network.config.hidden_size, self.hidden_size)
            self.transform_layer.to(self.reasoning_network.dtype)
            self.reverse_transform_layer.to(self.reasoning_network.dtype)
        else:
            self.transform_layer = torch.nn.Identity()
            self.reverse_transform_layer = torch.nn.Identity()

    def forward(self, hidden_states, attention_mask: Optional[torch.Tensor] = None):
        hidden_state_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(self.reasoning_network.dtype)

        latent_trajectory_base = self.latent_trajectory.unsqueeze(0).expand(hidden_states.size(0), -1, -1)
        latent_trajectory_base = latent_trajectory_base.to(self.reasoning_network.dtype)
        last_hidden_state = hidden_states[:, -1:, :]
        latent_trajectory_input = last_hidden_state * latent_trajectory_base

        transformed_hidden_states = self.transform_layer(hidden_states)
        transformed_latent_trajectory = self.transform_layer(latent_trajectory_input)
        reasoning_inputs = torch.cat([transformed_hidden_states, transformed_latent_trajectory], dim=1)

        reasoning_mask = None
        if attention_mask is not None:
            latent_mask = torch.ones(
                hidden_states.size(0),
                self.latent_trajectory_length,
                device=attention_mask.device,
                dtype=attention_mask.dtype,
            )
            reasoning_mask = torch.cat([attention_mask, latent_mask], dim=1)

        outputs = self.reasoning_network(
            inputs_embeds=reasoning_inputs,
            attention_mask=reasoning_mask,
            return_dict=True,
        )
        latent_trajectory_output = outputs.last_hidden_state[:, -self.latent_trajectory_length:, :]
        latent_trajectory = self.reverse_transform_layer(latent_trajectory_output).to(hidden_state_dtype)

        return latent_trajectory


class LatentReasoningModelBase(torch.nn.Module):
    def __init__(self, slow_reasoning_model, processor, reasoning_network, **kwargs):
        super(LatentReasoningModelBase, self).__init__(**kwargs)
        self.slow_reasoning_model = slow_reasoning_model
        self.processor = processor
        self.reasoning_network = reasoning_network
        self.latent_trajectory_length = reasoning_network.latent_trajectory_length

        self.slow_reasoning_model.requires_grad_(False)
        self.reasoning_network.requires_grad_(True)

        self.config = self.slow_reasoning_model.config

    @property
    def get_input_embeddings(self):
        return self.slow_reasoning_model.get_input_embeddings()

    @property
    def tokenizer(self):
        return getattr(self.processor, "tokenizer", self.processor)

    @property
    def pad_token_id(self):
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError("The tokenizer must define either pad_token_id or eos_token_id.")
        return pad_token_id

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.slow_reasoning_model, "gradient_checkpointing_enable"):
            try:
                if gradient_checkpointing_kwargs:
                    self.slow_reasoning_model.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
                    )
                else:
                    self.slow_reasoning_model.gradient_checkpointing_enable()
            except TypeError:
                self.slow_reasoning_model.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        if hasattr(self.slow_reasoning_model, "gradient_checkpointing_disable"):
            self.slow_reasoning_model.gradient_checkpointing_disable()


class LatentTransformerReasoningModel(LatentReasoningModelBase):
    def __init__(self, slow_reasoning_model, processor, reasoning_network, **kwargs):
        super(LatentTransformerReasoningModel, self).__init__(
            slow_reasoning_model,
            processor,
            reasoning_network,
            **kwargs
        )

    def _prefill_prompt(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        prompt_embeddings = self.get_input_embeddings(input_ids).to(self.slow_reasoning_model.dtype)

        output = self.slow_reasoning_model(
            inputs_embeds=prompt_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
            output_hidden_states=True,
            **kwargs,
        )

        prompt_hidden_states = output.hidden_states[-1].detach()
        return prompt_embeddings, prompt_hidden_states

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        if attention_mask is None:
            attention_mask = torch.ones(
                (input_ids.size(0), input_ids.size(1) + labels.size(1)),
                device=input_ids.device,
                dtype=torch.long,
            )

        with torch.no_grad():
            prompt_mask = attention_mask[:, :input_ids.size(1)]
            prompt_embeddings, prompt_hidden_states = self._prefill_prompt(
                input_ids=input_ids,
                attention_mask=prompt_mask,
                position_ids=position_ids,
                **kwargs,
            )

        latent_trajectory = self.reasoning_network(prompt_hidden_states, attention_mask=prompt_mask)
        latent_trajectory_mask = torch.ones(latent_trajectory.size(0), latent_trajectory.size(1)).to(input_ids.device).long()

        label_embeddings = self.get_input_embeddings(labels).to(self.slow_reasoning_model.dtype)
        labels_mask = attention_mask[:, input_ids.size(1):]

        input_embeddings = torch.cat([prompt_embeddings, latent_trajectory, label_embeddings], dim=1)
        input_mask = torch.cat([prompt_mask, latent_trajectory_mask, labels_mask], dim=1).long()

        labels = labels.long()
        labels = labels.masked_fill(labels_mask.to(labels.device) == 0, -100)
        labels = torch.cat((
            prompt_embeddings.new_ones(labels.size(0), prompt_embeddings.size(1)).long()*-100,
            latent_trajectory.new_ones(labels.size(0), latent_trajectory.size(1)).long()*-100,
            labels,), dim=1
        ).long()

        outputs = self.slow_reasoning_model(
            inputs_embeds=input_embeddings,
            attention_mask=input_mask,
            labels=labels,
            return_dict=True,
            **kwargs,
        )

        return outputs

    def generate(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        generation_config=None,
        **kwargs,
    ):
        if attention_mask is None:
            attention_mask = torch.ones(
                (input_ids.size(0), input_ids.size(1)),
                device=input_ids.device,
                dtype=torch.long,
            )

        prompt_mask = attention_mask.long()
        generation_kwargs = dict(kwargs)
        generation_kwargs.setdefault("use_cache", True)

        with torch.no_grad():
            prompt_embeddings, prompt_hidden_states = self._prefill_prompt(
                input_ids=input_ids,
                attention_mask=prompt_mask,
                position_ids=position_ids,
            )

            latent_trajectory = self.reasoning_network(
                prompt_hidden_states,
                attention_mask=prompt_mask,
            ).to(prompt_embeddings.dtype)
            latent_trajectory_mask = torch.ones(
                latent_trajectory.size(0),
                latent_trajectory.size(1),
                device=prompt_mask.device,
                dtype=prompt_mask.dtype,
            )

            input_embeddings = torch.cat([prompt_embeddings, latent_trajectory], dim=1)
            input_embeddings = input_embeddings.to(self.slow_reasoning_model.dtype)
            input_mask = torch.cat([prompt_mask, latent_trajectory_mask], dim=1).long()

            outputs = self.slow_reasoning_model.generate(
                inputs_embeds=input_embeddings,
                attention_mask=input_mask,
                generation_config=generation_config,
                **generation_kwargs,
            )

        return outputs
