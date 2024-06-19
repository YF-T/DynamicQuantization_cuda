import torch
import torch.nn.functional as F
import math
from ..quant.new_pack import triton_quantize_and_pack_along_last_dim
from ..quant.matmul import cuda_bmm_fA_qB_outer

class attention_kivi(torch.nn.Module):
    def __init__(self, bsz: int, max_seq_len: int, n_local_kv_heads: int, head_dim: int, device: str = "cuda:0", gemm_cuda=True, bits: int = 4, group_size: int = 64, residual_length: int = 128):
        super().__init__()
        self.bsz = bsz
        self.max_seq_len = max_seq_len
        self.n_local_kv_heads = n_local_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.gemm_cuda = gemm_cuda
        self.past_key_value = None

        self.hidden_size = head_dim
        self.num_heads = n_local_kv_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = n_local_kv_heads
        self.num_key_value_groups = 1
        self.k_bits = bits
        self.v_bits = bits
        self.group_size = group_size
        self.residual_length = residual_length

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, start_pos: int, seqlen: int, mask: torch.Tensor) -> torch.Tensor:
        # input: q.shape = (bsz, seqlen, n_local_kv_heads, head_dim)
        # input: k.shape = (bsz, seqlen, n_local_kv_heads, head_dim)
        # input: v.shape = (bsz, seqlen, n_local_kv_heads, head_dim)
        # output: output.shape = (bsz, seqlen, n_local_kv_heads, head_dim)
        # self.cache_k[:, start_pos : start_pos + seqlen] = k
        # self.cache_v[:, start_pos : start_pos + seqlen] = v
        #
        # keys = self.cache_k[:, : start_pos + seqlen]
        # values = self.cache_v[:, : start_pos + seqlen]
        #
        # q = q.transpose(1, 2) # (bs, n_local_heads, seqlen, head_dim)
        # keys = keys.transpose(1, 2) # (bs, n_local_heads, cache_len + seqlen, head_dim)
        # values = values.transpose(1, 2) # (bs, n_local_heads, cache_len + seqlen, head_dim)
        #
        # if mask is not None:
        #     scores = torch.matmul(q, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        #     scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        #     scores = F.softmax(scores.float(), dim=-1).type_as(q)
        #     output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        # elif self.gemm_cuda:
        #     scores = self.gemm(q, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        #     scores = F.softmax(scores.float(), dim=-1).type_as(q)
        #     output = self.gemm(scores, values)
        # else:
        #     scores = torch.matmul(q, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        #     scores = F.softmax(scores.float(), dim=-1).type_as(q)
        #     output = torch.matmul(scores, values)
        #
        # output = output.transpose(1, 2).contiguous()
        # return output
        # print(self.layer_idx,key_states.shape,value_states.shape)
        key_states = k.transpose(1, 2)
        value_states = v.transpose(1, 2)
        query_states = q.transpose(1, 2)
        past_key_value = self.past_key_value
        if past_key_value is not None:
            key_states_quant_trans = past_key_value[0]
            key_states_full = past_key_value[1]
            key_scale_trans = past_key_value[2]
            key_mn_trans = past_key_value[3]
            value_states_quant = past_key_value[4]
            value_states_full = past_key_value[5]
            value_scale = past_key_value[6]
            value_mn = past_key_value[7]

            if key_states_quant_trans is not None:
                att_qkquant = cuda_bmm_fA_qB_outer(self.group_size, query_states, key_states_quant_trans,
                                                   key_scale_trans, key_mn_trans, self.k_bits)
            else:
                att_qkquant = None

            if key_states_full is not None:
                key_states_full = torch.cat([key_states_full, key_states], dim=2)
            else:
                key_states_full = key_states
            att_qkfull = torch.matmul(query_states, key_states_full.transpose(2, 3))
            if att_qkquant is not None:
                attn_weights = torch.cat([att_qkquant, att_qkfull], dim=-1) / math.sqrt(self.head_dim)
            else:
                attn_weights = att_qkfull / math.sqrt(self.head_dim)

            if key_states_full.shape[-2] == self.residual_length:
                assert self.residual_length % self.group_size == 0
                key_states_quant_trans_new, key_scale_trans_new, key_mn_trans_new = triton_quantize_and_pack_along_last_dim(key_states_full.transpose(2, 3).contiguous(),
                                                                                                                            self.group_size,
                                                                                                                            self.k_bits)
                key_states_full = None
                if key_states_quant_trans is not None:
                    key_states_quant_trans = torch.cat([key_states_quant_trans, key_states_quant_trans_new], dim=3)
                    key_scale_trans = torch.cat([key_scale_trans, key_scale_trans_new], dim=3)
                    key_mn_trans = torch.cat([key_mn_trans, key_mn_trans_new], dim=3)
                else:
                    key_states_quant_trans = key_states_quant_trans_new
                    key_scale_trans = key_scale_trans_new
                    key_mn_trans = key_mn_trans_new

            if mask is not None:
                attn_weights = attn_weights + mask
                attn_weights = torch.max(
                    attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
                )

            # upcast attention to fp32
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

            value_states_full = torch.cat([value_states_full, value_states], dim=2)
            value_full_length = value_states_full.shape[-2]
            if value_states_quant is None:
                attn_output = torch.matmul(attn_weights, value_states_full)
            else:
                attn_output = cuda_bmm_fA_qB_outer(self.group_size, attn_weights[:, :, :, :-value_full_length], value_states_quant,
                                                   value_scale, value_mn, self.v_bits)
                attn_output += torch.matmul(attn_weights[:, :, :, -value_full_length:], value_states_full)

            if value_full_length > self.residual_length:
                assert value_full_length == self.residual_length + 1
                value_states_quant_new, scale, mn = triton_quantize_and_pack_along_last_dim(value_states_full[:, :, :1, :].contiguous(),
                                                                                            self.group_size,
                                                                                            self.v_bits)
                value_states_full = value_states_full[:, :, 1:, :].contiguous()
                if value_states_quant is not None:
                    value_states_quant = torch.cat([value_states_quant, value_states_quant_new], dim=2)
                    value_scale = torch.cat([value_scale, scale], dim=2)
                    value_mn = torch.cat([value_mn, mn], dim=2)
                else:
                    value_states_quant = value_states_quant_new
                    value_scale = scale
                    value_mn = mn

        else:
            attn_weights = torch.matmul(query_states,
                                        key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            # quantize
            if key_states.shape[-2] % self.residual_length != 0:
                if key_states.shape[-2] < self.residual_length:
                    key_states_quant = None
                    key_states_full = key_states
                else:
                    key_states_quant = key_states[:, :, :-(key_states.shape[-2] % self.residual_length), :].contiguous()
                    key_states_full = key_states[:, :, -(key_states.shape[-2] % self.residual_length):, :].contiguous()
            else:
                key_states_quant = key_states
                key_states_full = None
            if key_states_quant is not None:
                key_states_quant_trans, key_scale_trans, key_mn_trans = triton_quantize_and_pack_along_last_dim(key_states_quant.transpose(2, 3).contiguous(), self.group_size, self.k_bits)
            else:
                key_states_quant_trans = None
                key_scale_trans = None
                key_mn_trans = None

            if value_states.shape[-2] <= self.residual_length:
                value_states_quant = None
                value_states_full = value_states
                value_scale = None
                value_mn = None
            else:
                value_states_quant = value_states[:, :, :-self.residual_length, :].contiguous()
                value_states_full = value_states[:, :, -self.residual_length:, :].contiguous()
                value_states_quant, value_scale, value_mn = triton_quantize_and_pack_along_last_dim(value_states_quant,
                                                                                                    self.group_size,
                                                                                                    self.v_bits)

            if mask is not None:
                attn_weights = attn_weights + mask
                attn_weights = torch.max(
                    attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
                )

            # upcast attention to fp32
            attn_weights = F.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(query_states.dtype)

            attn_output = torch.matmul(attn_weights, value_states)
        past_key_value = (key_states_quant_trans, key_states_full, key_scale_trans, key_mn_trans, value_states_quant, value_states_full, value_scale, value_mn, start_pos+seqlen)
        self.past_key_value = past_key_value

        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output

    def prefill(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seqlen: int, mask: torch.Tensor) -> torch.Tensor:
        return self.forward(q, k, v, 0, seqlen, mask)

    def decoding(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, start_pos: int, seqlen: int) -> torch.Tensor:
        return self.forward(q, k, v, start_pos, seqlen, None)