import torch
import torch.nn.functional as F
import math
from .CompressUtils import CompressUnion

class attention_gear(torch.nn.Module):
    def __init__(self, bsz: int, max_seq_len: int, n_local_kv_heads: int, head_dim: int, device: str = "cuda:0", gemm_cuda=True):
        super().__init__()
        self.bsz = bsz
        self.max_seq_len = max_seq_len
        self.n_local_kv_heads = n_local_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.gemm_cuda = gemm_cuda
        self.past_key_value = None
        compress_config = {}
        compress_config["compress_mode"] = "gear_batch" # batchwise-GEAR
        compress_config["quantize_bit"] = 12 # outlier quantization bit
        compress_config["left"] = 0.02 # outlier extraction rate
        compress_config["rank"] = 0.02  # setting rank for Key and value cache quantization error
        compress_config["loop"] = 3 # a constant for power iteration(an efficient SVD solver)
        compress_config["stream"] = True # streaming-gear set to true to perform better efficiency
        compress_config["streaming_gap"] = 20 # re-compress every 20 iteration
        self.compress_config = compress_config

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
            # reuse k, v, self_attention
            if self.compress_config is not None:
                prev_keys = past_key_value[0].decompress()
                prev_values = past_key_value[1].decompress()

                key_states = torch.cat([prev_keys, key_states], dim=2)
                value_states = torch.cat([prev_values, value_states], dim=2)
                # if self.layer_idx == 0:
                #     print(prev_keys.shape,prev_values.shape,key_states.shape,value_states.shape)
            else:
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if mask is not None:
            attn_weights = attn_weights + mask
        # if self.layer_idx == 0:
        #     print("peak_usage:",torch.cuda.max_memory_allocated(device="cuda") / (1024**2))
        # upcast attention to fp32
        attn_weights = F.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        # detect_infnan(attn_weights,"attn weights2")
        attn_output = torch.matmul(attn_weights, value_states)

        # if self.layer_idx == 0:
        #     print("peak_usage:",torch.cuda.max_memory_allocated(device="cuda") / (1024**2))
        #### compress V
        use_cache = True
        if use_cache:
            if self.compress_config is not None:
                if past_key_value is None:
                    past_key_union = CompressUnion(compress_kwargs=self.compress_config)
                    past_vaule_union = CompressUnion(
                        compress_kwargs=self.compress_config
                    )
                    past_key_union.compress(key_states)
                    past_vaule_union.compress(value_states)
                    del key_states
                    del value_states
                    past_key_value = (past_key_union, past_vaule_union)
                else:
                    past_key_union, past_vaule_union = past_key_value
                    past_key_union.compress(key_states)
                    past_vaule_union.compress(value_states)
                    del key_states
                    del value_states
                    past_key_value = (past_key_union, past_vaule_union)

            else:
                past_key_value = (key_states, value_states)
        else:
            past_key_value = None
        self.past_key_value = past_key_value

        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output

    def prefill(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seqlen: int, mask: torch.Tensor) -> torch.Tensor:
        return self.forward(q, k, v, 0, seqlen, mask)

    def decoding(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, start_pos: int, seqlen: int) -> torch.Tensor:
        return self.forward(q, k, v, start_pos, seqlen, None)