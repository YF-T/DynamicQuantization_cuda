from k_cache_file import K_Cache_Class
from v_cache_file import V_Cache_Class
from KV_cache_cpp_extention import cuda_module
import torch
import torch.nn.functional as F
import os
import numpy as np
import math
import time

base_path = "/u/user75/llama2/llama"
q_path = os.path.join(base_path, "q_caches.pt")
score_path = os.path.join(base_path, "score_caches.pt")
kv_path = os.path.join(base_path, "kv_caches.pt")

q_caches = torch.load(q_path, map_location=torch.device('cpu')) # (layer, args.max_batch_size,args.max_seq_len,self.n_local_kv_heads,self.head_dim,)
kv_caches = torch.load(kv_path, map_location=torch.device('cpu')) # (2, layer, args.max_batch_size,args.max_seq_len,self.n_local_kv_heads,self.head_dim,)

class attention(torch.nn.Module):
    def __init__(self, bsz: int, max_seq_len: int, n_local_kv_heads: int, head_dim: int, device: str = "cuda:0", reference: bool = False):
        super().__init__()
        self.bsz = bsz
        self.max_seq_len = max_seq_len
        self.n_local_kv_heads = n_local_kv_heads
        self.head_dim = head_dim
        self.n_max_blocks = math.ceil(max_seq_len / 64)
        self.k_cache_model = K_Cache_Class(bsz, max_seq_len, n_local_kv_heads, head_dim, device=device)
        self.v_cache_model = V_Cache_Class(bsz, max_seq_len, n_local_kv_heads, head_dim, device=device)
        self.device = device
        self.reference = reference

    def prefill_k(self, q: torch.Tensor, k: torch.Tensor, seqlen: int) -> torch.Tensor:
        # input: q.shape = (bsz, seqlen, n_local_kv_heads, head_dim)
        # input: k.shape = (bsz, seqlen, n_local_kv_heads, head_dim)
        # output: score.shape = (bsz, n_local_kv_heads, seqlen, seqlen)
        self.k_cache_model.save(k, 0, seqlen)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        return scores

    def prefill_v(self, scores: torch.Tensor, v: torch.Tensor, seqlen: int) -> torch.Tensor:
        # input: scores.shape = (bsz, n_local_kv_heads, seqlen, seqlen)
        # input: v.shape = (bsz, seqlen, n_local_kv_heads, head_dim)
        # output: output.shape = (bsz, seqlen, n_local_kv_heads, head_dim)
        self.v_cache_model.save(v, 0, seqlen)
        v = v.transpose(1, 2)
        output = torch.matmul(scores, v)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous()
        return output

    def prefill(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seqlen: int, mask: torch.Tensor) -> torch.Tensor:
        scores = self.prefill_k(q, k, seqlen)
        if mask is not None:
            scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(q)
        output = self.prefill_v(scores, v, seqlen)
        return output

    def decoding_k(self, q: torch.Tensor, k: torch.Tensor, start_pos: int, seqlen: int) -> torch.Tensor:
        # input: q.shape = (bsz, seqlen, n_local_kv_heads, head_dim)
        # input: k.shape = (bsz, seqlen, n_local_kv_heads, head_dim)
        # output: score.shape = (bsz, n_local_kv_heads, seqlen, seqlen)
        self.k_cache_model.save(k, start_pos, seqlen)
        return self.k_cache_model.decoding_compute(q, start_pos, seqlen, self.reference)

    def decoding_v(self, scores: torch.Tensor, v: torch.Tensor, start_pos: int, seqlen: int) -> torch.Tensor:
        # input: scores.shape = (bsz, n_local_kv_heads, seqlen, seqlen)
        # input: v.shape = (bsz, seqlen, n_local_kv_heads, head_dim)
        # output: output.shape = (bsz, seqlen, n_local_kv_heads, head_dim)
        self.v_cache_model.save(v, start_pos, seqlen)
        return self.v_cache_model.decoding_compute(scores, start_pos, seqlen, self.reference)

    def decoding(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, start_pos: int, seqlen: int) -> torch.Tensor:
        scores = self.decoding_k(q, k, start_pos, seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(q)
        output = self.decoding_v(scores, v, start_pos, seqlen)
        return output

class attention_ref(torch.nn.Module):
    def __init__(self, bsz: int, max_seq_len: int, n_local_kv_heads: int, head_dim: int, device: str = "cuda:0", gemm_cuda=True):
        super().__init__()
        self.bsz = bsz
        self.max_seq_len = max_seq_len
        self.n_local_kv_heads = n_local_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.gemm_cuda = gemm_cuda
        self.cache_k = torch.zeros(
            (
                self.bsz,
                self.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            ), device=self.device, dtype=torch.half
        )
        self.cache_v = torch.zeros(
            (
                self.bsz,
                self.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            ), device=self.device, dtype=torch.half
        )

    def gemm(self, a, b):
        """
        torch::Tensor gemm_cuda(torch::Tensor A,
                        torch::Tensor B,
                        const int stride_batch_A,
                        const int stride_n_heads_A,
                        const int stride_l_A,
                        const int stride_m_A,
                        const int stride_batch_B,
                        const int stride_n_heads_B,
                        const int stride_m_B,
                        const int stride_n_B,
                        const int l,
                        const int m,
                        const int n)
        """
        a = a.contiguous()
        b = b.contiguous()
        flag = False
        if b.size(3) % 2 == 1:
            flag = True
            b = F.pad(b, (0, 1), "constant", 0)
        c = cuda_module.gemm_cuda(a.view(torch.uint8), b.view(torch.uint8), a.stride(0), a.stride(1), a.stride(2), a.stride(3), b.stride(0), b.stride(1), b.stride(2), b.stride(3), a.size(2), a.size(3), b.size(3))
        c = c.view(torch.half)
        if flag:
            c = c[:, :, :, :-1]
        return c

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, start_pos: int, seqlen: int, mask: torch.Tensor) -> torch.Tensor:
        # input: q.shape = (bsz, seqlen, n_local_kv_heads, head_dim)
        # input: k.shape = (bsz, seqlen, n_local_kv_heads, head_dim)
        # input: v.shape = (bsz, seqlen, n_local_kv_heads, head_dim)
        # output: output.shape = (bsz, seqlen, n_local_kv_heads, head_dim)
        self.cache_k[:, start_pos : start_pos + seqlen] = k
        self.cache_v[:, start_pos : start_pos + seqlen] = v

        keys = self.cache_k[:, : start_pos + seqlen]
        values = self.cache_v[:, : start_pos + seqlen]

        q = q.transpose(1, 2) # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2) # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(1, 2) # (bs, n_local_heads, cache_len + seqlen, head_dim)

        if mask is not None:
            scores = torch.matmul(q, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
            scores = F.softmax(scores.float(), dim=-1).type_as(q)
            output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        elif self.gemm_cuda:
            scores = self.gemm(q, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
            scores = F.softmax(scores.float(), dim=-1).type_as(q)
            output = self.gemm(scores, values)
        else:
            scores = torch.matmul(q, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
            scores = F.softmax(scores.float(), dim=-1).type_as(q)
            output = torch.matmul(scores, values)

        output = output.transpose(1, 2).contiguous()
        return output

    def prefill(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seqlen: int, mask: torch.Tensor) -> torch.Tensor:
        return self.forward(q, k, v, 0, seqlen, mask)

    def decoding(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, start_pos: int, seqlen: int) -> torch.Tensor:
        return self.forward(q, k, v, start_pos, seqlen, None)


# 生成：128-4000
bsz = 4
max_seq_len = 4096
n_local_kv_heads = 32
head_dim = 128
layer_num = 32
device = "cuda:0"
q_caches = q_caches.to(device)
kv_caches = kv_caches.to(device)
print("q_caches: ", q_caches.shape)
print("kv_caches: ", kv_caches.shape)

def test(prefill_len: int, gen_len: int, model: list):
    start_time = time.time()
    time_dict = {}

    # prefill
    mask = torch.full((prefill_len, prefill_len), float('-inf'), device=device)
    mask = torch.triu(mask, diagonal=1).view(1, 1, prefill_len, prefill_len)
    for i, attention_block in enumerate(model):
        q = q_caches[i, :, 0:prefill_len, :, :]
        q = torch.repeat_interleave(q, bsz // q.shape[0], dim=0)
        k = kv_caches[0, i, :, 0:prefill_len, :, :]
        k = torch.repeat_interleave(k, bsz // k.shape[0], dim=0)
        v = kv_caches[1, i, :, 0:prefill_len, :, :]
        v = torch.repeat_interleave(v, bsz // v.shape[0], dim=0)
        o = attention_block.prefill(q, k, v, prefill_len, mask)

    # decoding
    for start_pos in range(prefill_len, prefill_len + gen_len):
        if start_pos % 50 == 0:
            print("start_pos: ", start_pos)
            time_dict[start_pos] = time.time() - start_time
        for i, attention_block in enumerate(model):
            q = q_caches[i, :, start_pos:start_pos + 1, :, :]
            q = torch.repeat_interleave(q, bsz // q.shape[0], dim=0)
            k = kv_caches[0, i, :, start_pos:start_pos + 1, :, :]
            k = torch.repeat_interleave(k, bsz // k.shape[0], dim=0)
            v = kv_caches[1, i, :, start_pos:start_pos + 1, :, :]
            v = torch.repeat_interleave(v, bsz // v.shape[0], dim=0)
            o = attention_block.decoding(q, k, v, start_pos, 1)

    end_time = time.time()
    print("time: ", end_time - start_time)
    # print("time_dict: ", time_dict)
    del model
    return time_dict

start_pos = 128
# seqlen = 4000-128
seqlen = 4000 - start_pos
# test
model = [attention(bsz, max_seq_len, n_local_kv_heads, head_dim, device=device) for _ in range(layer_num)]
print("test")
time_dict = test(start_pos, seqlen, model)
del model
time.sleep(5)
# test_ref
model_ref = [attention_ref(bsz, max_seq_len, n_local_kv_heads, head_dim, device=device, gemm_cuda=False) for _ in range(layer_num)]
print("test_ref")
time_dict_ref = test(start_pos, seqlen, model_ref)
del model_ref
time.sleep(5)
# test_whole
model_p = [attention(bsz, max_seq_len, n_local_kv_heads, head_dim, device=device, reference=True) for _ in range(layer_num)]
print("test_whole")
time_dict_p = test(start_pos, seqlen, model_p)
del model_p
time.sleep(5)

print("time_dict: ", time_dict)
print("time_dict_p: ", time_dict_p)
print("time_dict_ref: ", time_dict_ref)
