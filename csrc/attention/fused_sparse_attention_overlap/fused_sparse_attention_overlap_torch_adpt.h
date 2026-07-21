#ifndef FUSED_SPARSE_ATTENTION_OVERLAP_TORCH_ADPT_H
#define FUSED_SPARSE_ATTENTION_OVERLAP_TORCH_ADPT_H

#include <string>
#include <vector>

namespace vllm_ascend {

inline void CheckPositiveShape(const at::Tensor &tensor, const char *name)
{
    for (size_t i = 0; i < tensor.sizes().size(); i++) {
        TORCH_CHECK(tensor.size(i) > 0, "All values within ", name, "'s shape should be greater than 0, but shape[",
                    i, "] is ", tensor.size(i));
    }
}

inline c10::optional<at::Tensor> MaybeQueryRope(const at::Tensor &query_rope)
{
    if (!query_rope.defined() || query_rope.numel() == 0) {
        return c10::nullopt;
    }
    return query_rope;
}

inline at::Tensor ConstructSelectionKvActualSeq(const at::Tensor &selection_kv_block_table,
                                                const at::Tensor &selection_topk_indices)
{
    std::vector<int64_t> actual_seq_shape = {selection_kv_block_table.size(0)};
    return at::empty(actual_seq_shape, selection_topk_indices.options().dtype(at::kInt));
}

inline at::Tensor ConstructSelectionKvActualSeqForSideEffect(const at::Tensor &selection_kv_block_table,
                                                             const at::Tensor &selection_kv_block_status,
                                                             const at::Tensor &selection_topk_indices)
{
    int64_t topk = selection_topk_indices.size(selection_topk_indices.dim() - 1);
    if (selection_kv_block_table.size(0) == 1 &&
        selection_kv_block_status.is_contiguous() &&
        selection_kv_block_status.numel() == topk + 1) {
        return selection_kv_block_status.view({1, topk + 1}).select(1, topk);
    }
    return ConstructSelectionKvActualSeq(selection_kv_block_table, selection_topk_indices);
}

inline int64_t GetSelectionTopkHeadNum(const at::Tensor &selection_topk_indices)
{
    TORCH_CHECK(selection_topk_indices.dim() == 3 || selection_topk_indices.dim() == 4,
                "selection_topk_indices dim should be 3 or 4, but got ",
                selection_topk_indices.dim());
    return selection_topk_indices.size(selection_topk_indices.dim() - 2);
}

inline at::Tensor FlattenTopkIndicesForSfa(const at::Tensor &selection_topk_indices,
                                           int64_t bsz_seq,
                                           int64_t sparse_head_num)
{
    int64_t topk = selection_topk_indices.size(selection_topk_indices.dim() - 1);
    if (selection_topk_indices.dim() == 4) {
        if (selection_topk_indices.is_contiguous()) {
            return selection_topk_indices.view({bsz_seq, sparse_head_num, topk});
        }
        return selection_topk_indices.contiguous().view({bsz_seq, sparse_head_num, topk});
    }
    if (selection_topk_indices.is_contiguous()) {
        return selection_topk_indices;
    }
    return selection_topk_indices.contiguous();
}

inline at::Tensor BuildActualSeqQueryForSfa(const at::Tensor &full_q_actual_seq,
                                            int64_t bsz_seq,
                                            const at::Tensor &like_tensor)
{
    // TND: full_q_actual_seq length is num_reqs (B), with cumulative query
    // lengths ending at total tokens (query.size(0) == bsz_seq). MTP decode
    // has num_tokens != num_reqs, so do NOT require numel == bsz_seq.
    // Legacy fallback (no Q seq provided): treat each token as its own seq.
    if (full_q_actual_seq.defined() && full_q_actual_seq.numel() > 0) {
        TORCH_CHECK(
            full_q_actual_seq.numel() <= bsz_seq,
            "full_q_actual_seq numel (", full_q_actual_seq.numel(),
            ") must be <= query token count (", bsz_seq,
            ") for TND fused sparse attention overlap");
        return full_q_actual_seq.contiguous();
    }
    return at::ones({bsz_seq}, like_tensor.options().dtype(at::kInt));
}

inline at::Tensor BuildFullCacheSparseIndicesForSfa(const at::Tensor &selection_topk_indices,
                                                    int64_t bsz_seq,
                                                    int64_t sparse_head_num)
{
    return FlattenTopkIndicesForSfa(selection_topk_indices, bsz_seq, sparse_head_num);
}

inline at::Tensor RunFusedSparseAttentionOverlapSideEffectSplit(
    const at::Tensor &query_nope,
    const at::Tensor &query_rope,
    const at::Tensor &selection_k_rope,
    const at::Tensor &selection_kv_cache,
    const at::Tensor &selection_kv_block_table,
    const at::Tensor &selection_kv_block_status,
    const at::Tensor &selection_topk_indices,
    const at::Tensor &full_k_rope,
    const at::Tensor &full_kv_cache,
    const at::Tensor &full_kv_block_table,
    const at::Tensor &full_kv_actual_seq,
    const at::Tensor &full_q_actual_seq,
    double scale_value,
    int64_t sparse_block_size,
    int64_t selection_topk_block_size,
    const std::string &layout_query_str,
    const std::string &layout_kv_str,
    int64_t sparse_mode)
{
    TORCH_CHECK(selection_topk_block_size == 1,
                "fused sparse attention overlap standalone side-effect path requires selection_topk_block_size=1, "
                "but got ", selection_topk_block_size);
    CheckPositiveShape(query_nope, "query_nope");
    int64_t bsz_seq = query_nope.size(0);
    int64_t sparse_head_num = GetSelectionTopkHeadNum(selection_topk_indices);
    int64_t kv_cache_dim = full_kv_cache.size(full_kv_cache.dim() - 1);
    TORCH_CHECK(query_nope.size(query_nope.dim() - 1) == kv_cache_dim,
                "query_nope last dim should equal full_kv_cache last dim, but got ",
                query_nope.size(query_nope.dim() - 1), " and ", kv_cache_dim);

    at::Tensor selection_kv_actual_seq = ConstructSelectionKvActualSeqForSideEffect(
        selection_kv_block_table, selection_kv_block_status, selection_topk_indices);
    at::Tensor sparse_indices = BuildFullCacheSparseIndicesForSfa(
        selection_topk_indices, bsz_seq, sparse_head_num);
    at::Tensor key = full_kv_cache.unsqueeze(2);
    at::Tensor value = key;
    at::Tensor actual_seq_query = BuildActualSeqQueryForSfa(full_q_actual_seq, bsz_seq, selection_topk_indices);
    at::Tensor actual_seq_kv = full_kv_actual_seq.contiguous();
    TORCH_CHECK(
        actual_seq_kv.numel() == actual_seq_query.numel(),
        "full_kv_actual_seq numel (", actual_seq_kv.numel(),
        ") must equal full_q_actual_seq numel (", actual_seq_query.numel(),
        ") for fused sparse attention overlap; if Q was expanded to ones(num_tokens), "
        "rebuild the C++ extension so TND cum_query_lens(length=num_reqs) is preserved");
    TORCH_CHECK(
        full_kv_block_table.size(0) == actual_seq_query.numel(),
        "full_kv_block_table batch (", full_kv_block_table.size(0),
        ") must equal full_q_actual_seq numel (", actual_seq_query.numel(), ")");
    at::Tensor sfa_output = at::empty(query_nope.sizes(), query_nope.options().dtype(query_nope.dtype()));

    c10::optional<at::Tensor> block_table_opt = full_kv_block_table;
    c10::optional<at::Tensor> actual_seq_query_opt = actual_seq_query;
    c10::optional<at::Tensor> actual_seq_kv_opt = actual_seq_kv;
    c10::optional<at::Tensor> query_rope_opt = MaybeQueryRope(query_rope);
    c10::optional<at::Tensor> key_rope_opt = c10::nullopt;
    if (query_rope_opt.has_value()) {
        key_rope_opt = full_k_rope.unsqueeze(2);
    }

    char *layout_query_ptr = const_cast<char *>(layout_query_str.c_str());
    char *layout_kv_ptr = const_cast<char *>(layout_kv_str.c_str());
    EXEC_NPU_CMD(aclnnFusedSparseAttentionOverlap,
                 query_nope,
                 key,
                 value,
                 sparse_indices,
                 block_table_opt,
                 actual_seq_query_opt,
                 actual_seq_kv_opt,
                 query_rope_opt,
                 key_rope_opt,
                 selection_k_rope,
                 selection_kv_cache,
                 selection_kv_block_table,
                 selection_kv_block_status,
                 scale_value,
                 sparse_block_size,
                 layout_query_ptr,
                 layout_kv_ptr,
                 sparse_mode,
                 sfa_output,
                 selection_kv_actual_seq);

    return sfa_output;
}

inline at::Tensor RunFusedSparseAttentionOverlapSideEffect(
    const at::Tensor &query,
    const at::Tensor &selection_k_rope,
    const at::Tensor &selection_kv_cache,
    const at::Tensor &selection_kv_block_table,
    const at::Tensor &selection_kv_block_status,
    const at::Tensor &selection_topk_indices,
    const at::Tensor &full_k_rope,
    const at::Tensor &full_kv_cache,
    const at::Tensor &full_kv_block_table,
    const at::Tensor &full_kv_actual_seq,
    const at::Tensor &full_q_actual_seq,
    double scale_value,
    int64_t sparse_block_size,
    int64_t selection_topk_block_size,
    const std::string &layout_query_str,
    const std::string &layout_kv_str,
    int64_t sparse_mode)
{
    CheckPositiveShape(query, "query");
    int64_t last_dim = query.dim() - 1;
    int64_t query_dim = query.size(last_dim);
    int64_t kv_cache_dim = full_kv_cache.size(full_kv_cache.dim() - 1);
    int64_t k_rope_dim = full_k_rope.defined() && full_k_rope.numel() > 0
                             ? full_k_rope.size(full_k_rope.dim() - 1)
                             : 0;
    TORCH_CHECK(query_dim >= kv_cache_dim,
                "query last dim should be >= full_kv_cache last dim, but got ", query_dim, " and ", kv_cache_dim);
    at::Tensor query_nope = query.slice(last_dim, 0, kv_cache_dim);
    at::Tensor query_rope = (k_rope_dim > 0 && query_dim >= kv_cache_dim + k_rope_dim)
                                ? query.slice(last_dim, kv_cache_dim, kv_cache_dim + k_rope_dim)
                                : at::Tensor();
    at::Tensor sfa_output = RunFusedSparseAttentionOverlapSideEffectSplit(
        query_nope,
        query_rope,
        selection_k_rope,
        selection_kv_cache,
        selection_kv_block_table,
        selection_kv_block_status,
        selection_topk_indices,
        full_k_rope,
        full_kv_cache,
        full_kv_block_table,
        full_kv_actual_seq,
        full_q_actual_seq,
        scale_value,
        sparse_block_size,
        selection_topk_block_size,
        layout_query_str,
        layout_kv_str,
        sparse_mode);

    at::Tensor attention_output = at::zeros(query.sizes(), query.options().dtype(query.dtype()));
    attention_output.slice(last_dim, 0, kv_cache_dim).copy_(sfa_output);
    return attention_output;
}

inline at::Tensor npu_fused_sparse_attention_overlap(
    const at::Tensor &query,
    const at::Tensor &selection_k_rope,
    const at::Tensor &selection_kv_cache,
    const at::Tensor &selection_kv_block_table,
    const at::Tensor &selection_kv_block_status,
    const at::Tensor &selection_topk_indices,
    const at::Tensor &full_k_rope,
    const at::Tensor &full_kv_cache,
    const at::Tensor &full_kv_block_table,
    const at::Tensor &full_kv_actual_seq,
    const at::Tensor &full_q_actual_seq,
    double scale_value,
    int64_t sparse_block_size,
    int64_t selection_topk_block_size,
    c10::string_view layout_query,
    c10::string_view layout_kv,
    int64_t sparse_mode)
{
    return RunFusedSparseAttentionOverlapSideEffect(
        query,
        selection_k_rope,
        selection_kv_cache,
        selection_kv_block_table,
        selection_kv_block_status,
        selection_topk_indices,
        full_k_rope,
        full_kv_cache,
        full_kv_block_table,
        full_kv_actual_seq,
        full_q_actual_seq,
        scale_value,
        sparse_block_size,
        selection_topk_block_size,
        std::string(layout_query),
        std::string(layout_kv),
        sparse_mode);
}

inline at::Tensor npu_fused_sparse_attention_overlap_cpu_source(
    const at::Tensor &query,
    const at::Tensor &selection_k_rope,
    const at::Tensor &selection_kv_cache,
    const at::Tensor &selection_kv_block_table,
    const at::Tensor &selection_kv_block_status,
    const at::Tensor &selection_topk_indices,
    const at::Tensor &full_k_rope,
    const at::Tensor &full_kv_cache,
    const at::Tensor &full_kv_block_table,
    const at::Tensor &full_kv_actual_seq,
    const at::Tensor &full_q_actual_seq,
    double scale_value,
    int64_t sparse_block_size,
    int64_t selection_topk_block_size,
    c10::string_view layout_query,
    c10::string_view layout_kv,
    int64_t sparse_mode)
{
    return npu_fused_sparse_attention_overlap(query,
                                              selection_k_rope,
                                              selection_kv_cache,
                                              selection_kv_block_table,
                                              selection_kv_block_status,
                                              selection_topk_indices,
                                              full_k_rope,
                                              full_kv_cache,
                                              full_kv_block_table,
                                              full_kv_actual_seq,
                                              full_q_actual_seq,
                                              scale_value,
                                              sparse_block_size,
                                              selection_topk_block_size,
                                              layout_query,
                                              layout_kv,
                                              sparse_mode);
}

} // namespace vllm_ascend
#endif
