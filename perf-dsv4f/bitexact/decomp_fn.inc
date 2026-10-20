static ggml_tensor * append_zero_row(ggml_context * ctx, ggml_tensor * t, bool neg_inf) {
    ggml_tensor * row = ggml_view_1d(ctx, t, t->ne[0], 0);
    row = neg_inf ? ggml_scale_bias(ctx, row, 0.0f, -INFINITY) : ggml_scale(ctx, row, 0.0f);
    row = ggml_reshape_2d(ctx, row, t->ne[0], 1);
    return ggml_concat(ctx, t, row, 1);
}

static ggml_tensor * decomposed(ggml_context * ctx0, ggml_tensor * kv_state, ggml_tensor * score_state, ggml_tensor * idxs, int64_t ratio, int64_t E, int64_t n_blocks, bool overlap) {
    if (!overlap) {
        ggml_tensor * kv = ggml_reshape_3d(ctx0, ggml_get_rows(ctx0, kv_state, idxs), E, ratio, n_blocks);
        ggml_tensor * score = ggml_reshape_3d(ctx0, ggml_get_rows(ctx0, score_state, idxs), E, ratio, n_blocks);
        ggml_tensor * values = ggml_cont(ctx0, ggml_permute(ctx0, kv, 1, 0, 2, 3));
        ggml_tensor * scores = ggml_cont(ctx0, ggml_permute(ctx0, score, 1, 0, 2, 3));
        ggml_tensor * comp = ggml_sum_rows(ctx0, ggml_mul(ctx0, values, ggml_soft_max(ctx0, scores)));
        return ggml_cont(ctx0, ggml_permute(ctx0, comp, 1, 0, 2, 3));
    }
    kv_state    = append_zero_row(ctx0, kv_state,    false);
    score_state = append_zero_row(ctx0, score_state, true);
    const int64_t n_read = ratio*n_blocks;
    ggml_tensor * kv_rows = ggml_get_rows(ctx0, kv_state, idxs);
    ggml_tensor * score_rows = ggml_get_rows(ctx0, score_state, idxs);
    ggml_tensor * kv_prev = ggml_reshape_3d(ctx0, ggml_cont(ctx0, ggml_view_2d(ctx0, kv_rows, E, n_read, kv_rows->nb[1], 0)), E, ratio, n_blocks);
    ggml_tensor * score_prev = ggml_reshape_3d(ctx0, ggml_cont(ctx0, ggml_view_2d(ctx0, score_rows, E, n_read, score_rows->nb[1], 0)), E, ratio, n_blocks);
    ggml_tensor * kv_cur = ggml_reshape_3d(ctx0, ggml_cont(ctx0, ggml_view_2d(ctx0, kv_rows, E, n_read, kv_rows->nb[1], n_read*kv_rows->nb[1] + ggml_row_size(kv_rows->type, E))), E, ratio, n_blocks);
    ggml_tensor * score_cur = ggml_reshape_3d(ctx0, ggml_cont(ctx0, ggml_view_2d(ctx0, score_rows, E, n_read, score_rows->nb[1], n_read*score_rows->nb[1] + ggml_row_size(score_rows->type, E))), E, ratio, n_blocks);
    ggml_tensor * values = ggml_cont(ctx0, ggml_permute(ctx0, ggml_concat(ctx0, kv_prev, kv_cur, 1), 1, 0, 2, 3));
    ggml_tensor * scores = ggml_cont(ctx0, ggml_permute(ctx0, ggml_concat(ctx0, score_prev, score_cur, 1), 1, 0, 2, 3));
    ggml_tensor * comp = ggml_sum_rows(ctx0, ggml_mul(ctx0, values, ggml_soft_max(ctx0, scores)));
    return ggml_cont(ctx0, ggml_permute(ctx0, comp, 1, 0, 2, 3));
}

