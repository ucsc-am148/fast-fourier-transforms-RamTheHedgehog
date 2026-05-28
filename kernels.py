    """STUDENT FILE: implement the Triton kernels and pipeline drivers.

You implement:
  - Six @triton.jit kernels: f1_kernel, f2_kernel, transpose_kernel,
    f4_kernel_L2, dft_kernel, bailey_scale_kernel.
  - The f1_launch and f2_launch grid-choice wrappers around them.
  - The pipeline drivers: f3_launch, f5_launch, _f6_rec, _f7_rec.
  - f6_factor: the chunk-recipe for F6/F7.

You do NOT implement (left given below):
  - The thin launch wrappers _transpose, _fft_chunk, _scale, _lookup_tw.
    These are mechanical "pick the grid and launch one kernel" helpers.
  - The tuning constants F4_L2_BLOCK_B, DFT_BLOCK_B, SCALE_BLOCK,
    TRANSPOSE_BLOCK.

The signatures below are the ones the harness calls -- your job is to fill
the bodies. When your code passes sanity_check.py, you're done.
"""

import math
import torch
import triton
import triton.language as tl

# Tunings -- GIVEN.
F4_L2_BLOCK_B = 2
DFT_BLOCK_B = 16
SCALE_BLOCK = 32
TRANSPOSE_BLOCK = 32

# =============================================================================
# Device-function helper: complex matmul
# =============================================================================

@triton.jit
def _cdot(a_re, a_im, b_re, b_im):
    """Complex matmul Y = A @ B as four real tl.dot calls."""
    y_re = tl.dot(a_re, b_re, out_dtype=tl.float32) - tl.dot(a_im, b_im, out_dtype=tl.float32)
    y_im = tl.dot(a_re, b_im, out_dtype=tl.float32) + tl.dot(a_im, b_re, out_dtype=tl.float32)
    return y_re, y_im

# =============================================================================
# Chunk factorization for F6 / F7
# =============================================================================

def f6_factor(N: int) -> list[int]:
    """Factor N = 2^k into FFT chunks."""
    chunks = []
    rem = N
    while rem >= 256:
        chunks.append(256)
        rem //= 256
    while rem >= 16:
        chunks.append(16)
        rem //= 16
    if rem > 1:
        chunks.append(rem)
    return chunks

f7_factor = f6_factor

# =============================================================================
# F1: DFT as one dense complex matmul (four tl.dot)
# =============================================================================

@triton.jit
def f1_kernel(
    x_re_ptr, x_im_ptr,    # (B, N) fp16
    W_re_ptr, W_im_ptr,    # (N, N) fp16; W[n, k]
    y_re_ptr, y_im_ptr,    # (B, N) fp32
    B,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = rm < B
    mask_n = rn < N

    acc_re = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_im = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, N, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)
        mask_k = rk < N

        x_re_offsets = rm[:, None] * N + rk[None, :]
        x_re = tl.load(x_re_ptr + x_re_offsets, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        x_im = tl.load(x_im_ptr + x_re_offsets, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        W_T_re_offsets = rk[:, None] * N + rn[None, :]
        W_T_re = tl.load(W_re_ptr + W_T_re_offsets, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        W_T_im = tl.load(W_im_ptr + W_T_re_offsets, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        tmp_re, tmp_im = _cdot(x_re, x_im, W_T_re, W_T_im)
        acc_re += tmp_re
        acc_im += tmp_im

    y_offsets = rm[:, None] * N + rn[None, :]
    tl.store(y_re_ptr + y_offsets, acc_re, mask=mask_m[:, None] & mask_n[None, :])
    tl.store(y_im_ptr + y_offsets, acc_im, mask=mask_m[:, None] & mask_n[None, :])


def f1_launch(x_re, x_im, W_re, W_im, y_re, y_im):
    B, N = x_re.shape
    BLOCK_M, BLOCK_N, BLOCK_K = 16, 16, 16
    grid = (triton.cdiv(B, BLOCK_M), triton.cdiv(N, BLOCK_N))
    f1_kernel[grid](
        x_re, x_im, W_re, W_im, y_re, y_im,
        B, N, BLOCK_M, BLOCK_K, BLOCK_N
    )

# =============================================================================
# F2: radix-2 Cooley-Tukey, single program per signal
# =============================================================================

@triton.jit
def f2_kernel(
    x_re_ptr, x_im_ptr,        # (B, N) fp32 input
    y_re_ptr, y_im_ptr,        # (B, N) fp32 output
    tw_re_ptr, tw_im_ptr,      # (N/2,) fp32 radix-2 twiddles
    perm_ptr,                   # (N,) int32 bit-reversal index
    bt_re_ptr, bt_im_ptr,       # (OUTER_DIM, N) fp32 Bailey twiddles
    OUTER_DIM, N_TOTAL,
    N: tl.constexpr,
    LOG2_N: tl.constexpr,
    BAILEY_EPILOGUE: tl.constexpr,
    STRIDED_STORE: tl.constexpr,
):
    pid = tl.program_id(0)
    if BAILEY_EPILOGUE or STRIDED_STORE:
        b_idx = pid // OUTER_DIM
        outer_idx = pid % OUTER_DIM
    else:
        b_idx = pid
        outer_idx = 0

    idx = tl.arange(0, N)
    rev_idx = tl.load(perm_ptr + idx)
    x_signal_offset = b_idx * N_TOTAL if STRIDED_STORE else pid * N

    v_re = tl.load(x_re_ptr + x_signal_offset + rev_idx)
    v_im = tl.load(x_im_ptr + x_signal_offset + rev_idx)

    for s in range(LOG2_N):
        span = 1 << s
        half_n = N >> (s + 1)
        is_odd = (idx >> s) & 1
        partner_idx = idx ^ span

        v_partner_re = tl.reshape(tl.gather(v_re, partner_idx), (N,))
        v_partner_im = tl.reshape(tl.gather(v_im, partner_idx), (N,))

        tw_idx = (idx & (span - 1)) * half_n
        w_re = tl.load(tw_re_ptr + tw_idx, mask=tw_idx < (N // 2), other=0.0)
        w_im = tl.load(tw_im_ptr + tw_idx, mask=tw_idx < (N // 2), other=0.0)

        t_re = w_re * v_partner_re - w_im * v_partner_im
        t_im = w_re * v_partner_im + w_im * v_partner_re

        v_re_new = tl.where(is_odd, v_partner_re - t_re, v_re + t_re)
        v_im_new = tl.where(is_odd, v_partner_im - t_im, v_im + t_im)

        v_re = v_re_new
        v_im = v_im_new

    if BAILEY_EPILOGUE:
        bt_offset = outer_idx * N + idx
        bt_re = tl.load(bt_re_ptr + bt_offset)
        bt_im = tl.load(bt_im_ptr + bt_offset)
        v_re, v_im = v_re * bt_re - v_im * bt_im, v_re * bt_im + v_im * bt_re

    if STRIDED_STORE:
        store_offset = b_idx * N_TOTAL + idx * OUTER_DIM + outer_idx
        tl.store(y_re_ptr + store_offset, v_re)
        tl.store(y_im_ptr + store_offset, v_im)
    else:
        tl.store(y_re_ptr + pid * N + idx, v_re)
        tl.store(y_im_ptr + pid * N + idx, v_im)


def f2_launch(x_re, x_im, y_re, y_im, tw_re, tw_im, perm):
    B, N = x_re.shape
    log2_n = int(math.log2(N))
    f2_kernel[(B,)](
        x_re, x_im, y_re, y_im, tw_re, tw_im, perm,
        x_re, x_im, 1, N, N, log2_n, False, False
    )

# =============================================================================
# transpose_kernel: (B, R, C) -> (B, C, R), paired re/im
# =============================================================================

@triton.jit
def transpose_kernel(
    x_re_ptr, x_im_ptr, y_re_ptr, y_im_ptr, R, C,
    BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr,
):
    pid_r, pid_c, pid_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    mask_x = (r[:, None] < R) & (c[None, :] < C)
    x_offset = pid_b * (R * C) + r[:, None] * C + c[None, :]

    a_re = tl.load(x_re_ptr + x_offset, mask=mask_x, other=0.0)
    a_im = tl.load(x_im_ptr + x_offset, mask=mask_x, other=0.0)

    mask_y = (c[:, None] < C) & (r[None, :] < R)
    y_offset = pid_b * (R * C) + c[:, None] * R + r[None, :]

    tl.store(y_re_ptr + y_offset, tl.trans(a_re), mask=mask_y)
    tl.store(y_im_ptr + y_offset, tl.trans(a_im), mask=mask_y)

# =============================================================================
# F4: tcFFT radix-16 single-program FFT (N = 256, L = 2)
# =============================================================================
@triton.jit
def f4_kernel_L2(
    x_re_ptr, x_im_ptr, y_re_ptr, y_im_ptr, F_re_ptr, F_im_ptr, tw_re_ptr, tw_im_ptr,
    B, M, BLOCK_B: tl.constexpr, STAGE_STOP: tl.constexpr, STORE_T: tl.constexpr,
):
    pid = tl.program_id(0)
    rb = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = rb < B

    idx_256 = tl.arange(0, 256)
    x_offsets = rb[:, None] * 256 + idx_256[None, :]

    tile_re = tl.load(x_re_ptr + x_offsets, mask=mask_b[:, None], other=0.0).to(tl.float32)
    tile_im = tl.load(x_im_ptr + x_offsets, mask=mask_b[:, None], other=0.0).to(tl.float32)

    tile_re = tl.reshape(tile_re, (BLOCK_B, 16, 16))
    tile_im = tl.reshape(tile_im, (BLOCK_B, 16, 16))

    r16 = tl.arange(0, 16)
    F_offsets = r16[:, None] * 16 + r16[None, :]
    f_re = tl.load(F_re_ptr + F_offsets).to(tl.float16)
    f_im = tl.load(F_im_ptr + F_offsets).to(tl.float16)
    tile_re = tl.reshape(tile_re, (BLOCK_B * 16, 16))
    tile_im = tl.reshape(tile_im, (BLOCK_B * 16, 16))
    t0_re, t0_im = _cdot(tile_re.to(tl.float16), tile_im.to(tl.float16), f_re, f_im)
    tile_re = tl.reshape(t0_re, (BLOCK_B, 16, 16))
    tile_im = tl.reshape(t0_im, (BLOCK_B, 16, 16))

    if STAGE_STOP > 1:
        tile_re = tl.permute(tile_re, (0, 2, 1))
        tile_im = tl.permute(tile_im, (0, 2, 1))

        w_re = tl.load(tw_re_ptr + 256 + F_offsets)[None, :, :]
        w_im = tl.load(tw_im_ptr + 256 + F_offsets)[None, :, :]

        res_re = tile_re * w_re - tile_im * w_im
        res_im = tile_re * w_im + tile_im * w_re

        res_re = tl.reshape(res_re, (BLOCK_B * 16, 16))
        res_im = tl.reshape(res_im, (BLOCK_B * 16, 16))
        t1_re, t1_im = _cdot(res_re.to(tl.float16), res_im.to(tl.float16), f_re, f_im)

        tile_re = tl.reshape(t1_re, (BLOCK_B, 16, 16))
        tile_im = tl.reshape(t1_im, (BLOCK_B, 16, 16))

    tile_re = tl.permute(tile_re, (0, 2, 1))
    tile_im = tl.permute(tile_im, (0, 2, 1))

    if STORE_T:
        tile_re_flat_T = tl.reshape(tile_re, (BLOCK_B, 256)).to(tl.float16)
        tile_im_flat_T = tl.reshape(tile_im, (BLOCK_B, 256)).to(tl.float16)

        mask_b_2D = tl.broadcast_to(mask_b[:, None], (BLOCK_B, 256))
        y_offsets_T = (rb // M)[:, None] * (256 * M) + idx_256[None, :] * M + (rb % M)[:, None]

        tl.store(y_re_ptr + y_offsets_T, tile_re_flat_T, mask=mask_b_2D)
        tl.store(y_im_ptr + y_offsets_T, tile_im_flat_T, mask=mask_b_2D)
    else:
        tile_re_flat = tl.reshape(tile_re, (BLOCK_B, 256)).to(tl.float16)
        tile_im_flat = tl.reshape(tile_im, (BLOCK_B, 256)).to(tl.float16)
        tl.store(y_re_ptr + x_offsets, tile_re_flat, mask=mask_b[:, None])
        tl.store(y_im_ptr + x_offsets, tile_im_flat, mask=mask_b[:, None])

# =============================================================================
# dft_kernel: padded length-R DFT for the small chunks (R in {2, 4, 8, 16})
# =============================================================================

@triton.jit
def dft_kernel(
    x_re_ptr, x_im_ptr, y_re_ptr, y_im_ptr, M_re_ptr, M_im_ptr, rows, M,
    R: tl.constexpr, BLOCK_B: tl.constexpr, STORE_T: tl.constexpr,
):
    pid = tl.program_id(0)
    rr = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_r = rr < rows
    idx_16 = tl.arange(0, 16)

    mask_load = mask_r[:, None] & (idx_16[None, :] < R)
    x_offsets = rr[:, None] * R + idx_16[None, :]

    in_re = tl.load(x_re_ptr + x_offsets, mask=mask_load, other=0.0)
    in_im = tl.load(x_im_ptr + x_offsets, mask=mask_load, other=0.0)

    m_offsets = idx_16[:, None] * 16 + idx_16[None, :]
    m_re = tl.load(M_re_ptr + m_offsets)
    m_im = tl.load(M_im_ptr + m_offsets)

    res_re, res_im = _cdot(in_re, in_im, m_re, m_im)

    out_re = res_re.to(tl.float16)
    out_im = res_im.to(tl.float16)

    if STORE_T:
        mask_store_T = mask_r[None, :] & (idx_16[:, None] < R)
        y_offsets_T = (rr // M)[None, :] * (R * M) + idx_16[:, None] * M + (rr % M)[None, :]

        tl.store(y_re_ptr + y_offsets_T, tl.trans(out_re), mask=mask_store_T)
        tl.store(y_im_ptr + y_offsets_T, tl.trans(out_im), mask=mask_store_T)
    else:
        mask_store = mask_r[:, None] & (idx_16[None, :] < R)
        tl.store(y_re_ptr + x_offsets, out_re, mask=mask_store)
        tl.store(y_im_ptr + x_offsets, out_im, mask=mask_store)

# =============================================================================
# bailey_scale_kernel: elementwise w_N^{n1 kM} multiply with optional fused T2
# =============================================================================

@triton.jit
def bailey_scale_kernel(
    x_re_ptr, x_im_ptr, y_re_ptr, y_im_ptr, tw_re_ptr, tw_im_ptr, m0, M,
    BLOCK_M0: tl.constexpr, BLOCK_M: tl.constexpr, STORE_T: tl.constexpr,
):
    pid_m0, pid_m, pid_row = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rm0 = pid_m0 * BLOCK_M0 + tl.arange(0, BLOCK_M0)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    mask_2d = (rm0[:, None] < m0) & (rm[None, :] < M)
    linear_in_offset = pid_row * (m0 * M) + rm0[:, None] * M + rm[None, :]

    v_re = tl.load(x_re_ptr + linear_in_offset, mask=mask_2d, other=0.0).to(tl.float32)
    v_im = tl.load(x_im_ptr + linear_in_offset, mask=mask_2d, other=0.0).to(tl.float32)

    w_re = tl.load(tw_re_ptr + rm0[:, None] * M + rm[None, :], mask=mask_2d, other=0.0).to(tl.float32)
    w_im = tl.load(tw_im_ptr + rm0[:, None] * M + rm[None, :], mask=mask_2d, other=0.0).to(tl.float32)

    res_re = (v_re * w_re - v_im * w_im).to(tl.float16)
    res_im = (v_re * w_im + v_im * w_re).to(tl.float16)

    if STORE_T:
        linear_out_offset = pid_row * (m0 * M) + rm[:, None] * m0 + rm0[None, :]
        tl.store(y_re_ptr + linear_out_offset, tl.trans(res_re), mask=(rm[:, None] < M) & (rm0[None, :] < m0))
        tl.store(y_im_ptr + linear_out_offset, tl.trans(res_im), mask=(rm[:, None] < M) & (rm0[None, :] < m0))
    else:
        tl.store(y_re_ptr + linear_in_offset, res_re, mask=mask_2d)
        tl.store(y_im_ptr + linear_in_offset, res_im, mask=mask_2d)


# =============================================================================
# Thin launch wrappers -- GIVEN
# =============================================================================

def _transpose(in_re, in_im, out_re, out_im, B, R, C):
    grid = (triton.cdiv(R, TRANSPOSE_BLOCK), triton.cdiv(C, TRANSPOSE_BLOCK), B)
    transpose_kernel[grid](
        in_re, in_im, out_re, out_im, R, C,
        BLOCK_R=TRANSPOSE_BLOCK, BLOCK_C=TRANSPOSE_BLOCK,
    )

def _fft_chunk(in_re, in_im, out_re, out_im, rows, m, plan, M=1, store_t=False):
    if m == 256:
        f4_plan = plan['f4_plan']
        f4_kernel_L2[(triton.cdiv(rows, F4_L2_BLOCK_B),)](
            in_re.view(rows, 256), in_im.view(rows, 256), out_re.view(rows, 256), out_im.view(rows, 256),
            f4_plan['F_re'], f4_plan['F_im'], f4_plan['tw_re'], f4_plan['tw_im'],
            rows, M, BLOCK_B=F4_L2_BLOCK_B, STAGE_STOP=f4_plan['L'], STORE_T=store_t,
            num_warps=4, num_stages=1,
        )
    else:
        M_re, M_im = plan['dft_mats'][m]
        dft_kernel[(triton.cdiv(rows, DFT_BLOCK_B),)](
            in_re.view(rows, m), in_im.view(rows, m), out_re.view(rows, m), out_im.view(rows, m),
            M_re, M_im, rows, M, R=m, BLOCK_B=DFT_BLOCK_B, STORE_T=store_t,
        )

def _scale(in_re, in_im, out_re, out_im, rows, m0, M, twr, twi, store_t=False):
    grid = (triton.cdiv(m0, SCALE_BLOCK), triton.cdiv(M, SCALE_BLOCK), rows)
    bailey_scale_kernel[grid](
        in_re, in_im, out_re, out_im, twr, twi, m0, M,
        BLOCK_M0=SCALE_BLOCK, BLOCK_M=SCALE_BLOCK, STORE_T=store_t,
    )

def _lookup_tw(plan, m0, M, N_i):
    for (a, b, n, tr, ti) in plan['tw']:
        if a == m0 and b == M and n == N_i:
            return tr, ti
    raise KeyError(f"no twiddle table for (m0={m0}, M={M}, N={N_i})")


# =============================================================================
# Pipeline Launches
# =============================================================================

def f3_launch(in_re, in_im, out_re, out_im, mid_re, mid_im, plan, B):
    N1, N2 = plan['N1'], plan['N2']
    _transpose(in_re, in_im, mid_re, mid_im, B, N2, N1)
    f2_kernel[(B * N1,)](
        mid_re, mid_im, out_re, out_im, plan['tw_re'], plan['tw_im'], plan['permN2'],
        plan['bt_re'], plan['bt_im'], N1, N1 * N2, N2, int(math.log2(N2)), True, False
    )
    _transpose(out_re, out_im, mid_re, mid_im, B, N1, N2)
    f2_kernel[(B * N2,)](
        mid_re, mid_im, out_re, out_im, plan['tw_re_N1'], plan['tw_im_N1'], plan['permN1'],
        mid_re, mid_im, N1, N1 * N2, N1, int(math.log2(N1)), False, True
    )

def f5_launch(in_re, in_im, b0_re, b0_im, b1_re, b1_im, b2_re, b2_im, plan, B):
    _transpose(in_re, in_im, b0_re, b0_im, B, 256, 256)
    _fft_chunk(b0_re, b0_im, b1_re, b1_im, B * 256, 256, plan)
    _scale(b1_re, b1_im, b0_re, b0_im, B, 256, 256, plan['bt_re'], plan['bt_im'])
    _transpose(b0_re, b0_im, b1_re, b1_im, B, 256, 256)
    _fft_chunk(b1_re, b1_im, b0_re, b0_im, B * 256, 256, plan)
    _transpose(b0_re, b0_im, b2_re, b2_im, B, 256, 256)

def _f6_rec(cur_re, cur_im, rows, chunks, plan, cyc):
    if len(chunks) == 1:
        out_re, out_im = cyc.alloc()
        _fft_chunk(cur_re, cur_im, out_re, out_im, rows, chunks[0], plan)
        return out_re, out_im

    m0 = chunks[0]
    M = math.prod(chunks[1:])
    N_i = m0 * M

    t1_re, t1_im = cyc.alloc()
    _transpose(cur_re, cur_im, t1_re, t1_im, rows, M, m0)
    rec_re, rec_im = _f6_rec(t1_re, t1_im, rows * m0, chunks[1:], plan, cyc)

    twr, twi = _lookup_tw(plan, m0, M, N_i)
    scale_re, scale_im = cyc.alloc()
    _scale(rec_re, rec_im, scale_re, scale_im, rows, m0, M, twr, twi, store_t=False)

    t2_re, t2_im = cyc.alloc()
    _transpose(scale_re, scale_im, t2_re, t2_im, rows, m0, M)
    fft_re, fft_im = cyc.alloc()
    _fft_chunk(t2_re, t2_im, fft_re, fft_im, rows * M, m0, plan)

    t3_re, t3_im = cyc.alloc()
    _transpose(fft_re, fft_im, t3_re, t3_im, rows, M, m0)
    return t3_re, t3_im

def _f7_rec(cur_re, cur_im, rows, chunks, plan, cyc):
    if len(chunks) == 1:
        out_re, out_im = cyc.alloc()
        _fft_chunk(cur_re, cur_im, out_re, out_im, rows, chunks[0], plan)
        return out_re, out_im

    m0 = chunks[0]
    M = math.prod(chunks[1:])
    N_i = m0 * M

    t1_re, t1_im = cyc.alloc()
    _transpose(cur_re, cur_im, t1_re, t1_im, rows, M, m0)
    rec_re, rec_im = _f7_rec(t1_re, t1_im, rows * m0, chunks[1:], plan, cyc)

    twr, twi = _lookup_tw(plan, m0, M, N_i)
    scale_t2_re, scale_t2_im = cyc.alloc()
    _scale(rec_re, rec_im, scale_t2_re, scale_t2_im, rows, m0, M, twr, twi, store_t=True)

    f7_out_re, f7_out_im = cyc.alloc()
    _fft_chunk(scale_t2_re, scale_t2_im, f7_out_re, f7_out_im, rows * M, m0, plan, M=M, store_t=True)
    return f7_out_re, f7_out_im