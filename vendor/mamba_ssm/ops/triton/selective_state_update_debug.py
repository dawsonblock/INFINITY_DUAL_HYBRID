import torch
import triton
import triton.language as tl
from einops import rearrange

@triton.jit
def _selective_scan_debug_kernel(
    # Pointers
    u_ptr, dt_ptr, A_ptr, B_ptr, C_ptr, D_ptr, z_ptr, dt_bias_ptr,
    out_ptr, trace_ptr,
    # Shapes
    batch, dim, dstate, seqlen,
    nheads, ngroups,
    # Strides
    stride_u_batch, stride_u_dim, stride_u_seqlen,
    stride_dt_batch, stride_dt_dim, stride_dt_seqlen,
    stride_A_dim, stride_A_dstate,
    stride_B_batch, stride_B_group, stride_B_dstate, stride_B_seqlen,
    stride_C_batch, stride_C_group, stride_C_dstate, stride_C_seqlen,
    stride_D_dim,
    stride_z_batch, stride_z_dim, stride_z_seqlen,
    stride_dt_bias_dim,
    stride_out_batch, stride_out_dim, stride_out_seqlen,
    stride_trace_batch, stride_trace_dim, stride_trace_seqlen, stride_trace_dstate,
    # Meta-parameters
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
):
    # Map grid to (batch, dim)
    pid_batch = tl.program_id(0)
    pid_dim = tl.program_id(1)

    # Offsets for DSTATE dimension
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)

    # Pointers to static parameters (A, D, dt_bias)
    # A: (dim, dstate)
    A_ptrs = A_ptr + pid_dim * stride_A_dim + offs_n * stride_A_dstate
    A_val = tl.load(A_ptrs, mask=offs_n < dstate, other=0.0).to(tl.float32)
    
    D_val = 0.0
    if HAS_D:
        D_val = tl.load(D_ptr + pid_dim * stride_D_dim).to(tl.float32)

    dt_bias_val = 0.0
    if HAS_DT_BIAS:
        dt_bias_val = tl.load(dt_bias_ptr + pid_dim * stride_dt_bias_dim).to(tl.float32)

    # Initialize state x to zeros: (dstate,)
    x = tl.zeros((BLOCK_SIZE_DSTATE,), dtype=tl.float32)

    # Group index for B/C (G = dim // (dim // ngroups) ? No, dim = nheads * d_head)
    # Assuming B/C are (batch, ngroups, dstate, seqlen)
    # We need to map pid_dim (which ranges 0..dim-1) to group index.
    # dim = nheads * d_head? No, usually dim IS nheads in the simplified view or (B, D, L)
    # Let's assume dim corresponds to the D dimension in (B, D, L).
    # If B has shape (B, G, N, L), we need G = dim // (dim/ngroups)
    # ratio = dim // ngroups
    ratio = dim // ngroups
    group_idx = pid_dim // ratio

    # Loop over sequence length
    for t in range(seqlen):
        # Load u[b, d, t]
        u_ptr_t = u_ptr + pid_batch * stride_u_batch + pid_dim * stride_u_dim + t * stride_u_seqlen
        u_val = tl.load(u_ptr_t).to(tl.float32)

        # Load dt[b, d, t]
        dt_ptr_t = dt_ptr + pid_batch * stride_dt_batch + pid_dim * stride_dt_dim + t * stride_dt_seqlen
        dt_val = tl.load(dt_ptr_t).to(tl.float32)

        if HAS_DT_BIAS:
            dt_val += dt_bias_val
        
        # Softplus
        dt_val = tl.log(1 + tl.exp(dt_val)) # naive softplus
        
        # Discretize A: dA = exp(dt * A)
        dA = tl.exp(dt_val * A_val)

        # Load B[b, g, n, t]
        B_ptrs_t = B_ptr + pid_batch * stride_B_batch + group_idx * stride_B_group + t * stride_B_seqlen + offs_n * stride_B_dstate
        B_val = tl.load(B_ptrs_t, mask=offs_n < dstate, other=0.0).to(tl.float32)

        # Discretize B: dB = dt * B
        dB = dt_val * B_val

        # State update: x = x * dA + dB * u
        x = x * dA + dB * u_val

        # Save state to trace: trace[b, d, t, :]
        if trace_ptr is not None:
            trace_ptrs_t = trace_ptr + pid_batch * stride_trace_batch + pid_dim * stride_trace_dim + t * stride_trace_seqlen + offs_n * stride_trace_dstate
            tl.store(trace_ptrs_t, x, mask=offs_n < dstate)

        # Load C[b, g, n, t]
        C_ptrs_t = C_ptr + pid_batch * stride_C_batch + group_idx * stride_C_group + t * stride_C_seqlen + offs_n * stride_C_dstate
        C_val = tl.load(C_ptrs_t, mask=offs_n < dstate, other=0.0).to(tl.float32)

        # Output: y = sum(x * C)
        y_val = tl.sum(x * C_val)

        if HAS_D:
            y_val += D_val * u_val

        if HAS_Z:
            z_ptr_t = z_ptr + pid_batch * stride_z_batch + pid_dim * stride_z_dim + t * stride_z_seqlen
            z_val = tl.load(z_ptr_t).to(tl.float32)
            # Silu: z * sigmoid(z)
            z_gate = z_val * (1.0 / (1.0 + tl.exp(-z_val)))
            y_val *= z_gate

        # Store output
        out_ptr_t = out_ptr + pid_batch * stride_out_batch + pid_dim * stride_out_dim + t * stride_out_seqlen
        tl.store(out_ptr_t, y_val)


def selective_scan_debug(
    u, dt, A, B, C, D=None, z=None, dt_bias=None, delta_softplus=True
):
    """
    Triton-based selective scan with state tracing.
    
    Args:
        u: (B, D, L)
        dt: (B, D, L)
        A: (D, N)
        B: (B, G, N, L)
        C: (B, G, N, L)
        D: (D,)
        z: (B, D, L)
        dt_bias: (D,)
    
    Returns:
        out: (B, D, L)
        trace: (B, D, L, N)
    """
    # Checks
    batch, dim, seqlen = u.shape
    dstate = A.shape[1]
    ngroups = B.shape[1]
    
    # Ensure contiguous if needed (Triton likes contiguous usually, but we pass strides)
    # We'll rely on strides.
    
    out = torch.empty_like(u)
    trace = torch.empty((batch, dim, seqlen, dstate), device=u.device, dtype=torch.float32)
    
    grid = (batch, dim)
    
    # Next power of 2 for dstate
    BLOCK_SIZE_DSTATE = triton.next_power_of_2(dstate)
    
    _selective_scan_debug_kernel[grid](
        u, dt, A, B, C, D, z, dt_bias,
        out, trace,
        batch, dim, dstate, seqlen,
        dim, ngroups, # nheads is dim here
        u.stride(0), u.stride(1), u.stride(2),
        dt.stride(0), dt.stride(1), dt.stride(2),
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1), B.stride(2), B.stride(3),
        C.stride(0), C.stride(1), C.stride(2), C.stride(3),
        D.stride(0) if D is not None else 0,
        z.stride(0) if z is not None else 0, z.stride(1) if z is not None else 0, z.stride(2) if z is not None else 0,
        dt_bias.stride(0) if dt_bias is not None else 0,
        out.stride(0), out.stride(1), out.stride(2),
        trace.stride(0), trace.stride(1), trace.stride(2), trace.stride(3),
        HAS_D=D is not None,
        HAS_Z=z is not None,
        HAS_DT_BIAS=dt_bias is not None,
        BLOCK_SIZE_DSTATE=BLOCK_SIZE_DSTATE,
    )
    
    return out, trace
