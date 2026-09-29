/* Gated short convolution kernels.  */

/* Run the gated short convolution over N_TOKENS tokens.  Each token's
   3 * N_EMBD floats at BCX hold the gates B and C and the input X, and
   its N_EMBD results go to OUT.  HISTORY holds B times X for the
   previous KERNEL - 1 tokens, oldest first, and moves forward one token
   at a time.  TAPS holds KERNEL taps per channel.  One thread handles
   one channel for every token in order.  */
kernel void
short_conv (device const float *bcx [[buffer (0)]],
            device const float *taps [[buffer (1)]],
            device float *history [[buffer (2)]],
            device float *out [[buffer (3)]],
            constant uint &n_embd [[buffer (4)]],
            constant uint &kernel_size [[buffer (5)]],
            constant uint &n_tokens [[buffer (6)]],
            uint ch [[thread_position_in_grid]])
{
  if (ch >= n_embd)
    return;

  device const float *channel_taps = taps + ch * kernel_size;
  for (uint t = 0; t < n_tokens; t++)
    {
      device const float *token_bcx = bcx + t * 3 * n_embd;
      float bx = token_bcx[ch] * token_bcx[2 * n_embd + ch];
      float sum = channel_taps[kernel_size - 1] * bx;
      for (uint k = 0; k + 1 < kernel_size; k++)
        sum += channel_taps[k] * history[k * n_embd + ch];
      for (uint k = 0; k + 2 < kernel_size; k++)
        history[k * n_embd + ch] = history[(k + 1) * n_embd + ch];
      history[(kernel_size - 2) * n_embd + ch] = bx;
      out[t * n_embd + ch] = token_bcx[n_embd + ch] * sum;
    }
}

/* Run the gated short convolution over N_TOKENS tokens at once, as the
   first of two passes.  Each token's 3 * N_EMBD floats at BCX hold the
   gates B and C and the input X, and its N_EMBD results go to OUT.
   HISTORY holds B times X for the KERNEL - 1 tokens before the batch,
   oldest first.  Each output depends only on inputs, so one thread
   handles one channel of one token.  */
kernel void
short_conv_batch (device const float *bcx [[buffer (0)]],
                  device const float *taps [[buffer (1)]],
                  device const float *history [[buffer (2)]],
                  device float *out [[buffer (3)]],
                  constant uint &n_embd [[buffer (4)]],
                  constant uint &kernel_size [[buffer (5)]],
                  constant uint &n_tokens [[buffer (6)]],
                  uint2 position [[thread_position_in_grid]])
{
  uint ch = position.x;
  uint t = position.y;
  if (ch >= n_embd || t >= n_tokens)
    return;

  device const float *channel_taps = taps + ch * kernel_size;
  float sum = 0.0f;
  for (uint k = 0; k < kernel_size; k++)
    {
      /* Tap K multiplies the input from KERNEL - 1 - K tokens ago, which
         comes from the history when it predates the batch.  */
      int source = int (t) - int (kernel_size - 1 - k);
      float bx;
      if (source >= 0)
        {
          device const float *row = bcx + uint (source) * 3 * n_embd;
          bx = row[ch] * row[2 * n_embd + ch];
        }
      else
        bx = history[uint (source + int (kernel_size - 1)) * n_embd + ch];
      sum += channel_taps[k] * bx;
    }
  out[t * n_embd + ch] = bcx[t * 3 * n_embd + n_embd + ch] * sum;
}

/* Move HISTORY forward past the N_TOKENS tokens at BCX, as the second
   pass of short_conv_batch.  One thread handles one channel.  */
kernel void
short_conv_history (device const float *bcx [[buffer (0)]],
                    device float *history [[buffer (1)]],
                    constant uint &n_embd [[buffer (2)]],
                    constant uint &kernel_size [[buffer (3)]],
                    constant uint &n_tokens [[buffer (4)]],
                    uint ch [[thread_position_in_grid]])
{
  if (ch >= n_embd)
    return;

  uint n_history = kernel_size - 1;
  float next[8];
  for (uint k = 0; k < n_history; k++)
    {
      /* Slot K of the new history holds the input from N_HISTORY - K
         tokens before the end of the batch.  */
      int source = int (n_tokens) - int (n_history - k);
      if (source >= 0)
        {
          device const float *row = bcx + uint (source) * 3 * n_embd;
          next[k] = row[ch] * row[2 * n_embd + ch];
        }
      else
        next[k] = history[uint (source + int (n_history)) * n_embd + ch];
    }
  for (uint k = 0; k < n_history; k++)
    history[k * n_embd + ch] = next[k];
}
