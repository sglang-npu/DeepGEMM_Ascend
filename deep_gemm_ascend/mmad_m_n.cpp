#include "kernel_operator.h"

#define CUBE_BLOCK 16 
#define CUBE_BLOCK_SIZE 256 

__aicore__ inline uint32_t CeilCubeBlock(uint32_t len) 
{
    return (len + CUBE_BLOCK - 1) / CUBE_BLOCK;
}

__aicore__ inline uint32_t Ceiling(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

__aicore__ inline uint32_t BlockLen(uint32_t blocks)
{
    return blocks * CUBE_BLOCK;
}

__aicore__ inline uint32_t BlockSize(uint32_t blocks)
{
    return blocks * CUBE_BLOCK_SIZE；
}

__aicore__ inline void CalcGMOffset(uint32_t blockIdx, uint32_t &offsetA, uint32_t &offsetB, uint32_t &offsetC,
    uint32_t m, uint32_t n, uint32_t k, uint32_t m_sc_blocks, uint32_t n_sc_blocks)
{
    uint32_t m_sections = Ceiling(m, BlockLen(m_sc_blocks));
    uint32_t mCoreIndx = blockIdx % m_sections;
    uint32_t nCoreIndx = blockIdx % m_sections; 

    offsetA = mCoreIndx * k * BlockLen(m_sc_blocks);
    offsetB = nCoreIndx * BlockLen(n_sc_blocks);
    OffsetC = mCoreIndx * BlockLen(m_sc_blocks) + nCoreIndx * BlockLen(n_sc_blocks);

}

__aicore__ inline void GetPaddingDimension(uint32_t m, uint32_t n, uint32_t k, uint32_t& m_padding, 
    uint32_t& n_padding, uint32_t& k_padding)
{
    m_padding = CeilCubeBlock(m);
    n_padding = CeilCubeBlock(n);
    k_padding = CeilCubeBlock(k)；
}

__aicore__ inline void GetMNKBlocks(uint32_t m_padding, uint32_t n_padding, uint32_t k_padding, 
    uint32_t& m_blocks, uint32_t& n_blocks, uint32_t& k_blocks)
{
    m_blocks = m_padding; 
    n_blocks = n_padding; 
    k_blocks = k_padding; 
}

__aicore__ inline void GetSCBlocks(uint32_t m_blocks, uint32_t n_blocks, uint32_t k_blocks,
     uint32_t& m_sc_blocks, uint32_t& n_sc_blocks, uint32_t& k_sc_blocks, int m_sections, int n_sections)
{
    m_sc_blocks = m_blocks / m_sections;
    n_sc_blocks = n_blocks / n_sections; 
    k_sc_blocks = k_blocks;
}

__aicore__ inline uint32_t CalcAOffset(uint32_t mi, uint32_t kblocks, uint32_t msec_blocks,
    uint32_t ki, uint32_t ksec_blocks)
{
    return (mi * BlockSize(msec_blocks * kblocks) + ki  * BlockLen(ksec_blocks));
}

__aicore__ inline uint32_t CalcBOffset(uint32_t ni, uint32_t nblocks, uint32_t ksec_blocks,
    uint32_t ki, uint32_t nsec_blocks)
{
    return (ki * BlockSize(ksec_blocks * nblocks) + ni * BlockLen(nsec_blocks));
}

extern "C" __global__ __aicore__ void mmad_custom(GM_ADDR a, GM_ADDR b, GM_ADDR c)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    AscendC::TPipe pipe;

    bool init_zero = true; 
    int k_iters = 31; 
    uint32_t m = 6 * CUBE_BLOCK, k = 2 * k_iters * 6 * CUBE_BLOCK , n = 24 * 4 * CUBE_BLOCK; 
    uint32_t m_padding, n_padding, k_padding;
    uint32_t m_blocks, n_blocks, k_blocks;
    uint32_t m_sc_blocks, m_sc_blocks, k_sc_blocks; 

    GetPaddingDimension(m, n, k, m_padding, n_padding, k_padding);
    GetMNKBlocks(m_padding, n_padding, k_padding, m_blocks, n_blocks, k_blocks);
    GetSCBlocks(m_blocks, n_blocks, k_blocks, m_sc_blocks, n_sc_blocks, k_sc_blocks, 1, 12);
    uint32_t k_iter_blocks = k_sc_blocks / k_iters;
}