#include "kernel_operator.h"

#define CUBE_BLOCK          16   // block的行列数，单位字节
#define CUBE_BLOCK_SIZE     256  // block的总大小，单位字节

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
    return blocks * CUBE_BLOCK_SIZE;
}

__aicore__ inline uint32_t CalcAOffset(uint32_t mi, uint32_t k, uint32_t msec_blocks,
    uint32_t ki, uint32_t ksec_blocks)
{
    return (mi * BlockLen(msec_blocks) * k + ki * BlockLen(ksec_blocks));
}

__aicore__ inline uint32_t CalcBOffset(uint32_t ni, uint32_t n, uint32_t ksec_blocks,
    uint32_t ki, uint32_t nsec_blocks)
{
    return (ki * BlockLen(ksec_blocks) * n + ni * BlockLen(nsec_blocks));
}

extern "C" __global__ __aicore__ void mmad_custom(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR params)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    AscendC::TPipe pipe;

    bool init_zero = true;

    AscendC::GlobalTensor<uint32_t> paramGM;
    paramGM.SetGlobalBuffer((__gm__ uint32_t *)params);
    // all the params | 28 
    uint32_t m = paramGM.GetValue(0);
    uint32_t n = paramGM.GetValue(1);
    uint32_t k = paramGM.GetValue(2);
    uint32_t m_sections = paramGM.GetValue(3); // 核间切分，m维度切分数量
    uint32_t n_sections = paramGM.GetValue(4); // 核间切分，n维度切分数量
    uint32_t m_sec_o_blocks = paramGM.GetValue(5); // 核内切分，m维度每块的大小，单位block
    uint32_t n_sec_o_blocks = paramGM.GetValue(6); // 核内切分，n维度每块的大小，单位block
    uint32_t k_o_iter_blocks = paramGM.GetValue(7); // 核内切分，k维度每块的大小，单位block
    uint32_t db_o_blocks = paramGM.GetValue(8); // 核内切分，double buffer每块的大小，单位block
    uint32_t batch = paramGM.GetValue(9);
    uint32_t k_iters = paramGM.GetValue(10);
    uint32_t m_blocks = paramGM.GetValue(11);
    uint32_t n_blocks = paramGM.GetValue(12);
    uint32_t k_blocks = paramGM.GetValue(13);
    uint32_t m_sc_blocks = paramGM.GetValue(14);
    uint32_t n_sc_blocks = paramGM.GetValue(15);

    uint32_t m_o_fix = paramGM.GetValue(16);
    uint32_t n_o_fix = paramGM.GetValue(17);
    uint32_t k_o_fix = paramGM.GetValue(18);
    uint32_t db_o_num = paramGM.GetValue(19);

    uint32_t m_parts = paramGM.GetValue(20);
    uint32_t n_parts = paramGM.GetValue(21);
    uint32_t r_m_parts = paramGM.GetValue(22);
    uint32_t r_n_parts = paramGM.GetValue(23);

    uint32_t r_m_blocks = paramGM.GetValue(24);
    uint32_t r_n_blocks = paramGM.GetValue(25);
    uint32_t r_k_blocks = paramGM.GetValue(26);
    uint32_t r_db_num = paramGM.GetValue(27);
    
    // expand the batch loop
    for(uint32_t bi = 0; bi < batch; bi++)
    {
        uint32_t offsetA = bi * m * k;
        uint32_t offsetB = bi * n * k;
        uint32_t offsetC = bi * m * n;
        uint32_t a_offset, b_offset;
        uint32_t msec_blocks, nsec_blocks, k_iter_blocks, db_blocks;
        uint32_t m_fix, n_fix, k_fix;
        uint32_t db_num;

        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t mCoreIndx = blockIdx % m_sections;
        uint32_t nCoreIndx = blockIdx / m_sections;
        bool is_last_m = mCoreIndx == (m_sections - 1);
        bool is_last_n = nCoreIndx == (n_sections - 1);

        if (is_last_m)
        {
            m_parts = r_m_parts;
        }
        if (is_last_n)
        {
            n_parts = r_n_parts;
        }

        offsetA += mCoreIndx * k * BlockLen(m_sc_blocks);
        offsetB += nCoreIndx * BlockLen(n_sc_blocks);
        offsetC += mCoreIndx * n * BlockLen(m_sc_blocks) + nCoreIndx * BlockLen(n_sc_blocks);
        
        // 核内每次参与计算的unit 切割大小
        int a_buffer_size = BlockSize(m_sec_o_blocks * k_o_iter_blocks);
        int b_buffer_size = BlockSize(n_sec_o_blocks * k_o_iter_blocks);
        int c_buffer_size = BlockSize(m_sec_o_blocks * n_sec_o_blocks);
        
        AscendC::GlobalTensor<half> aGM;
        aGM.SetGlobalBuffer((__gm__ half *)a);
        AscendC::TQue<AscendC::TPosition::A1, 1> inQueueA1;
        pipe.InitBuffer(inQueueA1, 2, a_buffer_size * sizeof(half)); // 每个A1分配出两块内存，每块大小为a_buffer_size * 2字节（half）

        AscendC::GlobalTensor<half> bGM;
        bGM.SetGlobalBuffer((__gm__ half *)b);
        AscendC::TQue<AscendC::TPosition::B1, 1> inQueueB1;
        pipe.InitBuffer(inQueueB1, 2, b_buffer_size * sizeof(half));

        AscendC::GlobalTensor<float> cGM;
        cGM.SetGlobalBuffer((__gm__ float *)c);






    }

}