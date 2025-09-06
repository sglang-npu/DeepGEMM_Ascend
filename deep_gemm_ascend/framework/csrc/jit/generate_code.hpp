#include <string>


namespace deep_gemm_ascend{
namespace utils{
const std::string FILE_HEAD =
R"(
#include \"kernel_operator.h\"

#define CUBE_BLOCK          16
#define CUBE_BLOCK_SIZE     256

__aicore__ inline uint32_t CeilCubeBlock(uint32_t len) {{
    return (len + CUBE_BLOCK - 1) / CUBE_BLOCK;
}}

__aicore__ inline uint32_t Ceiling(uint32_t a, uint32_t b)
{{
    return (a + b - 1) / b;
}}

__aicore__ inline uint32_t BlockLen(uint32_t blocks)
{{
    return blocks * CUBE_BLOCK;
}}

__aicore__ inline uint32_t BlockSize(uint32_t blocks)
{{
    return blocks * CUBE_BLOCK_SIZE;
}}

__aicore__ inline uint32_t CalcAOffset(uint32_t mi, uint32_t k, uint32_t msec_blocks,
    uint32_t ki, uint32_t ksec_blocks)
{{
    return (mi * BlockLen(msec_blocks) * k + ki * BlockLen(ksec_blocks));
}}

__aicore__ inline uint32_t CalcBOffset(uint32_t ni, uint32_t n, uint32_t ksec_blocks,
    uint32_t ki, uint32_t nsec_blocks)
{{
    return (ki * BlockLen(ksec_blocks) * n + ni * BlockLen(nsec_blocks));
}}
)";

const std::string FILE_ARGS =
R"(
extern "C" __global__ __aicore__ void mmad_custom(GM_ADDR a, GM_ADDR b, GM_ADDR c)
{{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    AscendC::TPipe pipe;

    bool init_zero = true;
    int k_iters = {};
    uint32_t batch = {};
    uint32_t m = {}, k = {}, n = {};
    uint32_t m_sections = {}, n_sections = {};
    uint32_t m_blocks = {}, n_blocks = {}, k_blocks = {};
    uint32_t m_sc_blocks = {}, n_sc_blocks = {};

    uint32_t msec_o_blocks = {};
    uint32_t nsec_o_blocks = {};
    uint32_t k_o_iter_blocks = {};
    uint32_t db_o_blocks = {};

    uint32_t m_o_fix = {}, n_o_fix = {}, k_o_fix = {};
    uint32_t db_o_num = {};

    uint32_t m_parts = {}, n_parts = {};
    uint32_t r_m_parts = {}, r_n_parts = {};

    uint32_t r_m_blocks = {};
    uint32_t r_n_blocks = {};
    uint32_t r_k_blocks = {};
    uint32_t r_db_num = {};
)";

const std::string FILE_EXEC =
R"(
    uint32_t offsetA, offsetB, offsetC;
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
    {{
        m_parts = r_m_parts;
    }}
    if (is_last_n)
    {{
        n_parts = r_n_parts;
    }}

    offsetA = mCoreIndx * k * BlockLen(m_sc_blocks);
    offsetB = nCoreIndx * BlockLen(n_sc_blocks);
    offsetC = mCoreIndx * n * BlockLen(m_sc_blocks) + nCoreIndx * BlockLen(n_sc_blocks);

    int a_buffer_size = BlockSize(msec_o_blocks * k_o_iter_blocks);
    int b_buffer_size = BlockSize(nsec_o_blocks * k_o_iter_blocks);
    int c_buffer_size = BlockSize(msec_o_blocks * nsec_o_blocks);

    AscendC::GlobalTensor<half> aGM;
    aGM.SetGlobalBuffer((__gm__ half *)a);
    AscendC::TQue<AscendC::TPosition::A1, 1> inQueueA1;
    pipe.InitBuffer(inQueueA1, 2, a_buffer_size * sizeof(half));

    AscendC::GlobalTensor<half> bGM;
    bGM.SetGlobalBuffer((__gm__ half *)b);
    AscendC::TQue<AscendC::TPosition::A1, 1> inQueueB1;
    pipe.InitBuffer(inQueueB1, 2, b_buffer_size * sizeof(half));

    AscendC::GlobalTensor<float> cGM;
    cGM.SetGlobalBuffer((__gm__ float *)c);

    AscendC::TQue<AscendC::TPosition::CO1, 1> outQueueCO1;
    pipe.InitBuffer(outQueueCO1, 1, c_buffer_size * sizeof(float));
    AscendC::LocalTensor<float> c1Local = outQueueCO1.AllocTensor<float>();

    AscendC::TQue<AscendC::TPosition::A2, 1> inQueueA2;
    pipe.InitBuffer(inQueueA2, 1, BlockSize(msec_blocks * db_blocks) * sizeof(half));
    AscendC::TQue<AscendC::TPosition::B2, 1> inQueueB2;
    pipe.InitBuffer(inQueueB2, 1, BlockSize(db_blocks * nsec_blocks) * sizeof(half));

    AscendC::LocalTensor<half> a1Local, b1Local, a1, b1;
    AscendC::LocalTensor<half> a2Local, b2Local, a2, b2;

    AscendC::Nd2NzParams nd2nzParams;
    AscendC::LoadData2DParams loadDataParams;

    uint32_t dstOffset, srcOffset;
    AscendC::MmadParams mmadParams;
    AscendC::FixpipeParamsV220 fixpipeParams;

    for (uint32_t mi = 0; mi < m_parts; mi++)
    {{
        if (is_last_m && (mi == m_parts - 1))
        {{
            msec_blocks = r_m_blocks;
            m_fix = m_o_fix;
        }}
        else
        {{
            msec_blocks = msec_o_blocks;
            m_fix = 0;
        }}

        for (uint32_t ni = 0; ni < n_parts; ni++)
        {{
            if (is_last_n && (ni == n_parts - 1))
            {{
                nsec_blocks = r_n_blocks;
                n_fix = n_o_fix;
            }}
            else
            {{
                nsec_blocks = nsec_o_blocks;
                n_fix = 0;
            }}

            init_zero = true;
            for (uint32_t ki = 0; ki < k_iters; ki ++)
            {{
                if (ki == (k_iters - 1))
                {{
                    k_iter_blocks = (db_o_num - 1) * db_o_blocks + r_k_blocks;
                    db_num = r_db_num;
                }}
                else
                {{
                    k_iter_blocks = k_o_iter_blocks;
                    db_num = db_o_num;
                }}

                a1Local = inQueueA1.AllocTensor<half>();
                a_offset = CalcAOffset(mi, k, msec_o_blocks, ki, k_o_iter_blocks);
                nd2nzParams.ndNum = 1;
                nd2nzParams.nValue = BlockLen(msec_blocks);
                nd2nzParams.dValue = BlockLen(k_iter_blocks);
                nd2nzParams.srcNdMatrixStride = 0;
                nd2nzParams.srcDValue = k;
                nd2nzParams.dstNzC0Stride = BlockLen(msec_blocks);
                nd2nzParams.dstNzNStride = 1;
                nd2nzParams.dstNzMatrixStride = 0;
                AscendC::DataCopy(a1Local, aGM[offsetA + a_offset], nd2nzParams);

                b1Local = inQueueB1.AllocTensor<half>();
                b_offset = CalcBOffset(ni, n, k_o_iter_blocks, ki, nsec_o_blocks);
                nd2nzParams.ndNum = 1;
                nd2nzParams.nValue = BlockLen(k_iter_blocks);
                nd2nzParams.dValue = BlockLen(nsec_blocks);
                nd2nzParams.srcNdMatrixStride = 0;
                nd2nzParams.srcDValue = n;
                nd2nzParams.dstNzC0Stride = BlockLen(k_iter_blocks);
                nd2nzParams.dstNzNStride = 1;
                nd2nzParams.dstNzMatrixStride = 0;
                AscendC::DataCopy(b1Local, bGM[offsetB + b_offset], nd2nzParams);

                inQueueA1.EnQue(a1Local);
                inQueueB1.EnQue(b1Local);

                a1 = inQueueA1.DeQue<half>();
                b1 = inQueueB1.DeQue<half>();

                for (int k = 0; k < db_num; k++)
                {{
                    if (k == (db_num - 1))
                    {{
                        db_blocks = r_k_blocks;
                        k_fix = k_o_fix;
                    }}
                    else
                    {{
                        db_blocks = db_o_blocks;
                        k_fix = 0;
                    }}

                    a2Local = inQueueA2.AllocTensor<half>();
                    b2Local = inQueueB2.AllocTensor<half>();

                    dstOffset = BlockSize(db_blocks);
                    srcOffset = BlockSize(k * db_o_blocks * msec_o_blocks);

                    // Nz -> Zz
                    loadDataParams.repeatTimes = db_blocks;
                    loadDataParams.srcStride = msec_blocks;
                    loadDataParams.dstGap = 0;
                    loadDataParams.ifTranspose = false;
                    for(int j = 0; j < msec_blocks; ++j)
                    {{
                        AscendC::LoadData(a2Local[j * dstOffset], a1[BlockSize(j) + srcOffset], loadDataParams);
                    }}
                    inQueueA2.EnQue<half>(a2Local);

                    dstOffset = BlockSize(nsec_blocks);
                    srcOffset = BlockSize(k * db_o_blocks);

                    // Nz -> Zn
                    loadDataParams.repeatTimes = nsec_blocks;
                    loadDataParams.srcStride = (db_num - 1) * db_o_blocks + db_blocks;
                    loadDataParams.dstGap = 0;
                    loadDataParams.ifTranspose = true;
                    for(int j = 0; j < db_blocks; j++)
                    {{
                        AscendC::LoadData(b2Local[j * dstOffset], b1[BlockSize(j) + srcOffset], loadDataParams);
                    }}
                    inQueueB2.EnQue<half>(b2Local);

                    a2 = inQueueA2.DeQue<half>();
                    b2 = inQueueB2.DeQue<half>();

                    mmadParams.m = BlockLen(msec_blocks) - m_fix;
                    mmadParams.n = BlockLen(nsec_blocks) - n_fix;
                    mmadParams.k = BlockLen(db_blocks) - k_fix;

                    if(!init_zero)
                    {{
                        mmadParams.cmatrixInitVal = false;
                        mmadParams.cmatrixSource = false;
                    }}
                    else
                    {{
                        mmadParams.cmatrixInitVal = true;
                        mmadParams.cmatrixSource = true;
                    }}

                    AscendC::Mmad(c1Local, a2, b2, mmadParams);
                    if(init_zero)
                    {{
                        init_zero = false;
                    }}

                    inQueueA2.FreeTensor(a2);
                    inQueueB2.FreeTensor(b2);
                }}

                inQueueA1.FreeTensor(a1);
                inQueueB1.FreeTensor(b1);
            }}

            outQueueCO1.EnQue<float>(c1Local);
            c1Local = outQueueCO1.DeQue<float>();
            fixpipeParams.nSize = BlockLen(nsec_blocks) - n_fix;
            fixpipeParams.mSize = BlockLen(msec_blocks) - m_fix;
            fixpipeParams.srcStride = BlockLen(msec_blocks);
            fixpipeParams.dstStride = n;

            fixpipeParams.ndNum = 1;
            fixpipeParams.srcNdStride = 0;
            fixpipeParams.dstNdStride = 0;
            
            AscendC::Fixpipe(
                cGM[offsetC + mi * BlockLen(msec_o_blocks) * n + ni * BlockLen(nsec_o_blocks)], 
                c1Local, 
                fixpipeParams);
        }}
    }}

    outQueueCO1.FreeTensor(c1Local);
}}
)";
}
}