#include "kernel_operator.h"

#define CUBE_BLOCK          16   // block的行列数，单位元素个数  
#define CUBE_BLOCK_SIZE     256  // block的总大小，单位元素个数

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
    /**
      每个核吃掉几组 循环吃掉 m_sec_o_blocks , k_o_iter_blocks
                    a_offset = CalcAOffset(mi, k, m_sec_o_blocks, ki, k_o_iter_blocks);
    */
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
    
    // 参数paramGM
    AscendC::GlobalTensor<uint32_t> paramGM;
    paramGM.SetGlobalBuffer((__gm__ uint32_t *)params);

    /** step-1：获取参数 */ 
    // step-1.1 - shape m、n、k
    uint32_t m = paramGM.GetValue(0);
    uint32_t n = paramGM.GetValue(1);
    uint32_t k = paramGM.GetValue(2);

    // step-1.2 - 作用： 参与计算的 AIcore 数量、单次GM->L1 (DataCopy) 挖取参与计算的cube unit数据量参数。
    // m_sections 左矩阵沿m轴横切几块 m_sections * n_sections = blockDims即参与并行计算matmul的AiCores个数。
    uint32_t m_sections = paramGM.GetValue(3); 
    // n_sections 右矩阵沿n轴竖切几块
    uint32_t n_sections = paramGM.GetValue(4); 
    // L1（A1）单次挖取参与计算的全量左矩阵上的小矩阵（cube unit-长方形）的边跨度，边长（即元素个数） = m_sec_o_blocks * CUBE_BLOCK（当前为16个元素）
    uint32_t m_sec_o_blocks = paramGM.GetValue(5); 
    // L1（B1）单次挖取参与计算的全量右矩阵上的小矩阵（cube unit-长方形）的边跨度，边长（即元素个数） = n_sec_o_blocks * CUBE_BLOCK（当前为16个元素）
    uint32_t n_sec_o_blocks = paramGM.GetValue(6); 
    // 作用： 单次挖取的左小矩阵元素个数 = (CUBE_BLOCK * m_sec_o_blocks) * (CUBE_BLOCK * k_o_iter_blocks）
    // 单次挖取的右小矩阵元素个数 = (CUBE_BLOCK * k_o_iter_blocks） * (CUBE_BLOCK * n_sec_o_blocks)
    uint32_t k_o_iter_blocks = paramGM.GetValue(7); 
    
    // step-1.3 - 作用：对单次通过DataCopy挖取的小矩阵沿K轴再进行垂直切割（Split），切割单位为 db_o_blocks *  CUBE_BLOCK（当前为16个元素）
    uint32_t db_o_blocks = paramGM.GetValue(8);
    // batch大小
    uint32_t batch = paramGM.GetValue(9);
    // step-1.4 - 
    // 作用：单核分配到的子矩阵 - 即核间切分
    // 左矩阵：分配到左子矩阵 宽度为 m_parts *  m_sec_o_blocks * CUBE_BLOCK 
    // 右矩阵：分配到的右子矩阵 长度为 n_parts * n_sec_o_blocks * CUBE_BLOCK 
    // m_parts = m_blocks / m_sec_o_blocks / m_sections 
    uint32_t m_parts = paramGM.GetValue(20); 
    uint32_t n_parts = paramGM.GetValue(21);
    uint32_t k_iters = paramGM.GetValue(10);
    // padding后左矩阵 m 按照 CUBE_BLOCK 分组 [align(m) / 16]可以分 m_blocks 组  
    uint32_t m_blocks = paramGM.GetValue(11);
    uint32_t n_blocks = paramGM.GetValue(12);
    uint32_t k_blocks = paramGM.GetValue(13);
    // 单核处理的左矩阵宽 m_parts * m_sec_o_blocks  
    uint32_t m_sc_blocks = paramGM.GetValue(14);
    // 单核处理的右矩阵长 n_parts * n_sec_o_blocks
    uint32_t n_sc_blocks = paramGM.GetValue(15);
    
    // padding后左矩阵m轴补充m_o_fix元素
    uint32_t m_o_fix = paramGM.GetValue(16);
    // padding后右矩阵n轴补充n_o_fix元素
    uint32_t n_o_fix = paramGM.GetValue(17);
    // padding后左右矩阵k轴补充 k_o_fix 元素
    uint32_t k_o_fix = paramGM.GetValue(18);
    // k_o_iter_blocks / db_o_blocks 影响L1->L2搬运次数。
    uint32_t db_o_num = paramGM.GetValue(19);
    // 最后一个 aicore 在m轴迭代的次数
    uint32_t r_m_parts = paramGM.GetValue(22);
    // 最后一个aicore 在n轴迭代的次数
    uint32_t r_n_parts = paramGM.GetValue(23);
    // remain_m_blocks 最后一个 aicore 处理的m轴边角料跨度 
    uint32_t r_m_blocks = paramGM.GetValue(24);
    // remain_n_blocks 最后一个 aicore 处理的n轴边角料跨度
    uint32_t r_n_blocks = paramGM.GetValue(25);
    // remain_k_blocks 每个核处理的k轴边角料跨度， 元素个数 r_k_blocks * CUBE_BLOCK
    uint32_t r_k_blocks = paramGM.GetValue(26);
    // remain_db_num 每个核处理的k轴边角料被搬运次数
    uint32_t r_db_num = paramGM.GetValue(27);
    
    // step-1.5 - 每个核在单次迭代计算中 挖取的计算矩阵大小  
    // 左矩阵：a_buffer_size - GM->L1 
    // 右矩阵：b_buffer_size - GM->L1
    // 输出矩阵：c_buffer_size - CO1 -> CM 
    int a_buffer_size = BlockSize(m_sec_o_blocks * k_o_iter_blocks);
    int b_buffer_size = BlockSize(n_sec_o_blocks * k_o_iter_blocks);
    int c_buffer_size = BlockSize(m_sec_o_blocks * n_sec_o_blocks);
    
    // 初始化 GlobalTensor 类型的 aGM变量 
    AscendC::GlobalTensor<bfloat16_t> aGM;
    // 左矩阵数据填充至aGM
    aGM.SetGlobalBuffer((__gm__ bfloat16_t *)a);

    // 初始化 GlobalTensor 类型的 bGM变量
    AscendC::GlobalTensor<bfloat16_t> bGM;
    // 右矩阵数据填充至bGM
    bGM.SetGlobalBuffer((__gm__ bfloat16_t *)b);
    
    // 初始化 GlobalTensor 类型的 cGM变量
    AscendC::GlobalTensor<float> cGM;
    // 左右矩阵matmul后得到的 output tensor 
    cGM.SetGlobalBuffer((__gm__ float *)c);
    
    /** step2 搬运流水初始化 */
    // 队列 GM -> L1 [position, depth] 中间连续n次enque, 队列机制用来实现流水线并行， db 在这个基础上进一步提高流水线利用率，编译器对这种场景做了特殊优化，性能比较好，推荐设置为1
    AscendC::TQue<AscendC::TPosition::A1, 1> inQueueA1;
    // 每个A1分配出2块内存，每块大小为c_buffer_size * 2字节（float）
    pipe.InitBuffer(inQueueA1, 2, a_buffer_size * sizeof(bfloat16_t)); 
    AscendC::TQue<AscendC::TPosition::B1, 1> inQueueB1;
    pipe.InitBuffer(inQueueB1, 2, b_buffer_size * sizeof(bfloat16_t));
    // 小矩阵计算结果写入到 CO1位置，大小为 c_buffer_size * 4bytes
    AscendC::TQue<AscendC::TPosition::CO1, 1> outQueueCO1;
    pipe.InitBuffer(outQueueCO1, 1, c_buffer_size * sizeof(float)); 
    // 数据流向 L1 -> L2 用于split 小矩阵 进行更小粒度计算前的数据传输
    AscendC::TQue<AscendC::TPosition::A2, 1> inQueueA2;
    pipe.InitBuffer(inQueueA2, 1, BlockSize(m_sec_o_blocks * db_o_blocks) * sizeof(bfloat16_t));
    AscendC::TQue<AscendC::TPosition::B2, 1> inQueueB2;
    pipe.InitBuffer(inQueueB2, 1, BlockSize(db_o_blocks * n_sec_o_blocks) * sizeof(bfloat16_t));
    // [[l1, r1],[l2, r2],[l3, r3]] 则 batch = 3， 相当于做[batch, m, k] * [batch, k, n] 两个张量乘法 -> [batch, m, n ]
    for(uint32_t bi = 0; bi < batch; bi++)
    {  
        // 每组左矩阵首个元素起始索引
        uint32_t offsetA = bi * m * k;
        // 每组右矩阵首个元素起始索引
        uint32_t offsetB = bi * n * k;
        // 每组输出矩阵首个元素起始索引
        uint32_t offsetC = bi * m * n;
        // DataCopy 操作 - 左右矩阵 GM -> L1 tensor 的初始索引
        uint32_t a_offset, b_offset;
        // 当前核搬运的左右子矩阵(参与计算)大小， 
        // 左子矩阵大小  msec_blocks * k_iter_blocks * CUBE_BLOCK_SIZE(256个元素)  
        // 右子矩阵大小  nsec_blocks * k_iter_blocks * CUBE_BLOCK_SIZE(256个元素)
        uint32_t msec_blocks, nsec_blocks, k_iter_blocks, db_blocks;
        // m、n、k轴尾部padding补充多少元素 
        uint32_t m_fix, n_fix, k_fix;
        // 切分后矩阵在K轴循环次数
        uint32_t db_num;
        // 当前AiCore ID
        uint32_t blockIdx = AscendC::GetBlockIdx();
        // 判断当前核是否处理最后一部分子矩阵
        // m_sections = 3
        // n_sections = 4 
        // 11 % 3 = 2 
        uint32_t mCoreIndx = blockIdx % m_sections;
        // 11 / 3 = 3 
        uint32_t nCoreIndx = blockIdx / m_sections;
        bool is_last_m = mCoreIndx == (m_sections - 1);
        bool is_last_n = nCoreIndx == (n_sections - 1);
        // 如果当前核处理最后左矩阵左右一块 子矩阵， m轴遍历 m_parts 次每次遍历计算读m_sec_o_blocks 块
        if (is_last_m)
        {
            m_parts = r_m_parts;
        }
        if (is_last_n)
        {
            n_parts = r_n_parts;
        }
        
        // 核间切分 当前核处理的左子矩阵 首个元素最先开始的索引
        offsetA += mCoreIndx * k * BlockLen(m_sc_blocks);
        // 核间切分 当前核处理的右子矩阵 首个元素最先开始的偏移
        offsetB += nCoreIndx * BlockLen(n_sc_blocks);
        // 输出结果偏移位置
        offsetC += mCoreIndx * n * BlockLen(m_sc_blocks) + nCoreIndx * BlockLen(n_sc_blocks);
        
        // xlLocal copydata 的dst tensor 
        // x1，b1 从 inQueuex1 中 Deque 的 tensor 
        AscendC::LocalTensor<bfloat16_t> a1Local, b1Local, a1, b1;
        AscendC::LocalTensor<bfloat16_t> a2Local, b2Local, a2, b2;
        // 初始化 ND2NZ Params 变量， GM -> L1（A1/B1）的拷贝在大多数情况下会存在随路转换，数据物理存储格式会发生改变
        AscendC::Nd2NzParams nd2nzParams;
        // 初始化 loadDataParams， L1（A1/B1） -> L2（A2/B2） 数据加载 （LoadData）过程中会伴随随路转换的参数
        AscendC::LoadData2DParams loadDataParams;
        // 目标数据偏移、源数据偏移
        uint32_t dstOffset, srcOffset;
        // AscendC::mmad 指令计算api调用参数
        AscendC::MmadParams mmadParams;
        // 计算结果搬运 参数
        AscendC::FixpipeParamsV220 fixpipeParams;
        // 左子矩阵遍历m_parts次，每次读取 msec_blocks 跨度宽
        for (uint32_t mi = 0; mi < m_parts; mi++)
        {
            // 判断是核为最后一个核，且处理左子矩阵最后一块
            if (is_last_m && (mi == m_parts - 1))
            {   
                // 剩余处理块跨度赋值到当前核处理的块跨度
                msec_blocks = r_m_blocks;
                // padding 数据个数赋值给m_fix
                m_fix = m_o_fix;
            }
            else
            {
                // 在catlass一些框架中被命名为 mTile 即 m_sec_o_blocks。 
                msec_blocks = m_sec_o_blocks;
                // 非尾块，则为0
                m_fix = 0;
            }
            // 当前核对右子矩阵遍历n_parts次，每次读取 nsec_blocks 跨度长
            for (uint32_t ni = 0; ni < n_parts; ni++)
            {   
                 // 判断当前核核为处理最后一部分子矩阵的核，且处理右子矩阵最后一块
                if (is_last_n && (ni == n_parts - 1))
                {   
                    // 剩余处理块跨度赋值到当前核处理的块跨度
                    nsec_blocks = r_n_blocks;
                    // 沿n轴 padding 数据个数赋值给n_fix
                    n_fix = n_o_fix;
                }
                else
                {  
                    // 在catlass一些框架中被命名为 nTile 即 n_sec_o_blocks。 
                    nsec_blocks = n_sec_o_blocks;
                    // 非尾块则为0
                    n_fix = 0;
                }
                
                // 初次计算设置为true
                init_zero = true;
                //  存储子矩阵运算结果的c1Local localTensor 
                AscendC::LocalTensor<float> c1Local = outQueueCO1.AllocTensor<float>();
                
                // 沿左右矩阵 的K轴按照单次迭代 k_o_iter_blocks 块跨度k_iters次
                for (uint32_t ki = 0; ki < k_iters; ki ++)
                {   
                    // 当前核处理k轴最后一块
                    if (ki == (k_iters - 1))
                    {   
                        // k轴 remain部分的块跨度
                        k_iter_blocks = (r_db_num - 1) * db_o_blocks + r_k_blocks;
                        // L1（A1/B1）中子矩阵被切分成A2/B2更小矩阵计算的个数
                        db_num = r_db_num;
                    }
                    else
                    { 
                        // 当前核未处理到K轴最后一部分
                        k_iter_blocks = k_o_iter_blocks;
                        db_num = db_o_num;
                    }
                
                    a1Local = inQueueA1.AllocTensor<bfloat16_t>();
                    // 每个核循环吃掉 m_sec_o_blocks , k_o_iter_blocks
                    a_offset = CalcAOffset(mi, k, m_sec_o_blocks, ki, k_o_iter_blocks);
                    if (BlockLen(msec_blocks) - m_fix == 1) {//假如baseM = 1,连续搬16个元素
                        AscendC::DataCopy(a1Local, aGM[offsetA + a_offset], CeilCubeBlock(k) * CUBE_BLOCK);
                    } else {// notice 随路格式转换，atlasascendc_api_07_0105.html
                        nd2nzParams.ndNum = 1; // 每次只需要搬一个矩阵
                        nd2nzParams.nValue = BlockLen(msec_blocks); // 矩阵大小为 nValue * dValue 每次搬运量 m * k 
                        nd2nzParams.dValue = BlockLen(k_iter_blocks); 
                        nd2nzParams.srcNdMatrixStride = 0; // 只有1个矩阵，不需要偏移
                        nd2nzParams.srcDValue = k; // 每行偏移为k
                        nd2nzParams.dstNzC0Stride = BlockLen(msec_blocks); // 转换后，每行为BlockLen(msec_blocks) * CUBE_BLOCK个数据,  ND转换到NZ格式候，源操作数中的一行会转换为目的操作数的多行，多行其实地址之间的便宜就是A1和A2在dst中的偏移，偏移为11个datablock
                        nd2nzParams.dstNzNStride = 1;// 代表dst中一个ndMatrix，src中的第x行和第x+1行之间的偏移，即A1和B1之间的距离，即为2个datablock。
                        nd2nzParams.dstNzMatrixStride = 0;
                        AscendC::DataCopy(a1Local, aGM[offsetA + a_offset], nd2nzParams);
                    }
                    // 随路转换参数-优化点1 可以使用aiv core 左随路转换
                    b1Local = inQueueB1.AllocTensor<bfloat16_t>();
                    b_offset = CalcBOffset(ni, n, k_o_iter_blocks, ki, n_sec_o_blocks);
                    nd2nzParams.ndNum = 1;
                    nd2nzParams.nValue = BlockLen(k_iter_blocks);
                    nd2nzParams.dValue = BlockLen(nsec_blocks);
                    nd2nzParams.srcNdMatrixStride = 0;
                    nd2nzParams.srcDValue = n;
                    nd2nzParams.dstNzC0Stride = BlockLen(k_iter_blocks);
                    nd2nzParams.dstNzNStride = 1;
                    nd2nzParams.dstNzMatrixStride = 0;
                    // 从右矩阵copy数据到b1Local.
                    AscendC::DataCopy(b1Local, bGM[offsetB + b_offset], nd2nzParams);
                    // 对应 atlas_ascendc_10_00006.html
                    // x1Local分别入 inQueueA1、inQueueB1 队列
                    inQueueA1.EnQue(a1Local);
                    inQueueB1.EnQue(b1Local);
                    // 对应左上 DeQue， 出队数据参与后续计算。
                    a1 = inQueueA1.DeQue<bfloat16_t>();
                    b1 = inQueueB1.DeQue<bfloat16_t>();
                    // A1/B1 -> A2/B2  再次split,切割成更小矩阵进行计算。
                    for (int kii = 0; kii < db_num; kii++)
                    {
                        // 对尾块的处理
                        if ((ki == (k_iters - 1)) && kii == (db_num - 1))
                        {
                            db_blocks = r_k_blocks;
                            k_fix = k_o_fix;
                        }
                        else
                        {
                            db_blocks = db_o_blocks;
                            k_fix = 0;
                        }
                        // 初始化 a2Local, b2Local    
                        a2Local = inQueueA2.AllocTensor<bfloat16_t>();
                        b2Local = inQueueB2.AllocTensor<bfloat16_t>();
                        // 目标偏移
                        dstOffset = BlockSize(db_blocks); // sub_k_o_iter_blocks = 10 * 256(CUBE_BLOCK_SIZE)
                        if (BlockLen(msec_blocks) - m_fix == 1) { // msec_blocks = 1 的情况
                            srcOffset = BlockLen(kii * db_o_blocks * msec_blocks); // 一行 
                        } else {
                            // msec_blocks != 1 跨度超过16个元素
                            srcOffset = BlockSize(kii * db_o_blocks * (mi == (m_parts - 1) ? msec_blocks : m_sec_o_blocks)); // 1块
                        }

                        // Nz -> Zz
                        loadDataParams.repeatTimes = db_blocks;
                        loadDataParams.srcStride = msec_blocks;
                        loadDataParams.dstGap = 0;
                        loadDataParams.ifTranspose = false;
                        for(int j = 0; j < msec_blocks; ++j)
                        {
                            AscendC::LoadData(a2Local[j * dstOffset], a1[BlockSize(j) + srcOffset], loadDataParams); // 
                        }
                        inQueueA2.EnQue<bfloat16_t>(a2Local);

                        dstOffset = BlockSize(nsec_blocks);
                        srcOffset = BlockSize(kii * db_o_blocks);

                        // Nz -> Zn
                        loadDataParams.repeatTimes = nsec_blocks;
                        loadDataParams.srcStride = k_iter_blocks;
                        loadDataParams.dstGap = 0;
                        loadDataParams.ifTranspose = true;
                        for(int j = 0; j < db_blocks; j++)
                        {   // 从b1 loadData加载数据到 b2 
                            AscendC::LoadData(b2Local[j * dstOffset], b1[BlockSize(j) + srcOffset], loadDataParams);
                        }
                        // b2Local 入队
                        inQueueB2.EnQue<bfloat16_t>(b2Local);
                        // 入队数据出队
                        a2 = inQueueA2.DeQue<bfloat16_t>();
                        // 入队数据出队 
                        b2 = inQueueB2.DeQue<bfloat16_t>();
                        // 参与matmul计算的最小矩阵 shape参数 
                        mmadParams.m = BlockLen(msec_blocks) - m_fix;
                        mmadParams.n = BlockLen(nsec_blocks) - n_fix;
                        mmadParams.k = BlockLen(db_blocks) - k_fix;
                        
                        if(!init_zero)
                        {
                            mmadParams.cmatrixInitVal = false;
                            mmadParams.cmatrixSource = false;
                        }
                        else
                        {
                            mmadParams.cmatrixInitVal = true;
                            mmadParams.cmatrixSource = true;
                        }
                        // 执行mmad操作， a2 @ b2 并更新c1Local 具体位置的值
                        AscendC::Mmad(c1Local, a2, b2, mmadParams);
                        if(init_zero)
                        {
                            init_zero = false;
                        }
                        // 清空a2内存空间
                        inQueueA2.FreeTensor(a2);
                        // 清空b2内存空间
                        inQueueB2.FreeTensor(b2);
                    }
                    // 清空 a1 localtensor 内存空间
                    inQueueA1.FreeTensor(a1);
                    // 清空 b1 localtensor 内存空间
                    inQueueB1.FreeTensor(b1);
                }
                // 对应右上 CO1 queue 
                outQueueCO1.EnQue<float>(c1Local);
                c1Local = outQueueCO1.DeQue<float>();
                // 讲C1Local计算值刷新到 cGM中。
                fixpipeParams.nSize = BlockLen(nsec_blocks) - n_fix;
                fixpipeParams.mSize = BlockLen(msec_blocks) - m_fix;
                fixpipeParams.srcStride = BlockLen(msec_blocks);
                fixpipeParams.dstStride = n;

                fixpipeParams.ndNum = 1;
                fixpipeParams.srcNdStride = 0;
                fixpipeParams.dstNdStride = 0;

                AscendC::Fixpipe(
                    cGM[offsetC + mi * BlockLen(m_sec_o_blocks) * n + ni * BlockLen(n_sec_o_blocks)],
                    c1Local,
                    fixpipeParams);
                outQueueCO1.FreeTensor(c1Local);
            }
        }
    }
}