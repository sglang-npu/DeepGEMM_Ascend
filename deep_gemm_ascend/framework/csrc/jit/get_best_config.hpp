#ifndef GET_BEST_CONFIG_HPP
#define GET_BEST_CONFIG_HPP

#include <iostream>
#include <cmath>
#include <cstdint>

namespace deep_gemm_ascend {
uint32_t align16(uint32_t x) {
    return (x + 15) & ~15;
}

struct Config {
    uint32_t k_iters;
    uint32_t batch;

    uint32_t m, n, k;
    uint32_t m_sections, n_sections;
    uint32_t m_blocks, n_blocks, k_blocks;
    uint32_t m_sc_blocks, n_sc_blocks;

    uint32_t m_sec_o_blocks, n_sec_o_blocks, k_o_iter_blocks, db_o_blocks;

    uint32_t m_o_fix, n_o_fix, k_o_fix;
    uint32_t db_o_num;

    uint32_t m_parts, n_parts;
    uint32_t r_m_parts, r_n_parts;

    uint32_t r_m_blocks, r_n_blocks, r_k_blocks;
    uint32_t r_db_num;
};

Config get_best_config(uint32_t batch, uint32_t m, uint32_t n, uint32_t k) {
    Config args;

    // todo 调参
    args.m_sections = 1;
    args.n_sections = 1;
    args.m_blocks = align16(m) / 16;
    args.n_blocks = align16(n) / 16;
    args.k_blocks = align16(k) / 16;
    args.m_o_fix = align16(m) - m; // 看m轴补了多少个数字
    args.n_o_fix = align16(n) - n; // 看n轴补了多少
    args.k_o_fix = align16(k) - k; // 看k轴补了多少

    // todo - 有约束 byte 大小，这些block是长方形
    args.m_sec_o_blocks = 3; // m轴上每次搬运 3个block成为被搬运的一组。理想情况下
    args.n_sec_o_blocks = 8; // n轴上每次搬运 （8个block成为被搬运的一组）。完整的block
    args.k_o_iter_blocks = 20; // k轴上每次搬运20个block - （20个block为一组）。
    args.db_o_blocks = 10; // db -> double buffer, 第一次搬运 GM -> L2 -> L1 没办法直接从L2 -> L1, 限制，
    args.db_o_num = args.k_o_iter_blocks / args.db_o_blocks; // 理论上从L2 -> L1 需要搬2次。

    args.r_m_blocks = args.m_blocks % args.m_sec_o_blocks; // r 是remain，边角料还剩多少块？
    args.r_n_blocks = args.n_blocks % args.n_sec_o_blocks; // r 是remain，边角料还剩多少块，这些块是不够搬一次的。

    if (args.r_m_blocks == 0) {
        args.r_m_blocks = args.m_sec_o_blocks; // 足够搬，m轴最后一块刚够搬运。
    }
    if (args.r_n_blocks == 0) {
        args.r_n_blocks = args.n_sec_o_blocks; // 足够搬，n轴最后一块刚够搬
    }

    args.k_iters = (args.k_blocks + args.k_o_iter_blocks - 1) / args.k_o_iter_blocks; // k轴上总共可以搬运（迭代iter）多少次
    uint32_t k_tail_blocks = args.k_blocks % args.k_o_iter_blocks; // k_tail_blocks 剩余边角料 的block 还剩多少 < 20
    if (k_tail_blocks == 0) {
        args.r_db_num = args.db_o_num;
        args.r_k_blocks = args.db_o_blocks;
    } else {
        args.r_db_num = (k_tail_blocks + args.db_o_blocks - 1) / args.db_o_blocks; // r_db_num 剩余的block 应被搬多少次。假如 k_tail_blocks = 1  搬运1次
        args.r_k_blocks = k_tail_blocks - ((args.r_db_num - 1) * args.db_o_blocks);
    }

    uint32_t m_iters = (args.m_blocks + args.m_sec_o_blocks - 1) / args.m_sec_o_blocks; // m轴上需要循环搬运 m_iters 次, 搬运单位 每次3个block
    uint32_t n_iters = (args.n_blocks + args.n_sec_o_blocks - 1) / args.n_sec_o_blocks; // n轴上需要循环搬运 n_iters 次。

    args.m_parts = m_iters / args.m_sections; // 每个Ai_core 在M轴上分别搬运 m_parts 次。
    args.n_parts = n_iters / args.n_sections; // 每个Ai_core 在N轴上分别搬运 n_parts 次。

    args.m_sc_blocks = args.m_parts * args.m_sec_o_blocks; // 每个 Ai core 在m轴总计搬运的blocks， m_sec_o_blocks 是跨度
    args.n_sc_blocks = args.n_parts * args.n_sec_o_blocks; // 每个 Ai core 在n轴总计搬运的blocks， n_sec_o_blocks 是跨度

    args.r_m_parts = m_iters - ((args.m_sections - 1) * args.m_parts); // 最后一个核在m轴需要搬运的次数
    args.r_n_parts = n_iters - ((args.n_sections - 1) * args.n_parts); // 最后一个核在n轴需要搬运的次数

    args.batch = batch; // batch 大小
    args.m = m;
    args.n = n;
    args.k = k;

    return args;
}

Config get_bench_config(uint32_t m, uint32_t n, uint32_t k,
    uint32_t m_sections, uint32_t n_sections,
    uint32_t m_sec_o_blocks, uint32_t n_sec_o_blocks,
    uint32_t k_o_iter_blocks, uint32_t db_o_blocks)
{
    Config args;

    args.m_sections = m_sections;
    args.n_sections = n_sections;
    args.m_blocks = align16(m) / 16;
    args.n_blocks = align16(n) / 16;
    args.k_blocks = align16(k) / 16;
    args.m_o_fix = align16(m) - m;
    args.n_o_fix = align16(n) - n;
    args.k_o_fix = align16(k) - k;

    args.m_sec_o_blocks = m_sec_o_blocks;
    args.n_sec_o_blocks = n_sec_o_blocks;
    args.k_o_iter_blocks = k_o_iter_blocks;
    args.db_o_blocks = db_o_blocks;
    args.db_o_num = args.k_o_iter_blocks / args.db_o_blocks;

    args.r_m_blocks = args.m_blocks % args.m_sec_o_blocks;
    args.r_n_blocks = args.n_blocks % args.n_sec_o_blocks;

    if (args.r_m_blocks == 0) {
        args.r_m_blocks = args.m_sec_o_blocks;
    }
    if (args.r_n_blocks == 0) {
        args.r_n_blocks = args.n_sec_o_blocks;
    }

    args.k_iters = (args.k_blocks + args.k_o_iter_blocks - 1) / args.k_o_iter_blocks;
    uint32_t k_tail_blocks = args.k_blocks % args.k_o_iter_blocks;
    if (k_tail_blocks == 0) {
        args.r_db_num = args.db_o_num;
        args.r_k_blocks = args.db_o_blocks;
    } else {
        args.r_db_num = (k_tail_blocks + args.db_o_blocks - 1) / args.db_o_blocks; // r_db_num 剩余的block 应被搬多少次。假如 k_tail_blocks = 1  搬运1次
        args.r_k_blocks = k_tail_blocks - ((args.r_db_num - 1) * args.db_o_blocks);
    }

    uint32_t m_iters = (args.m_blocks + args.m_sec_o_blocks - 1) / args.m_sec_o_blocks;
    uint32_t n_iters = (args.n_blocks + args.n_sec_o_blocks - 1) / args.n_sec_o_blocks;

    args.m_parts = m_iters / args.m_sections;
    args.n_parts = n_iters / args.n_sections;

    args.m_sc_blocks = args.m_parts * args.m_sec_o_blocks;
    args.n_sc_blocks = args.n_parts * args.n_sec_o_blocks;

    args.r_m_parts = m_iters - ((args.m_sections - 1) * args.m_parts);
    args.r_n_parts = n_iters - ((args.n_sections - 1) * args.n_parts);

    args.batch = 1;
    args.m = m;
    args.n = n;
    args.k = k;

    return args;
}
} // namespace deep_gemm_ascend

#endif