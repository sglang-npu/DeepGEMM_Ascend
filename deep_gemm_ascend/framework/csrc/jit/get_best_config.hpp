#include <iostream>
#include <cmath>
#include <cstdint>

uint32_t align16(uint32_t x) {
    return (x + 15) & ~15;
}

struct Args {
    int k_iters;
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

Args get_best_config(uint32_t batch, uint32_t m, uint32_t n, uint32_t k) {
    Args args;

    args.m_sections = 1;
    args.n_sections = 1;
    args.m_blocks = align16(m) / 16;
    args.n_blocks = align16(n) / 16;
    args.k_blocks = align16(k) / 16;
    args.m_o_fix = align16(m) - m;
    args.n_o_fix = align16(n) - n;
    args.k_o_fix = align16(k) - k;

    args.m_sec_o_blocks = 3;
    args.n_sec_o_blocks = 8;
    args.k_o_iter_blocks = 20;
    args.db_o_blocks = 10;
    args.db_o_num = args.k_o_iter_blocks / args.db_o_blocks;

    args.r_m_blocks = args.m_blocks % args.m_sec_o_blocks;
    args.r_n_blocks = args.n_blocks % args.n_sec_o_blocks;

    args.k_iters = (args.k_blocks + args.k_o_iter_blocks - 1) / args.k_o_iter_blocks;
    uint32_t k_tail_blocks = args.k_blocks % args.k_o_iter_blocks;
    args.r_k_blocks = k_tail_blocks % args.db_o_blocks;
    args.r_db_num = (k_tail_blocks + args.db_o_blocks - 1) / args.db_o_blocks;

    uint32_t m_iters = (args.m_blocks + args.m_sec_o_blocks - 1) / args.m_sec_o_blocks;
    uint32_t n_iters = (args.n_blocks + args.n_sec_o_blocks - 1) / args.n_sec_o_blocks;

    args.m_parts = (m_iters + args.m_sections - 1) / args.m_sections;
    args.n_parts = (n_iters + args.n_sections - 1) / args.n_sections;

    args.m_sc_blocks = args.m_parts * args.m_sec_o_blocks;
    args.n_sc_blocks = args.n_parts * args.n_sec_o_blocks;

    args.r_m_parts = m_iters - ((args.m_sections - 1) * args.m_parts);
    args.r_n_parts = n_iters - ((args.n_sections - 1) * args.n_parts);

    args.batch = batch;
    args.m = m;
    args.n = n;
    args.k = k;

    return args;
}