// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "rans_c.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

extern "C" void dcvc_pmf_to_quantized_cdf(const float* pmf, int pmf_n, int precision,
                                          uint32_t* out_cdf)
{
    std::vector<uint32_t> cdf(static_cast<size_t>(pmf_n) + 1);
    cdf[0] = 0;
    for (int i = 0; i < pmf_n; i++) {
        cdf[static_cast<size_t>(i) + 1] =
            static_cast<uint32_t>(std::round(pmf[i] * (1 << precision)) + 0.5);
    }

    const uint32_t total = std::accumulate(cdf.begin(), cdf.end(), 0u);
    for (size_t i = 0; i < cdf.size(); i++) {
        cdf[i] = static_cast<uint32_t>((((1ull << precision) * cdf[i]) / total));
    }
    std::partial_sum(cdf.begin(), cdf.end(), cdf.begin());
    cdf.back() = 1u << precision;

    for (int i = 0; i < static_cast<int>(cdf.size()) - 1; ++i) {
        if (cdf[i] == cdf[static_cast<size_t>(i) + 1]) {
            uint32_t best_freq = ~0u;
            int best_steal = -1;
            for (int j = 0; j < static_cast<int>(cdf.size()) - 1; ++j) {
                uint32_t freq = cdf[static_cast<size_t>(j) + 1] - cdf[j];
                if (freq > 1 && freq < best_freq) {
                    best_freq = freq;
                    best_steal = j;
                }
            }
            assert(best_steal != -1);
            if (best_steal < i) {
                for (int j = best_steal + 1; j <= i; ++j) cdf[j]--;
            } else {
                for (int j = i + 1; j <= best_steal; ++j) cdf[j]++;
            }
        }
    }

    std::memcpy(out_cdf, cdf.data(), cdf.size() * sizeof(uint32_t));
}
