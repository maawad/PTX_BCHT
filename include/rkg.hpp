#pragma once
#include <fstream>
#include <helpers.hpp>
#include <random>
#include <string>
#include <unordered_set>

namespace rkg {
template <typename key_type, typename size_type>
void generate_uniform_unique_keys(std::vector<key_type>& keys,
                                  size_type num_keys,
                                  key_type min_key = 0,
                                  unsigned seed    = 0) {
  keys.resize(num_keys);
  std::random_device rd;
  std::mt19937 rng(seed);
  auto max_key = std::numeric_limits<key_type>::max() - 1;
  std::uniform_int_distribution<key_type> uni(min_key, max_key);
  std::unordered_set<key_type> unique_keys;
  while (unique_keys.size() < num_keys) { unique_keys.insert(uni(rng)); }
  std::copy(unique_keys.cbegin(), unique_keys.cend(), keys.begin());
  std::shuffle(keys.begin(), keys.end(), rng);
}

template <typename key_type, typename size_type>
void prep_experiment_find_with_exist_ratio(float exist_ratio,
                                           size_type num_keys,
                                           const std::vector<key_type>& keys,
                                           std::vector<key_type>& find_keys) {
  assert(num_keys * 2 == keys.size());
  unsigned int end_index                = num_keys * (-exist_ratio + 2);
  unsigned int start_index              = end_index - num_keys;
  static constexpr uint32_t EMPTY_VALUE = 0xFFFFFFFF;
  std::fill(find_keys.begin(), find_keys.end(), EMPTY_VALUE);
  std::copy(keys.begin() + start_index, keys.begin() + end_index, find_keys.begin());
}
}  // namespace rkg