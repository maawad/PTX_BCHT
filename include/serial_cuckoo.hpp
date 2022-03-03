#pragma once
#include <assert.h>
#include <stdio.h>
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <hash_function.hpp>
#include <iostream>
#include <random>
#include <vector>

#include <cstdint>

namespace serial_bcht {

class cuckoo_hash_set {
 public:
  using key_type   = uint32_t;
  using value_type = uint32_t;
  using size_type  = uint32_t;
  using hash_type  = universal_hash<key_type>;

  static constexpr key_type empty_key = std::numeric_limits<key_type>::max();

  cuckoo_hash_set(size_type bucket_size,
                  size_type num_hfs,
                  size_type capacity,
                  float load_factor,
                  hash_type hf0,
                  hash_type hf1,
                  hash_type hf2,
                  unsigned seed = 0)
      : load_factor_(load_factor), bucket_size_(bucket_size), num_hfs_(num_hfs), num_keys_(0) {
    std::mt19937 rng(seed);
    rng_ = rng;

    hash_funs_.push_back(hf0);
    hash_funs_.push_back(hf1);
    hash_funs_.push_back(hf2);

    lf_pairs_capacity_ = size_type(std::ceil(float(capacity) / load_factor_));
    pairs_per_bucket_  = bucket_size_;
    num_buckets_       = (lf_pairs_capacity_ + pairs_per_bucket_ - 1) / pairs_per_bucket_;
    table_.resize(num_buckets_ * bucket_size_, empty_key);
    float lg_input_size           = (float)(log((double)capacity) / log(2.0));
    const unsigned max_iter_const = 7;
    max_chains_                   = max_iter_const * lg_input_size;
  }

  inline size_type determine_next_location(const key_type key, const size_type previous_location) {
    std::vector<size_type> locations(num_hfs_);
    for (size_type i = 0; i < num_hfs_; ++i) { locations[i] = hash_funs_[i](key) % num_buckets_; }
    size_type next_location = locations[0];
    for (int i = int(num_hfs_) - 2; i >= 0; --i) {
      next_location = (previous_location == locations[i] ? locations[i + 1] : next_location);
    }
    return next_location;
  }
  inline bool insert(const key_type key) {
    key_type cur_key     = key;
    size_type num_chains = 0;
    auto cur_bucket      = hash_funs_[0](cur_key) % num_buckets_;
    while (num_chains < max_chains_) {
      auto bucket_count = count_bucket(cur_bucket);
      if (bucket_count < pairs_per_bucket_) {
        set_key_at(cur_bucket, bucket_count, cur_key);
        num_keys_++;
        return true;
      } else {
        auto rand_location = rng_() % pairs_per_bucket_;
        cur_key            = swap_key_at(cur_bucket, rand_location, cur_key);
        cur_bucket         = determine_next_location(cur_key, cur_bucket);
        num_chains++;
      }
    }
    return false;
  }

  inline bool insert(const std::vector<key_type>& keys) {
    for (const auto& key : keys) {
      if (!insert(key)) { return false; }
    }
    return true;
  }

  inline float compute_load_factor() const { return float(num_keys_) / float(table_.size()); }

  inline bool find(const key_type key) const {
    for (size_type cur_hf = 0; cur_hf < num_hfs_; ++cur_hf) {
      auto cur_bucket = hash_funs_[cur_hf](key) % num_buckets_;
      bool key_exist  = find_in_bucket(key, cur_bucket);
      if (key_exist) { return true; }
    }
    return false;
  }

  inline std::vector<bool> find(const std::vector<key_type>& keys) const {
    std::vector<bool> results(keys.size(), false);
    for (size_t i = 0; i < keys.size(); i++) { results[i] = find(keys[i]); }
    return results;
  }
  std::vector<key_type> data() { return table_; }
  size_type get_buckets_count() { return num_buckets_; }
  inline void print() const {
    for (size_type bucket_id = 0; bucket_id < num_buckets_; ++bucket_id) {
      for (size_type i = 0; i < bucket_size_; ++i) {
        auto key = table_[bucket_id * bucket_size_ + i];
        std::cout << bucket_id << ", " << i << " -> " << key << '\n';
      }
    }
  }

 private:
  inline void set_key_at(const size_type& bucket_id,
                         const size_type& location,
                         const key_type& key) {
    table_[bucket_id * bucket_size_ + location] = key;
  }

  inline key_type swap_key_at(const size_type& bucket_id,
                              const size_type& location,
                              const key_type& key) {
    auto old_key                                = table_[bucket_id * bucket_size_ + location];
    table_[bucket_id * bucket_size_ + location] = key;
    return old_key;
  }

  inline size_type count_bucket(size_type bucket_id) const {
    size_type count = 0;
    for (size_type i = 0; i < bucket_size_; ++i) {
      auto key = table_[bucket_id * bucket_size_ + i];
      if (key == empty_key) { break; }
      count++;
    }
    return count;
  }
  inline bool find_in_bucket(const key_type& look_up_key, const size_type& bucket_id) const {
    for (size_type i = 0; i < bucket_size_; ++i) {
      auto key = table_[bucket_id * bucket_size_ + i];
      if (key == empty_key) { break; }
      if (key == look_up_key) { return true; }
    }
    return false;
  }

  size_type pairs_per_bucket_;

  float load_factor_;
  size_type lf_pairs_capacity_;
  size_type num_buckets_;
  size_type bucket_size_;
  std::vector<hash_type> hash_funs_;
  size_type max_chains_;
  size_type num_hfs_;
  size_type num_keys_;
  std::vector<key_type> table_;
  std::mt19937 rng_;
};
}  // namespace serial_bcht
