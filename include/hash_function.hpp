#pragma once

template <typename Key>
struct universal_hash {
  using key_type    = Key;
  using result_type = Key;
  constexpr universal_hash(uint32_t hash_x, uint32_t hash_y, uint32_t prime_divisor)
      : hash_x_(hash_x), hash_y_(hash_y), prime_divisor_(prime_divisor) {}
  constexpr universal_hash() : hash_x_(0), hash_y_(0), prime_divisor_(0) {}

  constexpr result_type operator()(const key_type& key) const {
    return (((hash_x_ ^ key) + hash_y_) % prime_divisor_);
  }
  universal_hash<key_type>& operator=(const universal_hash<key_type> other) {
    prime_divisor_ = other.prime_divisor_;
    hash_x_        = other.hash_x_;
    hash_y_        = other.hash_y_;
    return *this;
  }
  friend std::ostream& operator<<(std::ostream& os, const universal_hash<uint32_t>& dt);

 private:
  uint32_t hash_x_;
  uint32_t hash_y_;
  uint32_t prime_divisor_;
};
std::ostream& operator<<(std::ostream& os, const universal_hash<uint32_t>& dt) {
  os << dt.hash_x_ << '\t' << dt.hash_y_ << '\t' << dt.prime_divisor_;
  return os;
}

template <typename key_type, typename rng_type>
universal_hash<key_type> generated_random_hf(rng_type& rng, uint32_t prime_divisor = 4294967291u) {
  uint32_t x = rng() % prime_divisor;
  if (x < 1u) { x = 1; }
  uint32_t y = rng() % prime_divisor;
  return {x, y, prime_divisor};
}
