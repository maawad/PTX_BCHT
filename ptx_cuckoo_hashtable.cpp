
// CUDA driver & runtime
#include <cuda.h>
#include <thrust/device_vector.h>
#include <helpers.hpp>
#include <ptx_kernel.hpp>

// C++
#include <cstdint>
#include <iomanip>
#include <numeric>
#include <rkg.hpp>
#include <serial_cuckoo.hpp>
#include <vector>

// Timer
#include <gpu_timer.hpp>

void gpu_build(int argc, char** argv) {
  std::vector<std::string> arguments(argv, argv + argc);

  uint32_t num_keys = get_arg_value<uint32_t>(arguments, "num-keys").value_or(32);
  float load_factor = get_arg_value<float>(arguments, "load-factor").value_or(0.5f);
  float exist_ratio = get_arg_value<float>(arguments, "exist-ratio").value_or(0.5f);
  int device_id     = get_arg_value<int>(arguments, "device-id").value_or(0);
  bool quiet        = get_arg_value<bool>(arguments, "quiet").value_or(false);
  bool validate     = get_arg_value<bool>(arguments, "validate").value_or(false);

  const uint32_t bucket_size = 32;
  if (!quiet) {
    std::cout << "num_keys: " << num_keys << '\n';
    std::cout << "load_factor: " << load_factor << '\n';
    std::cout << "device_id: " << device_id << '\n';
  }

  // Device setup
  CUdevice dev = get_cuda_device(device_id, quiet);
  int driver_version;
  cudaDriverGetVersion(&driver_version);
  if (driver_version < CUDART_VERSION) {
    printf("driver_version = %d < CUDART_VERSION = %d \n", driver_version, CUDART_VERSION);
    std::terminate();
  }

  // Setup the input
  using key_type = uint32_t;
  cuda_array<key_type> d_keys;                // input keys
  cuda_array<key_type> d_find_keys;           // input keys
  cuda_array<bool> d_key_exist(num_keys, 0);  // hash table

  std::vector<key_type> h_keys;
  rkg::generate_uniform_unique_keys(h_keys, num_keys * 2, 1u);

  std::vector<key_type> h_find_keys(num_keys);
  rkg::prep_experiment_find_with_exist_ratio(exist_ratio, num_keys, h_keys, h_find_keys);
  h_keys.resize(num_keys);

  // construct a serial BCHT
  // Hash functions
  unsigned seed = 0;
  std::mt19937 rng(seed);
  universal_hash hf0{generated_random_hf<key_type>(rng)};
  universal_hash hf1{generated_random_hf<key_type>(rng)};
  universal_hash hf2{generated_random_hf<key_type>(rng)};

  // build the ref table
  serial_bcht::cuckoo_hash_set ref_set(bucket_size, 3, num_keys, load_factor, hf0, hf1, hf2);
  bool success = ref_set.insert(h_keys);
  if (!success) {
    std::cout << "reference bcht failed\n";
    std::terminate();
  }
  key_type sentinel_key{0xffffffff};
  uint32_t num_buckets = ref_set.get_buckets_count();
  cuda_array<key_type> d_table(ref_set.get_buckets_count() * bucket_size, sentinel_key);

  // Move keys
  d_keys      = h_keys;
  d_find_keys = h_find_keys;

  key_type* d_keys_ptr(d_keys.data());
  key_type* d_find_keys_ptr(d_find_keys.data());
  key_type* d_table_ptr(d_table.data());
  bool* d_key_exist_ptr(d_key_exist.data());

  // Insert kernel
  void* insert_kernel_args[7] = {
      &d_table_ptr, &d_keys_ptr, &num_keys, &hf0, &hf1, &hf2, &num_buckets};
  ptx_kernel insert_kernel(std::string(PTX_INCLUDE_DIR) + "/bcht_insert_kernel.ptx", "bcht_insert");
  insert_kernel.compile(quiet);
  gpu_timer insert_timer;
  insert_timer.start_timer();
  insert_kernel.launch(num_keys, insert_kernel_args, quiet);
  insert_timer.stop_timer();
  cuda_try(cudaDeviceSynchronize());

  // Find kernel
  void* find_kernel_args[8] = {
      &d_table_ptr, &d_find_keys_ptr, &num_keys, &d_key_exist_ptr, &hf0, &hf1, &hf2, &num_buckets};
  ptx_kernel find_kernel(std::string(PTX_INCLUDE_DIR) + "/bcht_find_kernel.ptx", "bcht_find");
  find_kernel.compile(quiet);
  gpu_timer find_timer;
  find_timer.start_timer();
  find_kernel.launch(num_keys, find_kernel_args, quiet);
  find_timer.stop_timer();
  cuda_try(cudaDeviceSynchronize());

  auto ref_results  = ref_set.find(h_find_keys);
  bool* h_key_exist = new bool[num_keys];
  d_key_exist.copy_to_host(h_key_exist);

  uint32_t found_count = 0;
  for (std::size_t i = 0; i < num_keys; i++) {
    if (h_key_exist[i] != ref_results[i]) {
      std::cout << i << ") error at Key: " << h_find_keys[i] << "-> ";
      std::cout << (h_key_exist[i] ? "exists\n" : "doesn't exist.\n");
      std::terminate();
    }
    if (ref_results[i]) found_count++;
  }
  if (!quiet) { std::cout << "Success!\n"; }
  auto find_seconds = find_timer.get_elapsed_s();
  auto find_rate    = static_cast<double>(num_keys) / 1e6 / find_seconds;
  auto find_ratio   = double(found_count) / num_keys * 100.0;

  auto insert_seconds = insert_timer.get_elapsed_s();
  auto insert_rate    = static_cast<double>(num_keys) / 1e6 / insert_seconds;
  if (!quiet) {
    std::cout << "Find rate:      " << find_rate << " million keys/s\n";
    std::cout << "Insert rate:    " << insert_rate << " million keys/s\n";
    std::cout << "Find ratio was: " << find_ratio << "%\n";
  } else {
    std::cout << std::setw(8) << num_keys / 1e6;
    std::cout << std::setw(15) << load_factor * 100;
    std::cout << std::setw(19) << std::setprecision(2) << std::fixed << insert_rate;
    std::cout << std::setw(14) << std::setprecision(2) << std::fixed << find_rate << '\n';
  }
  // free memory
  delete[] h_key_exist;
  d_keys.free();
  d_table.free();
  d_key_exist.free();
}

int main(int argc, char** argv) { gpu_build(argc, argv); }
