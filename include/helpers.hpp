#pragma once
#include <algorithm>
#include <iostream>
#include <optional>
#include <string_view>
#include <typeinfo>
#include <vector>

#define cuda_try(call)                                                                \
  do {                                                                                \
    cudaError_t err = static_cast<cudaError_t>(call);                                 \
    if (err != cudaSuccess) {                                                         \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, cudaGetErrorName(err)); \
      std::terminate();                                                               \
    }                                                                                 \
  } while (0)

std::string str_tolower(const std::string_view s) {
  std::string output(s.length(), ' ');
  std::transform(
      s.begin(), s.end(), output.begin(), [](unsigned char c) { return std::tolower(c); });
  return output;
}

// Finds an argument value
// auto arguments = std::vector<std::string>(argv, argv + argc);
// Example:
// auto k = get_arg_value<T>(arguments, "-flag")
// auto arguments = std::vector<std::string>(argv, argv + argc);
template <typename T>
std::optional<T> get_arg_value(const std::vector<std::string>& arguments, const char* flag) {
  uint32_t first_argument = 1;
  for (uint32_t i = first_argument; i < arguments.size(); i++) {
    std::string_view argument = std::string_view(arguments[i]);
    auto key_start            = argument.find_first_not_of("-");
    auto value_start          = argument.find("=");

    bool failed = argument.length() == 0;              // there is an argument
    failed |= key_start == std::string::npos;          // it has a -
    failed |= value_start == std::string::npos;        // it has an =
    failed |= key_start > 2;                           // - or -- at beginning
    failed |= (value_start - key_start) == 0;          // there is a key
    failed |= (argument.length() - value_start) == 1;  // = is not last

    if (failed) {
      std::cout << "Invalid argument: " << argument << " ignored.\n";
      std::cout << "Use: -flag=value " << std::endl;
      std::terminate();
    }

    std::string_view argument_name = argument.substr(key_start, value_start - key_start);
    value_start++;  // ignore the =
    std::string_view argument_value = argument.substr(value_start, argument.length() - key_start);

    if (argument_name == std::string_view(flag)) {
      if constexpr (std::is_same<T, float>::value) {
        return static_cast<T>(std::strtof(argument_value.data(), nullptr));
      } else if constexpr (std::is_same<T, double>::value) {
        return static_cast<T>(std::strtod(argument_value.data(), nullptr));
      } else if constexpr (std::is_same<T, int>::value) {
        return static_cast<T>(std::strtol(argument_value.data(), nullptr, 10));
      } else if constexpr (std::is_same<T, long long>::value) {
        return static_cast<T>(std::strtoll(argument_value.data(), nullptr, 10));
      } else if constexpr (std::is_same<T, uint32_t>::value) {
        return static_cast<T>(std::strtoul(argument_value.data(), nullptr, 10));
      } else if constexpr (std::is_same<T, uint64_t>::value) {
        return static_cast<T>(std::strtoull(argument_value.data(), nullptr, 10));
      } else if constexpr (std::is_same<T, std::string>::value) {
        return std::string(argument_value);
      } else if constexpr (std::is_same<T, bool>::value) {
        return str_tolower(argument_value) == "true" || str_tolower(argument_value) == "yes" ||
               str_tolower(argument_value) == "1" || str_tolower(argument_value) == "y";
      } else {
        std::cout << "Unknown type" << std::endl;
        std::terminate();
      }
    }
  }
  return {};
}
CUdevice get_cuda_device(const int device_id, bool quiet = false) {
  CUdevice device;
  int device_count = 0;

  cuda_try(cuInit(0));  // Flag parameter must be zero
  cuda_try(cuDeviceGetCount(&device_count));

  if (device_count == 0) {
    std::cout << "No CUDA capable device found." << std::endl;
    std::terminate();
  }

  cuda_try(cuDeviceGet(&device, device_id));

  cudaDeviceProp device_prop;
  cudaGetDeviceProperties(&device_prop, device_id);

  if (!quiet) { std::cout << "Device[" << device_id << "]: " << device_prop.name << std::endl; }

  return device;
}

template <typename T>
struct cuda_array {
  cuda_array(const std::size_t size) : size_(size), ptr_(nullptr) { allocate(); }
  cuda_array(const std::size_t size, const T value) : size_(size), ptr_(nullptr) {
    allocate();
    set_value(value);
  }
  cuda_array() : size_(0), ptr_(nullptr) {}

  // copy constructor
  cuda_array<T>(const cuda_array<T>&) = delete;
  cuda_array<T>(const std::vector<T>& input) {
    size_ = input.size();
    allocate();
    from_std_vector(input);
  }
  // move constructor
  cuda_array<T>(cuda_array<T>&&) = delete;
  // move assignment operator
  cuda_array<T>& operator=(cuda_array<T>&&) = delete;
  // copy assignment operator
  cuda_array<T>& operator=(cuda_array<T>&) = delete;
  cuda_array<T>& operator                  =(const std::vector<T>& input) {
    free();
    size_ = input.size();
    allocate();
    from_std_vector(input);
    return *this;
  };

  void copy_to_host(T* input) {
    cuda_try(cudaMemcpy(input, ptr_, sizeof(T) * size_, cudaMemcpyDeviceToHost));
  }

  void free() {
    if (ptr_) cuda_try(cudaFree(ptr_));
  }

  ~cuda_array<T>() {}

  const T* data() const { return ptr_; }
  T* data() { return ptr_; }

  std::vector<T> to_std_vector() {
    assert(ptr_ != nullptr);
    std::vector<T> h_cpy(size_, static_cast<T>(0));
    auto raw_ptr = h_cpy.data();
    copy_to_host(raw_ptr);
    return h_cpy;
  }

 private:
  void set_value(const T value) { cuda_try(cudaMemset(ptr_, value, sizeof(T) * size_)); }
  void allocate() { cuda_try(cudaMalloc((void**)&ptr_, sizeof(T) * size_)); }
  void copy_to_device(const T* input) {
    cuda_try(cudaMemcpy(ptr_, input, sizeof(T) * size_, cudaMemcpyHostToDevice));
  }
  void from_std_vector(const std::vector<T>& input) {
    // make sure everything is correct
    assert(input.size() == size_);
    assert(ptr_ != nullptr);
    copy_to_device(input.data());
  }

  std::size_t size_;
  T* ptr_;
};
