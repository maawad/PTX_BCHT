#pragma once
// CUDA driver & runtime
#include <cuda.h>
#include <cuda_runtime.h>

// C++
#include <cstdint>

// TODO: pass optimization/other flags when compling the PTX
std::string inline read_ptx(const std::string ptx_file_path) {
  FILE* fp = fopen(ptx_file_path.c_str(), "rb");
  if (fp != NULL) {
    fseek(fp, 0, SEEK_END);
    int file_size = ftell(fp);
    char* buf     = new char[file_size + 1];
    fseek(fp, 0, SEEK_SET);
    fread(buf, sizeof(char), file_size, fp);
    fclose(fp);
    buf[file_size]         = '\0';
    std::string ptx_source = buf;
    delete[] buf;
    return ptx_source;
  } else {
    std::cout << "Failed to open the ptx source at " << ptx_file_path << std::endl;
    std::terminate();
  }
  return "";
}

std::string ptxJIT(CUmodule* ph_module,
            CUfunction* ph_kernel,
            CUlinkState* linker_state,
            const std::string ptx_source,
            const std::string kernel_entry_string,
            const bool quiet = false) {
  // see JIT Sample
  // C:\ProgramData\NVIDIA Corporation\CUDA Samples\v11.4\0_Simple\inlinePTX_nvrtc
  const unsigned int num_options = 7;
  CUjit_option options[num_options];
  void* option_values[num_options];
  float walltime;
  const unsigned int log_size = 8192;
  char error_log[log_size], info_log[log_size];
  void* cubin_out;
  size_t cubin_size;

  // Setup linker options
  options[0]       = CU_JIT_WALL_TIME;  // Return walltime from JIT compilation
  option_values[0] = (void*)&walltime;
  options[1]       = CU_JIT_INFO_LOG_BUFFER;  // Pass a buffer for info messages
  option_values[1] = (void*)info_log;
  options[2]       = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;  // Pass the size of the info buffer
  option_values[2] = (void*)(long)log_size;
  options[3]       = CU_JIT_ERROR_LOG_BUFFER;  // Pass a buffer for error message
  option_values[3] = (void*)error_log;
  options[4]       = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;  // Pass the size of the error buffer
  option_values[4] = (void*)(long)log_size;
  options[5]       = CU_JIT_LOG_VERBOSE;  // Make the linker verbose
  option_values[5] = (void*)1;
  options[6]       = CU_JIT_OPTIMIZATION_LEVEL;  // Make the linker verbose
  option_values[6] = (void*)4;

  // Creates a linker invoation
  cuda_try(cuLinkCreate(num_options, options, option_values, linker_state));


  // Load the PTX from the ptx file
  cuda_try(cuLinkAddData(*linker_state,
                         CU_JIT_INPUT_PTX,
                         (void*)ptx_source.c_str(),
                         strlen(ptx_source.c_str()) + 1,
                         0,
                         0,
                         0,
                         0));

  // Complete the linker step
  cuda_try(cuLinkComplete(*linker_state, &cubin_out, &cubin_size));

  // Linker walltime and info_log were requested in options above.
  if (!quiet) {
    printf("--------------------------------------------------------\n"
           "CUDA Link Completed in %f ms.\n"
           "Linker Output:\n%s"
           "\n--------------------------------------------------------\n",
           walltime,
           info_log);
  }
  // Load resulting cuBin into module
  cuda_try(cuModuleLoadData(ph_module, cubin_out));

  // Locate the kernel entry point
  cuda_try(cuModuleGetFunction(ph_kernel, *ph_module, kernel_entry_string.c_str()));

  // Destroy the linker invocation
  cuda_try(cuLinkDestroy(*linker_state));

  return std::string(info_log);
}

struct ptx_kernel {
  ptx_kernel(const std::string kernel_path, const std::string kernel_entry_string)
      : h_module_(0)
      , h_kernel_(0)
      , link_state_(0)
      , kernel_path_(kernel_path)
      , kernel_source_("")
      , kernel_entry_string_(kernel_entry_string) {}


   ptx_kernel()
      : h_module_(0)
      , h_kernel_(0)
      , link_state_(0)
      , kernel_path_("")
      , kernel_source_("")
      , kernel_entry_string_("") {}

  void launch(uint32_t num_threads, void** args, const bool quiet = false) {
    uint32_t block_size = 64;
    uint32_t num_blocks = (num_threads + block_size - 1) / block_size;
    dim3 block(block_size, 1, 1);
    dim3 grid(num_blocks, 1, 1);

    // Launch the kernel
    cuda_try(cuLaunchKernel(
        h_kernel_, grid.x, grid.y, grid.z, block.x, block.y, block.z, 0, NULL, args, NULL));
    if (!quiet) {
      std::cout << "CUDA kernel: " << kernel_entry_string_ << " launched" << std::endl;
    }
  }
  // TODO: add optional flags
  std::string compile(const bool quiet = false) {
    // JIT Compile the Kernel from PTX and get the Handles (Driver API)
    // This has to be after allocating memory or cuLinkCreate fails(?)
    if (kernel_source_ == "") { 
        kernel_source_ = read_ptx(kernel_path_);
    }
    return ptxJIT(&h_module_, &h_kernel_, &link_state_, kernel_source_, kernel_entry_string_, quiet);
  }
  void set_kernel_source(const std::string source) {
      kernel_source_ = source;
  }
  void set_kernel_entry(const std::string entry) { 
      kernel_entry_string_ = entry; 
  }
 private:
  CUmodule h_module_;
  CUfunction h_kernel_;
  CUlinkState link_state_;
  std::string kernel_path_;
  std::string kernel_source_;
  std::string kernel_entry_string_;
};