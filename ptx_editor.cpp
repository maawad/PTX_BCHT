// QT
#include <qapplication.h>
#include <qpushbutton.h>
#include <qregularexpression.h>
#include <qstylefactory.h>
#include <qtextedit.h>
#include <QFileDialog>
#include <QHBoxLayout>

// CUDA
#include <cuda.h>
#include <cuda_runtime.h>

// C++
#include <iostream>
#include <vector>

// includes
#include <helpers.hpp>
#include <ptx_highlighter.hpp>
#include <ptx_kernel.hpp>
#include <rkg.hpp>
#include <serial_cuckoo.hpp>
#include <hash_function.hpp>
using key_type = uint32_t;

struct keys_io {
  keys_io(uint32_t num_keys) { make_keys(num_keys); }
  void make_keys(uint32_t num_keys, const float exist_ratio = 0.5) {
    std::vector<key_type> h_keys;
    rkg::generate_uniform_unique_keys(h_keys, num_keys * 2, 1u);
    std::vector<key_type> h_find_keys(num_keys);
    rkg::prep_experiment_find_with_exist_ratio(exist_ratio, num_keys, h_keys, h_find_keys);
    h_keys.resize(num_keys);

    d_keys      = h_keys;
    d_find_keys = h_find_keys;
  }
  cuda_array<key_type> d_keys;                // input keys
  cuda_array<key_type> d_find_keys;           // input find keys
  cuda_array<bool> d_key_exist;  // query result
};

struct gpu_hashset {
  gpu_hashset(std::size_t capacity) : capacity_(capacity) {
    num_buckets_ = (capacity + bucket_size_ - 1) / bucket_size_;
    d_table_.resize(num_buckets_ * bucket_size_);

    unsigned seed = 0;
    std::mt19937 rng(seed);
    hf0_ = universal_hash<key_type>(generated_random_hf<key_type>(rng));
    hf1_ = universal_hash<key_type>(generated_random_hf<key_type>(rng));
    hf2_ = universal_hash<key_type>(generated_random_hf<key_type>(rng));
  }

 private:
  cuda_array<key_type> d_table_;
  static const int bucket_size_;
  std::size_t capacity_;
  std::size_t num_buckets_;
  universal_hash<key_type> hf0_;
  universal_hash<key_type> hf1_;
  universal_hash<key_type> hf2_;
};

int main(int argc, char **argv) {
  QApplication test(argc, argv);

  // Dark style
  test.setStyle(QStyleFactory::create("Fusion"));
  QPalette newPalette;
  newPalette.setColor(QPalette::Window, QColor(37, 37, 37));
  newPalette.setColor(QPalette::WindowText, QColor(212, 212, 212));
  newPalette.setColor(QPalette::Base, QColor(60, 60, 60));
  newPalette.setColor(QPalette::AlternateBase, QColor(45, 45, 45));
  newPalette.setColor(QPalette::PlaceholderText, QColor(127, 127, 127));
  newPalette.setColor(QPalette::Text, QColor(212, 212, 212));
  newPalette.setColor(QPalette::Button, QColor(45, 45, 45));
  newPalette.setColor(QPalette::ButtonText, QColor(212, 212, 212));
  newPalette.setColor(QPalette::BrightText, QColor(240, 240, 240));
  newPalette.setColor(QPalette::Highlight, QColor(38, 79, 120));
  newPalette.setColor(QPalette::HighlightedText, QColor(240, 240, 240));
  newPalette.setColor(QPalette::Light, QColor(60, 60, 60));
  newPalette.setColor(QPalette::Midlight, QColor(52, 52, 52));
  newPalette.setColor(QPalette::Dark, QColor(30, 30, 30));
  newPalette.setColor(QPalette::Mid, QColor(37, 37, 37));
  newPalette.setColor(QPalette::Shadow, QColor(0, 0, 0));
  newPalette.setColor(QPalette::Disabled, QPalette::Text, QColor(127, 127, 127));
  test.setPalette(newPalette);

  // input keys
  // todo: add spinbox for num keys
  const int device_id = 0;
  CUdevice gpu        = get_cuda_device(device_id);

  const int num_keys = 1'000'000;
  keys_io input(num_keys);

  // Window
  QWidget *window = new QWidget;
  window->resize(512, 512);

  QTextEdit *editor            = new QTextEdit;
  QTextEdit *compiler_output   = new QTextEdit;
  QPushButton *load_button     = new QPushButton("Load PTX");
  QPushButton *compile_button  = new QPushButton("Compile PTX");
  ptx_highlighter *highlighter = new ptx_highlighter(editor->document());

  // window layout
  QGridLayout *main_layout    = new QGridLayout(window);
  QHBoxLayout *buttons_layout = new QHBoxLayout;
  compiler_output->setFixedHeight(128);
  buttons_layout->addWidget(load_button);
  buttons_layout->addWidget(compile_button);
  // todo: fix the layout
  main_layout->addWidget(editor, 0, 0);
  main_layout->addWidget(compiler_output, 1, 0);
  main_layout->addLayout(buttons_layout, 2, 0);

  // Signals and slots
  QObject::connect(load_button, &QPushButton::clicked, [&]() {
    QString ptx_fname =
        QFileDialog::getOpenFileName(0, "Select source file", ".", "PTX files (*.ptx)");
    QFile ptx_file(ptx_fname);
    if (!ptx_file.open(QIODevice::ReadOnly | QIODevice::Text)) return;
    QTextStream in(&ptx_file);
    editor->clear();
    while (!in.atEnd()) {
      QString line = in.readLine();
      editor->append(line);
    }
  });

  ptx_kernel insertion_kernel;
  ptx_kernel find_kernel;
  insertion_kernel.set_kernel_entry("bcht_insert");

   QObject::connect(
      compile_button, &QPushButton::clicked, [&]() { 
        insertion_kernel.set_kernel_source(editor->toPlainText().toStdString());
        auto compiler_log = insertion_kernel.compile();
        compiler_output->clear();
    compiler_output->setText(QString::fromStdString(compiler_log));
  });


  cudaDeviceProp device_prop;
  cudaGetDeviceProperties(&device_prop, device_id);
  auto gpu_name = device_prop.name;
  editor->setText(gpu_name);

  window->show();
  return test.exec();
}
