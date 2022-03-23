// QT
#include <qapplication.h>
#include <qcheckbox.h>
#include <qpushbutton.h>
#include <qregularexpression.h>
#include <qstylefactory.h>
#include <qtablewidget.h>
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
#include <hash_function.hpp>
#include <helpers.hpp>
#include <ptx_highlighter.hpp>
#include <ptx_kernel.hpp>
#include <rkg.hpp>
#include <serial_cuckoo.hpp>
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
  cuda_array<key_type> d_keys;       // input keys
  cuda_array<key_type> d_find_keys;  // input find keys
  cuda_array<bool> d_key_exist;      // query result
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
  bool use_hot_reload = false;
  // Window
  QWidget *window = new QWidget;
  window->resize(512, 512);

  QTextEdit *insert_editor   = new QTextEdit;
  QTextEdit *find_editor     = new QTextEdit;
  QTextEdit *compiler_output = new QTextEdit;

  QPushButton *load_insert_button = new QPushButton("Load Insert Kernel PTX");
  QPushButton *load_find_button   = new QPushButton("Load Find Kernel PTX");
  QPushButton *compile_button     = new QPushButton("Compile PTX");

  QCheckBox *hot_reload_cbox = new QCheckBox("Hot Reload");
  hot_reload_cbox->setChecked(false);

  ptx_highlighter *insert_highlighter = new ptx_highlighter(insert_editor->document());
  ptx_highlighter *find_highlighter   = new ptx_highlighter(find_editor->document());

  // window layout
  QGridLayout *main_layout    = new QGridLayout(window);
  QHBoxLayout *buttons_layout = new QHBoxLayout;
  compiler_output->setFixedHeight(200);
  buttons_layout->addWidget(load_insert_button);
  buttons_layout->addWidget(load_find_button);
  buttons_layout->addWidget(compile_button);
  buttons_layout->addWidget(hot_reload_cbox);
  // todo: fix the layout
  main_layout->addWidget(insert_editor, 0, 0);
  main_layout->addWidget(find_editor, 0, 1);
  main_layout->addWidget(compiler_output, 1, 0, 1, 2);
  main_layout->addLayout(buttons_layout, 2, 0, 1, 2);

  // Signals and slots
  QObject::connect(load_insert_button, &QPushButton::clicked, [&]() {
    QString ptx_fname =
        QFileDialog::getOpenFileName(0, "Select source file", ".", "PTX files (*.ptx)");
    QFile ptx_file(ptx_fname);
    if (!ptx_file.open(QIODevice::ReadOnly | QIODevice::Text)) return;
    QTextStream in(&ptx_file);
    insert_editor->clear();
    while (!in.atEnd()) {
      QString line = in.readLine();
      insert_editor->append(line);
    }
  });
  QObject::connect(load_find_button, &QPushButton::clicked, [&]() {
    QString ptx_fname =
        QFileDialog::getOpenFileName(0, "Select source file", ".", "PTX files (*.ptx)");
    QFile ptx_file(ptx_fname);
    if (!ptx_file.open(QIODevice::ReadOnly | QIODevice::Text)) return;
    QTextStream in(&ptx_file);
    find_editor->clear();
    while (!in.atEnd()) {
      QString line = in.readLine();
      find_editor->append(line);
    }
  });

  QObject::connect(
      hot_reload_cbox, &QCheckBox::stateChanged, [&]() { use_hot_reload = !use_hot_reload; });

  ptx_kernel insertion_kernel;
  ptx_kernel find_kernel;
  insertion_kernel.set_kernel_entry("bcht_insert");
  find_kernel.set_kernel_entry("bcht_find");

  auto compile_lambda = [&]() {
    insertion_kernel.set_kernel_source(insert_editor->toPlainText().toStdString());
    std::string compiler_log;
    try {
      compiler_log = insertion_kernel.compile();
    } catch (const std::exception &e) { compiler_log = std::string("Error: ") + e.what(); }
    compiler_output->clear();
    compiler_output->setText(QString::fromStdString(compiler_log));

    find_kernel.set_kernel_source(find_editor->toPlainText().toStdString());
    try {
      compiler_log = find_kernel.compile();
    } catch (const std::exception &e) { compiler_log = std::string("Error: ") + e.what(); }
    compiler_output->append(QString::fromStdString(compiler_log));
  };

  QObject::connect(insert_editor, &QTextEdit::textChanged, [&]() {
    if (use_hot_reload) compile_lambda();
  });

  QObject::connect(find_editor, &QTextEdit::textChanged, [&]() {
    if (use_hot_reload) compile_lambda();
  });
  QObject::connect(compile_button, &QPushButton::clicked, compile_lambda);

  cudaDeviceProp device_prop;
  cudaGetDeviceProperties(&device_prop, device_id);
  auto gpu_name = device_prop.name;
  insert_editor->setText(gpu_name);

  window->show();
  return test.exec();
}
