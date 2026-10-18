#ifndef MLX_IO_PRIVATE_H
#define MLX_IO_PRIVATE_H

#include <iostream>
#include <streambuf>

#include "mlx/mlx.h"

namespace {

class CReader : public mlx::core::io::Reader {
 public:
  void* desc;
  mlx_io_vtable vtable;

  CReader(void* desc, mlx_io_vtable vtable) : desc(desc), vtable(vtable) {};
  virtual bool is_open() const override {
    return vtable.is_open(desc);
  };
  virtual bool good() const override {
    return vtable.good(desc);
  };
  virtual size_t tell() override {
    return vtable.tell(desc);
  }
  virtual void seek(
      int64_t off,
      std::ios_base::seekdir way = std::ios_base::beg) override {
    int status = 0;
    switch (way) {
      case std::ios_base::beg:
        status = vtable.seek(desc, off, SEEK_SET);
        break;
      case std::ios_base::cur:
        status = vtable.seek(desc, off, SEEK_CUR);
        break;
      case std::ios_base::end:
        status = vtable.seek(desc, off, SEEK_END);
        break;
      default:
        reader_error("invalid seek way");
    }
    if (status < 0) {
      reader_error("unable to seek");
    }
  }
  virtual void read(char* data, size_t n) override {
    auto read_n = vtable.read(desc, data, n);
    if (read_n != n) {
      reader_error(
          std::format("unable to read {} bytes (read {} instead)", n, read_n));
    }
  };
  virtual void read(char* data, size_t n, size_t offset) override {
    auto read_n = vtable.read_at_offset(desc, data, n, offset);
    if (read_n != n) {
      reader_error(
          std::format("unable to read {} bytes (read {} instead)", n, read_n));
    }
  };
  virtual std::string label() const override {
    return vtable.label(desc);
  };
  void reader_error(const std::string& msg) {
    throw std::runtime_error(
        std::format("[mlx_io_reader] {} in {}", msg, label()));
  }
  virtual ~CReader() {
    vtable.free(desc);
  }
};

class CWriter : public mlx::core::io::Writer {
 public:
  void* desc;
  mlx_io_vtable vtable;

  CWriter(void* desc, mlx_io_vtable vtable) : desc(desc), vtable(vtable) {};
  virtual bool is_open() const override {
    return vtable.is_open(desc);
  };
  virtual bool good() const override {
    return vtable.good(desc);
  };
  virtual size_t tell() override {
    return vtable.tell(desc);
  }
  virtual void seek(
      int64_t off,
      std::ios_base::seekdir way = std::ios_base::beg) override {
    int status = 0;
    switch (way) {
      case std::ios_base::beg:
        status = vtable.seek(desc, off, SEEK_SET);
        break;
      case std::ios_base::cur:
        status = vtable.seek(desc, off, SEEK_CUR);
        break;
      case std::ios_base::end:
        status = vtable.seek(desc, off, SEEK_END);
        break;
      default:
        writer_error("invalid seek way");
    }
    if (status < 0) {
      writer_error("unable to seek");
    }
  }
  virtual void write(const char* data, size_t n) override {
    auto wrote_n = vtable.write(desc, data, n);
    if (wrote_n != n) {
      writer_error(
          std::format(
              "unable to write {} bytes (wrote {} instead)", n, wrote_n));
    }
  };
  virtual std::string label() const override {
    return vtable.label(desc);
  };
  void writer_error(const std::string& msg) {
    throw std::runtime_error(
        std::format("[mlx_io_writer] {} in {}", msg, label()));
  }
  virtual ~CWriter() {
    vtable.free(desc);
  }
};

struct creader_holder {
  std::shared_ptr<CReader> ptr;
};

inline mlx_io_reader mlx_io_reader_new_(void* uctx, mlx_io_vtable vtable) {
  return mlx_io_reader(
      {new creader_holder({std::make_shared<CReader>(uctx, vtable)})});
}

inline std::shared_ptr<CReader> mlx_io_reader_get_(mlx_io_reader d) {
  if (!d.ctx) {
    throw std::runtime_error("expected a non-empty mlx_io_reader");
  }
  return static_cast<creader_holder*>(d.ctx)->ptr;
}

inline void mlx_io_reader_free_(mlx_io_reader io) {
  if (io.ctx) {
    delete static_cast<creader_holder*>(io.ctx);
  }
}

struct cwriter_holder {
  std::shared_ptr<CWriter> ptr;
};

inline mlx_io_writer mlx_io_writer_new_(void* uctx, mlx_io_vtable vtable) {
  return mlx_io_writer(
      {new cwriter_holder({std::make_shared<CWriter>(uctx, vtable)})});
}

inline std::shared_ptr<CWriter> mlx_io_writer_get_(mlx_io_writer d) {
  if (!d.ctx) {
    throw std::runtime_error("expected a non-empty mlx_io_writer");
  }
  return static_cast<cwriter_holder*>(d.ctx)->ptr;
}

inline void mlx_io_writer_free_(mlx_io_writer io) {
  if (io.ctx) {
    delete static_cast<cwriter_holder*>(io.ctx);
  }
}

class CFileStreamBuf : public std::streambuf {
 public:
  explicit CFileStreamBuf(FILE* file) : file_(file) {}

 protected:
  int_type overflow(int_type c) override {
    if (c != traits_type::eof()) {
      if (std::fputc(c, file_) == EOF) {
        return traits_type::eof();
      }
    }
    return c;
  }
  std::streamsize xsputn(const char* s, std::streamsize n) override {
    return std::fwrite(s, 1, n, file_);
  }
  int sync() override {
    return std::fflush(file_) == 0 ? 0 : -1;
  }

 private:
  FILE* file_;
};

class CFileOutputStream : public std::ostream {
 public:
  explicit CFileOutputStream(FILE* file) : std::ostream(&buf_), buf_(file) {}

  template <typename T>
  static T& as_lvalue(T&& t) {
    return t;
  }

 private:
  CFileStreamBuf buf_;
};

} // namespace

#endif
