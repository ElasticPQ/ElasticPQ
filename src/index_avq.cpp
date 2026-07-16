#include "epq/index_avq.h"

#include <Python.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace epq {
namespace {

std::string fetch_python_error() {
    if (!PyErr_Occurred()) {
        return "unknown Python error";
    }

    PyObject* type = nullptr;
    PyObject* value = nullptr;
    PyObject* traceback = nullptr;
    PyErr_Fetch(&type, &value, &traceback);
    PyErr_NormalizeException(&type, &value, &traceback);

    std::string message = "unknown Python error";
    if (value != nullptr) {
        PyObject* value_str = PyObject_Str(value);
        if (value_str != nullptr) {
            const char* text = PyUnicode_AsUTF8(value_str);
            if (text != nullptr) {
                message = text;
            }
            Py_DECREF(value_str);
        }
    }

    Py_XDECREF(type);
    Py_XDECREF(value);
    Py_XDECREF(traceback);
    return message;
}

[[noreturn]] void throw_python_error(const std::string& context) {
    throw std::runtime_error(context + ": " + fetch_python_error());
}

class PyObjectPtr {
   public:
    PyObjectPtr() = default;
    explicit PyObjectPtr(PyObject* obj) : obj_(obj) {}

    ~PyObjectPtr() {
        reset();
    }

    PyObjectPtr(const PyObjectPtr&) = delete;
    PyObjectPtr& operator=(const PyObjectPtr&) = delete;

    PyObjectPtr(PyObjectPtr&& other) noexcept : obj_(other.release()) {}

    PyObjectPtr& operator=(PyObjectPtr&& other) noexcept {
        if (this != &other) {
            reset(other.release());
        }
        return *this;
    }

    PyObject* get() const {
        return obj_;
    }

    PyObject* release() {
        PyObject* out = obj_;
        obj_ = nullptr;
        return out;
    }

    void reset(PyObject* obj = nullptr) {
        if (obj_ != nullptr) {
            Py_DECREF(obj_);
        }
        obj_ = obj;
    }

    explicit operator bool() const {
        return obj_ != nullptr;
    }

   private:
    PyObject* obj_ = nullptr;
};

class GilLock {
   public:
    GilLock() : state_(PyGILState_Ensure()) {}
    ~GilLock() {
        PyGILState_Release(state_);
    }

   private:
    PyGILState_STATE state_;
};

class PythonRuntime {
   public:
    static PythonRuntime& instance() {
        static PythonRuntime runtime;
        return runtime;
    }

    PyObject* numpy_module() const {
        return numpy_module_.get();
    }

    PyObject* scann_module() const {
        return scann_module_.get();
    }

   private:
    PythonRuntime() {
        Py_Initialize();
        PyEval_InitThreads();
        {
            GilLock gil;
            initialize_paths();
            numpy_module_.reset(PyImport_ImportModule("numpy"));
            if (!numpy_module_) {
                throw_python_error("failed to import numpy");
            }
            scann_module_.reset(
                    PyImport_ImportModule("scann.scann_ops.py.scann_ops_pybind"));
            if (!scann_module_) {
                throw_python_error(
                        "failed to import scann.scann_ops.py.scann_ops_pybind");
            }
        }
    }

    void append_sys_path(const std::filesystem::path& path) {
        if (path.empty() || !std::filesystem::exists(path)) {
            return;
        }
        PyObject* sys_path = PySys_GetObject("path");
        if (sys_path == nullptr) {
            throw_python_error("failed to access sys.path");
        }
        PyObjectPtr py_path(PyUnicode_FromString(path.string().c_str()));
        if (!py_path) {
            throw_python_error("failed to encode Python path");
        }
        if (PyList_Append(sys_path, py_path.get()) != 0) {
            throw_python_error("failed to append to sys.path");
        }
    }

    void initialize_paths() {
        const char* env_path = std::getenv("EPQ_AVQ_PYTHONPATH");
        if (env_path != nullptr && *env_path != '\0') {
            append_sys_path(env_path);
            return;
        }

        // Otherwise rely on the active Python environment's normal sys.path.
    }

    PyObjectPtr numpy_module_;
    PyObjectPtr scann_module_;
};

PyObjectPtr make_numpy_array(
        const RowMatrixXf& matrix,
        PyObject* numpy_module) {
    PyObjectPtr buffer(PyMemoryView_FromMemory(
            const_cast<char*>(
                    reinterpret_cast<const char*>(matrix.data())),
            static_cast<Py_ssize_t>(matrix.size() * sizeof(float)),
            PyBUF_READ));
    if (!buffer) {
        throw_python_error("failed to create Python memoryview");
    }

    PyObjectPtr frombuffer(PyObject_GetAttrString(numpy_module, "frombuffer"));
    PyObjectPtr float32_dtype(PyObject_GetAttrString(numpy_module, "float32"));
    if (!frombuffer || !float32_dtype) {
        throw_python_error("failed to access numpy.frombuffer/float32");
    }

    PyObjectPtr array_1d(
            PyObject_CallFunctionObjArgs(
                    frombuffer.get(), buffer.get(), float32_dtype.get(), nullptr));
    if (!array_1d) {
        throw_python_error("numpy.frombuffer failed");
    }

    PyObjectPtr reshaped(
            PyObject_CallMethod(
                    array_1d.get(),
                    "reshape",
                    "(nn)",
                    static_cast<Py_ssize_t>(matrix.rows()),
                    static_cast<Py_ssize_t>(matrix.cols())));
    if (!reshaped) {
        throw_python_error("numpy.reshape failed");
    }

    PyObjectPtr copied(PyObject_CallMethod(reshaped.get(), "copy", nullptr));
    if (!copied) {
        throw_python_error("numpy.copy failed");
    }
    return copied;
}

template <typename T>
void copy_python_array_2d(
        PyObject* array_obj,
        T* out,
        faiss::idx_t rows,
        faiss::idx_t cols,
        const char* context) {
    Py_buffer view;
    if (PyObject_GetBuffer(array_obj, &view, PyBUF_STRIDES | PyBUF_FORMAT) != 0) {
        throw_python_error(std::string(context) + ": failed to read Python buffer");
    }

    auto release = [&view]() {
        PyBuffer_Release(&view);
    };

    if (view.ndim != 2 ||
        view.shape[0] != rows ||
        view.shape[1] != cols) {
        release();
        std::ostringstream oss;
        oss << context << ": unexpected Python array shape";
        throw std::runtime_error(oss.str());
    }

    for (Py_ssize_t i = 0; i < view.shape[0]; ++i) {
        for (Py_ssize_t j = 0; j < view.shape[1]; ++j) {
            const char* ptr =
                    static_cast<const char*>(view.buf) +
                    i * view.strides[0] + j * view.strides[1];
            if constexpr (std::is_same_v<T, float>) {
                if (view.itemsize == static_cast<Py_ssize_t>(sizeof(float))) {
                    out[static_cast<size_t>(i * cols + j)] =
                            *reinterpret_cast<const float*>(ptr);
                } else if (view.itemsize ==
                           static_cast<Py_ssize_t>(sizeof(double))) {
                    out[static_cast<size_t>(i * cols + j)] =
                            static_cast<float>(
                                    *reinterpret_cast<const double*>(ptr));
                } else {
                    release();
                    throw std::runtime_error(
                            std::string(context) +
                            ": unsupported floating-point buffer element size");
                }
            } else {
                if (view.itemsize == static_cast<Py_ssize_t>(sizeof(int64_t))) {
                    out[static_cast<size_t>(i * cols + j)] =
                            static_cast<faiss::idx_t>(
                                    *reinterpret_cast<const int64_t*>(ptr));
                } else if (view.itemsize == static_cast<Py_ssize_t>(sizeof(int32_t))) {
                    out[static_cast<size_t>(i * cols + j)] =
                            static_cast<faiss::idx_t>(
                                    *reinterpret_cast<const int32_t*>(ptr));
                } else {
                    release();
                    throw std::runtime_error(
                            std::string(context) +
                            ": unsupported integer buffer element size");
                }
            }
        }
    }
    release();
}

}  // namespace

struct IndexAVQ::Impl {
    mutable std::mutex mutex;
    PyObjectPtr searcher;
};

IndexAVQ::IndexAVQ(int d_in, int total_bits_in)
        : faiss::Index(d_in, faiss::METRIC_L2),
          total_bits(total_bits_in),
          impl_(new Impl()) {}

IndexAVQ::~IndexAVQ() {
    delete impl_;
}

int IndexAVQ::resolve_dimensions_per_block() const {
    if (dimensions_per_block > 0) {
        return dimensions_per_block;
    }
    if (total_bits <= 0 || d <= 0) {
        return 2;
    }

    const double raw = static_cast<double>(d) * 4.0 / total_bits;
    return std::max(1, static_cast<int>(std::round(raw)));
}

int IndexAVQ::resolve_effective_budget_bits(int dims_per_block) const {
    const int blocks = (d + dims_per_block - 1) / dims_per_block;
    return blocks * 4;
}

void IndexAVQ::train(faiss::idx_t n, const float* x) {
    const auto rows = static_cast<Eigen::Index>(n);
    train_sample_.resize(rows, d);
    if (n > 0) {
        std::memcpy(
                train_sample_.data(),
                x,
                static_cast<size_t>(n) * static_cast<size_t>(d) * sizeof(float));
    }
    is_trained = true;
    training_stats_ = {};
}

void IndexAVQ::add(faiss::idx_t n, const float* x) {
    if (metric_type != faiss::METRIC_L2) {
        throw std::runtime_error("IndexAVQ currently supports METRIC_L2 only");
    }
    if (!is_trained) {
        train(0, nullptr);
    }

    database_.resize(static_cast<Eigen::Index>(n), d);
    if (n > 0) {
        std::memcpy(
                database_.data(),
                x,
                static_cast<size_t>(n) * static_cast<size_t>(d) * sizeof(float));
    }

    const int dims_per_block = resolve_dimensions_per_block();
    effective_budget_bits_ = resolve_effective_budget_bits(dims_per_block);

    PythonRuntime& runtime = PythonRuntime::instance();
    GilLock gil;
    std::lock_guard<std::mutex> lock(impl_->mutex);

    PyObjectPtr db_array = make_numpy_array(database_, runtime.numpy_module());
    PyObjectPtr builder_func(
            PyObject_GetAttrString(runtime.scann_module(), "builder"));
    if (!builder_func) {
        throw_python_error("failed to access scann builder");
    }

    PyObjectPtr builder(PyObject_CallFunction(
            builder_func.get(),
            "Ois",
            db_array.get(),
            default_num_neighbors,
            "squared_l2"));
    if (!builder) {
        throw_python_error("failed to create AVQ builder");
    }

    if (training_threads > 0) {
        PyObjectPtr ignored(PyObject_CallMethod(
                builder.get(),
                "set_n_training_threads",
                "i",
                training_threads));
        if (!ignored) {
            throw_python_error("failed to set AVQ training threads");
        }
    }

    PyObjectPtr score_ah_result(PyObject_CallMethod(
            builder.get(),
            "score_ah",
            "if",
            dims_per_block,
            anisotropic_quantization_threshold));
    if (!score_ah_result) {
        throw_python_error("failed to configure AVQ score_ah");
    }

    PyObjectPtr searcher(PyObject_CallMethod(builder.get(), "build", nullptr));
    if (!searcher) {
        throw_python_error("failed to build AVQ searcher");
    }
    if (search_threads > 0) {
        PyObjectPtr ignored(PyObject_CallMethod(
                searcher.get(),
                "set_num_threads",
                "i",
                search_threads));
        if (!ignored) {
            throw_python_error("failed to set AVQ search threads");
        }
    }

    impl_->searcher = std::move(searcher);
    ntotal = n;
}

void IndexAVQ::search(
        faiss::idx_t n,
        const float* x,
        faiss::idx_t k,
        float* distances,
        faiss::idx_t* labels,
        const faiss::SearchParameters* params) const {
    (void)params;
    if (!impl_->searcher) {
        throw std::runtime_error("IndexAVQ search called before add/build");
    }

    RowMatrixXf queries(static_cast<Eigen::Index>(n), d);
    if (n > 0) {
        std::memcpy(
                queries.data(),
                x,
                static_cast<size_t>(n) * static_cast<size_t>(d) * sizeof(float));
    }

    PythonRuntime& runtime = PythonRuntime::instance();
    GilLock gil;
    std::lock_guard<std::mutex> lock(impl_->mutex);

    PyObjectPtr q_array = make_numpy_array(queries, runtime.numpy_module());
    PyObjectPtr result;
    if (search_threads > 1) {
        result.reset(PyObject_CallMethod(
                impl_->searcher.get(),
                "search_batched_parallel",
                "Oiiii",
                q_array.get(),
                static_cast<int>(k),
                -1,
                -1,
                search_batch_size));
    } else {
        result.reset(PyObject_CallMethod(
                impl_->searcher.get(),
                "search_batched",
                "Oiii",
                q_array.get(),
                static_cast<int>(k),
                -1,
                -1));
    }
    if (!result) {
        throw_python_error("AVQ batch search failed");
    }
    if (!PyTuple_Check(result.get()) || PyTuple_Size(result.get()) != 2) {
        throw std::runtime_error("AVQ search returned an unexpected value");
    }

    PyObject* idx_obj = PyTuple_GetItem(result.get(), 0);
    PyObject* dist_obj = PyTuple_GetItem(result.get(), 1);
    copy_python_array_2d(idx_obj, labels, n, k, "AVQ search labels");
    copy_python_array_2d(dist_obj, distances, n, k, "AVQ search distances");
}

void IndexAVQ::reset() {
    database_.resize(0, d);
    train_sample_.resize(0, d);
    ntotal = 0;
    effective_budget_bits_ = 0;
    if (impl_ != nullptr) {
        GilLock gil;
        std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->searcher.reset();
    }
}

void IndexAVQ::reconstruct(faiss::idx_t key, float* recons) const {
    if (key < 0 || key >= ntotal) {
        throw std::runtime_error("IndexAVQ reconstruct key out of range");
    }
    std::memcpy(
            recons,
            database_.row(static_cast<Eigen::Index>(key)).data(),
            static_cast<size_t>(d) * sizeof(float));
}

const AVQTrainingStats& IndexAVQ::training_stats() const noexcept {
    return training_stats_;
}

int IndexAVQ::effective_budget_bits() const noexcept {
    return effective_budget_bits_;
}

const RowMatrixXf& IndexAVQ::database() const noexcept {
    return database_;
}

}  // namespace epq
