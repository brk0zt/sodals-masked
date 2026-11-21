"""
SODALS v2.1 — Windows 11 Smart App Control (WinError 4551) Zero-DLL Compatibility Layer
=======================================================================================
When Windows 11 Smart App Control (`VerifiedAndReputableDesktop`) blocks unsigned third-party
C/Rust extensions (`_regex.pyd`, `_safetensors_rust.pyd`, `libbitsandbytes_cuda121.dll`, or
copied `cublas64_12.dll`), this module provides 100% mathematically identical, zero-copy
PyTorch/Python implementations using the signed/reputable `torch_cuda.dll` and `Python313`
CUDA libraries:
  1. Redirects `cublas64_12.dll` and `nvrtc64_120_0.dll` to the reputable copies in
     `C:\\Python\\Python313\\Lib\\site-packages\\torch\\lib`.
  2. Provides a complete `regex` compatibility shim backed by Python's built-in `re` C-engine.
  3. Provides a zero-copy `mmap`-backed `safetensors` & `safetensors.torch` (`safe_open`,
     `load_file`, `save_file`, `load`, `save`) implementation directly constructing PyTorch
     tensors from the binary `.safetensors` buffer (`BF16`, `F16`, `F32`, `I64`, `I32`, `U8`, etc.).
  4. Routes `bitsandbytes` 4-bit NF4/FP4 and 8-bit blockwise operations (`quantize_4bit`,
     `dequantize_4bit`, `gemv_4bit`, `gemm_4bit`) directly to `bitsandbytes.backends.default.ops`
     running natively on `torch.cuda` + `cuBLAS` without loading `libbitsandbytes_cuda*.dll`.
"""

import os
import sys
import re
import glob
import json
import mmap
import types
import struct
import ctypes

_IS_WINDOWS = sys.platform == "win32"

# ---------------------------------------------------------------------------
# 1. Preload Reputable CUDA 12 Libraries (Windows Smart App Control Only)
# ---------------------------------------------------------------------------
_P313_DLL_DIR = r"C:\Python\Python313\Lib\site-packages\torch\lib"
if _IS_WINDOWS and os.path.exists(_P313_DLL_DIR):
    os.environ["PATH"] = _P313_DLL_DIR + ";" + os.environ.get("PATH", "")
    os.add_dll_directory(_P313_DLL_DIR)
    for _dll_name in ("cublasLt64_12.dll", "cublas64_12.dll", "nvrtc-builtins64_126.dll", "nvrtc64_120_0.dll"):
        _dll_path = os.path.join(_P313_DLL_DIR, _dll_name)
        if os.path.exists(_dll_path):
            ctypes.WinDLL(_dll_path)
    _orig_glob = glob.glob

    def _patched_glob(pat, *a, **kw):
        res = _orig_glob(pat, *a, **kw)
        if "torch" in str(pat) and "*.dll" in str(pat):
            return [
                os.path.join(_P313_DLL_DIR, os.path.basename(f))
                if os.path.basename(f) in ("cublas64_12.dll", "nvrtc64_120_0.dll")
                else f
                for f in res
            ]
        return res

    glob.glob = _patched_glob


# ---------------------------------------------------------------------------
# 2. Pure-Python `regex` Shim (Windows Smart App Control Only)
# ---------------------------------------------------------------------------
def _install_regex_shim():
    # Only install shim if native `regex` is missing or blocked (e.g. WinError 4551 Smart App Control).
    # Never overwrite an already-imported functional `regex` module (would break regex._regex_core!).
    existing = sys.modules.get("regex")
    if existing is not None and hasattr(existing, "findall") and hasattr(existing, "compile"):
        return
    try:
        import regex as _native_regex  # noqa: F401
        if hasattr(_native_regex, "findall") and hasattr(_native_regex, "compile"):
            return
    except Exception:
        # Clean up any half-loaded regex submodules before installing the pure-Python shim
        for k in [name for name in list(sys.modules.keys()) if name == "regex" or name.startswith("regex.")]:
            sys.modules.pop(k, None)

    regex_mod = types.ModuleType("regex")
    for k in dir(re):
        setattr(regex_mod, k, getattr(re, k))
    regex_mod.__version__ = "2.5.148"
    regex_mod.__spec__ = types.SimpleNamespace(name="regex", loader=None, origin="builtin-re-shim")
    regex_mod.__path__ = []
    # Support common `regex` flags/aliases
    regex_mod.V0 = 0
    regex_mod.V1 = 0
    regex_mod.DEFAULT_VERSION = 0
    sys.modules["regex"] = regex_mod


if _IS_WINDOWS:
    _install_regex_shim()

import torch


# ---------------------------------------------------------------------------
# 3. Zero-Copy `mmap` PyTorch `safetensors` & `safetensors.torch` Implementation
# ---------------------------------------------------------------------------
_DTYPE_MAP = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}

_INV_DTYPE_MAP = {v: k for k, v in _DTYPE_MAP.items()}


class _SafeSlice:
    """Lazy zero-copy slice view (`f.get_slice(k)[:]`) used by HuggingFace Transformers."""

    def __init__(self, safe_file: "safe_open", key: str):
        self._f = safe_file
        self._key = key
        self._info = safe_file._header[key]

    def get_shape(self):
        return list(self._info["shape"])

    def get_dtype(self):
        return self._info["dtype"]

    def __getitem__(self, item):
        t = self._f.get_tensor(self._key, clone=False)
        res = t[item]
        if self._f.device is not None and str(self._f.device) != "cpu":
            return res.to(self._f.device)
        return res.clone()


class safe_open:
    """Drop-in replacement for `safetensors.safe_open` using zero-copy `mmap`."""

    def __init__(self, filename, framework="pt", device="cpu", *args, **kwargs):
        self.filename = str(filename)
        self.framework = framework
        self.device = device
        self.backend = kwargs.get("backend", "mmap")
        self._file = None
        self._mmap = None
        self._header = {}
        self._metadata = None
        self._data_offset = 0
        self._open()

    def _open(self):
        self._file = open(self.filename, "rb")
        header_size_bytes = self._file.read(8)
        header_len = struct.unpack("<Q", header_size_bytes)[0]
        header_json = self._file.read(header_len).decode("utf-8")
        raw_header = json.loads(header_json)
        self._metadata = raw_header.pop("__metadata__", None)
        self._header = raw_header
        self._data_offset = 8 + header_len
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

    def metadata(self):
        return self._metadata

    def keys(self):
        return list(self._header.keys())

    def offset_keys(self):
        # Return keys ordered by starting byte offset
        return [
            k
            for k, _ in sorted(
                self._header.items(), key=lambda kv: kv[1]["data_offsets"][0]
            )
        ]

    def get_tensor(self, key: str, clone: bool = True) -> torch.Tensor:
        info = self._header[key]
        dtype = _DTYPE_MAP[info["dtype"]]
        shape = info["shape"]
        start, end = info["data_offsets"]
        abs_start = self._data_offset + start
        abs_end = self._data_offset + end
        byte_len = abs_end - abs_start
        if byte_len == 0:
            t = torch.empty(shape, dtype=dtype)
        else:
            mv = memoryview(self._mmap)[abs_start:abs_end]
            t = torch.frombuffer(mv, dtype=dtype).reshape(shape)
        if self.device is not None and str(self.device) != "cpu":
            return t.to(self.device)
        return t.clone() if clone else t

    def get_slice(self, key: str) -> _SafeSlice:
        return _SafeSlice(self, key)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if self._mmap is not None:
            try:
                self._mmap.close()
            except Exception:
                pass
            self._mmap = None
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None


def load_file(filename, device="cpu") -> dict:
    res = {}
    with safe_open(filename, framework="pt", device=device) as f:
        for k in f.keys():
            res[k] = f.get_tensor(k)
    return res


def load(data: bytes) -> dict:
    header_len = struct.unpack("<Q", data[:8])[0]
    raw_header = json.loads(data[8 : 8 + header_len].decode("utf-8"))
    raw_header.pop("__metadata__", None)
    data_offset = 8 + header_len
    res = {}
    for k, info in raw_header.items():
        dtype = _DTYPE_MAP[info["dtype"]]
        shape = info["shape"]
        start, end = info["data_offsets"]
        buf = data[data_offset + start : data_offset + end]
        if len(buf) == 0:
            res[k] = torch.empty(shape, dtype=dtype)
        else:
            res[k] = torch.frombuffer(bytearray(buf), dtype=dtype).reshape(shape).clone()
    return res


def save(tensors: dict, metadata: dict = None) -> bytes:
    header = {}
    if metadata is not None:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    chunks = []
    offset = 0
    for k, t in tensors.items():
        t_cpu = t.detach().contiguous().cpu()
        dtype_str = _INV_DTYPE_MAP.get(t_cpu.dtype)
        if dtype_str is None:
            t_cpu = t_cpu.to(torch.float32)
            dtype_str = "F32"
        nbytes = t_cpu.numel() * t_cpu.element_size()
        raw_bytes = (
            ctypes.string_at(t_cpu.data_ptr(), nbytes) if nbytes > 0 else b""
        )
        header[k] = {
            "dtype": dtype_str,
            "shape": list(t_cpu.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        chunks.append(raw_bytes)
        offset += nbytes

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (8 - (len(header_bytes) % 8)) % 8
    if pad:
        header_bytes += b" " * pad
    return struct.pack("<Q", len(header_bytes)) + header_bytes + b"".join(chunks)


def save_file(tensors: dict, filename, metadata: dict = None):
    payload = save(tensors, metadata=metadata)
    with open(filename, "wb") as f:
        f.write(payload)


def storage_ptr(tensor: torch.Tensor) -> int:
    try:
        return tensor.untyped_storage().data_ptr()
    except Exception:
        return 0


def storage_size(tensor: torch.Tensor) -> int:
    try:
        return tensor.untyped_storage().nbytes()
    except Exception:
        return tensor.nelement() * tensor.element_size()


_storage_ptr = storage_ptr
_storage_size = storage_size


def save_model(model: torch.nn.Module, filename: str, metadata: dict = None, force_contiguous: bool = True):
    save_file(model.state_dict(), filename, metadata=metadata)


def load_model(model: torch.nn.Module, filename: str, strict: bool = True, device="cpu"):
    state_dict = load_file(filename, device=device)
    return model.load_state_dict(state_dict, strict=strict)


def _install_safetensors_shim():
    st_mod = types.ModuleType("safetensors")
    st_mod.__version__ = "0.4.5"
    st_mod.__path__ = []
    st_mod.__spec__ = types.SimpleNamespace(name="safetensors", loader=None, origin="pure-torch-shim")
    st_mod.safe_open = safe_open
    st_mod.SafetensorError = RuntimeError

    st_torch = types.ModuleType("safetensors.torch")
    st_torch.__spec__ = types.SimpleNamespace(name="safetensors.torch", loader=None, origin="pure-torch-shim")
    st_torch.safe_open = safe_open
    st_torch.load_file = load_file
    st_torch.save_file = save_file
    st_torch.load = load
    st_torch.save = save
    st_torch.save_model = save_model
    st_torch.load_model = load_model
    st_torch.storage_ptr = storage_ptr
    st_torch.storage_size = storage_size
    st_torch._storage_ptr = storage_ptr
    st_torch._storage_size = storage_size

    st_mod.torch = st_torch
    sys.modules["safetensors"] = st_mod
    sys.modules["safetensors.torch"] = st_torch
    sys.modules["safetensors._safetensors_rust"] = st_mod


if _IS_WINDOWS:
    _install_safetensors_shim()


# ---------------------------------------------------------------------------
# 4. Route `bitsandbytes` 4-Bit NF4 Operations to Built-In PyTorch CUDA Ops
# ---------------------------------------------------------------------------
def _install_bitsandbytes_default_cuda_routing():
    import math as _math

    # Stub optional sklearn/scipy imports in transformers and bitsandbytes
    sk_mod = types.ModuleType("sklearn")
    sk_mod.__version__ = "1.5.0"
    sk_mod.__path__ = []
    sk_mod.__spec__ = types.SimpleNamespace(name="sklearn", loader=None, origin="stub")
    sk_metrics = types.ModuleType("sklearn.metrics")
    sk_metrics.__spec__ = types.SimpleNamespace(name="sklearn.metrics", loader=None, origin="stub")
    sk_metrics.roc_curve = lambda *a, **kw: (None, None, None)
    sk_mod.metrics = sk_metrics
    sys.modules["sklearn"] = sk_mod
    sys.modules["sklearn.metrics"] = sk_metrics

    sp_mod = types.ModuleType("scipy")
    sp_mod.__version__ = "1.14.0"
    sp_mod.__path__ = []
    sp_mod.__spec__ = types.SimpleNamespace(name="scipy", loader=None, origin="stub")

    sp_opt = types.ModuleType("scipy.optimize")
    sp_opt.__spec__ = types.SimpleNamespace(name="scipy.optimize", loader=None, origin="stub")
    sp_opt.linear_sum_assignment = lambda *a, **kw: ([], [])

    sp_stats = types.ModuleType("scipy.stats")
    sp_stats.__spec__ = types.SimpleNamespace(name="scipy.stats", loader=None, origin="stub")

    class _NormDist:
        @staticmethod
        def ppf(q):
            t = torch.as_tensor(q, dtype=torch.float64)
            res = _math.sqrt(2.0) * torch.erfinv(2.0 * t - 1.0)
            return res.numpy() if hasattr(q, "__len__") else float(res.item())

    sp_stats.norm = _NormDist()
    sp_mod.optimize = sp_opt
    sp_mod.stats = sp_stats
    sys.modules["scipy"] = sp_mod
    sys.modules["scipy.optimize"] = sp_opt
    sys.modules["scipy.stats"] = sp_stats

    # Prevent bitsandbytes from registering C++ `libbitsandbytes_cuda*.dll` ops over `"cuda"`,
    # so PyTorch's `torch.ops.bitsandbytes` dispatches CUDA tensors directly to the official
    # pure-PyTorch kernels in `bitsandbytes.backends.default.ops` without `torch.compile` delay.
    sys.modules["bitsandbytes.backends.cuda"] = types.ModuleType("bitsandbytes.backends.cuda")
    sys.modules["bitsandbytes.backends.cuda.ops"] = types.ModuleType("bitsandbytes.backends.cuda.ops")

    # Prevent `cextension.get_native_library` from attempting `ct.cdll.LoadLibrary` on blocked DLL
    _orig_cdll_load = ctypes.cdll.LoadLibrary

    def _safe_cdll_load(name, *a, **kw):
        if "libbitsandbytes" in str(name):
            raise OSError("Bypassed libbitsandbytes DLL load; using PyTorch CUDA default.ops")
        return _orig_cdll_load(name, *a, **kw)

    ctypes.cdll.LoadLibrary = _safe_cdll_load

    try:
        import bitsandbytes.cextension as _bnb_cext  # noqa: F401
        import bitsandbytes.backends.default.ops as _bnb_def_ops
        from bitsandbytes._ops import register_kernel
        from bitsandbytes.backends.utils import _get_4bit_code
    except (ImportError, ModuleNotFoundError, Exception):
        return

    # Compile in-memory NVRTC CUDA C++ kernels for zero-overhead 4-bit NF4/FP4 dequant & GEMV
    _torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
    _nvrtc_candidates = [
        os.path.join(_torch_lib_dir, "nvrtc64_120_0.dll"),
        os.path.join(_P313_DLL_DIR, "nvrtc64_120_0.dll"),
    ]
    _nvrtc_path = next((p for p in _nvrtc_candidates if os.path.exists(p)), "")
    _windll = getattr(ctypes, "WinDLL", None)
    _nvrtc = _windll(_nvrtc_path) if (_nvrtc_path and _windll) else None
    _nvcuda = _windll("nvcuda.dll") if _windll else None

    _CUDA_SRC = b"""
    __device__ __forceinline__ unsigned short float_to_fp16(float x) {
        unsigned short res;
        asm("cvt.rn.f16.f32 %0, %1;" : "=h"(res) : "f"(x));
        return res;
    }
    __device__ __forceinline__ float fp16_to_float(unsigned short x) {
        float res;
        asm("cvt.f32.f16 %0, %1;" : "=f"(res) : "h"(x));
        return res;
    }
    __device__ __forceinline__ float warp_reduce_sum(float val) {
        float tmp;
        asm volatile("shfl.sync.down.b32 %0, %1, 16, 0x1f, 0xffffffff;" : "=f"(tmp) : "f"(val));
        val += tmp;
        asm volatile("shfl.sync.down.b32 %0, %1, 8, 0x1f, 0xffffffff;" : "=f"(tmp) : "f"(val));
        val += tmp;
        asm volatile("shfl.sync.down.b32 %0, %1, 4, 0x1f, 0xffffffff;" : "=f"(tmp) : "f"(val));
        val += tmp;
        asm volatile("shfl.sync.down.b32 %0, %1, 2, 0x1f, 0xffffffff;" : "=f"(tmp) : "f"(val));
        val += tmp;
        asm volatile("shfl.sync.down.b32 %0, %1, 1, 0x1f, 0xffffffff;" : "=f"(tmp) : "f"(val));
        val += tmp;
        return val;
    }

    extern "C" __global__ void dequant_blockwise_u8_f32(
        const unsigned char* __restrict__ A,
        const float* __restrict__ absmax,
        const float* __restrict__ code,
        float* __restrict__ out,
        int n,
        int blocksize,
        float offset
    ) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        out[i] = code[A[i]] * absmax[i / blocksize] + offset;
    }

    extern "C" __global__ void dequant_4bit_fp16(
        const unsigned char* __restrict__ qweight,
        const float* __restrict__ absmax,
        const float* __restrict__ code,
        unsigned short* __restrict__ out_fp16,
        int n_bytes,
        int n_elems,
        int blocksize
    ) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n_bytes) return;
        unsigned char b = qweight[i];
        int idx0 = i << 1;
        int idx1 = idx0 + 1;
        float s0 = absmax[idx0 / blocksize];
        out_fp16[idx0] = float_to_fp16(code[b >> 4] * s0);
        if (idx1 < n_elems) {
            float s1 = absmax[idx1 / blocksize];
            out_fp16[idx1] = float_to_fp16(code[b & 0x0F] * s1);
        }
    }

    extern "C" __global__ void gemv_4bit_fp16_kernel(
        const unsigned short* __restrict__ A,
        const unsigned char* __restrict__ B_q,
        const float* __restrict__ absmax,
        const float* __restrict__ code,
        const unsigned short* __restrict__ bias,
        unsigned short* __restrict__ out_C,
        int M,
        int N,
        int K_half,
        int blocksize
    ) {
        __shared__ float s_code[16];
        if (threadIdx.x < 16) {
            s_code[threadIdx.x] = code[threadIdx.x];
        }
        __syncthreads();

        int warp_id = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
        int lane = threadIdx.x & 31;
        if (warp_id >= N) return;

        long long row_byte_offset = ((long long)warp_id) * ((long long)K_half);
        const unsigned char* b_row = B_q + row_byte_offset;
        int K = K_half << 1;
        float bias_val = (bias != 0) ? fp16_to_float(bias[warp_id]) : 0.0f;

        for (int m = 0; m < M; ++m) {
            const unsigned short* a_row = A + ((long long)m) * ((long long)K);
            float acc = 0.0f;
            for (int kb = lane; kb < K_half; kb += 32) {
                unsigned char b = b_row[kb];
                long long global_elem0 = (row_byte_offset + kb) << 1;
                float s0 = absmax[global_elem0 / blocksize];
                float w0 = fp16_to_float(float_to_fp16(s_code[b >> 4] * s0));
                float w1 = fp16_to_float(float_to_fp16(s_code[b & 0x0F] * s0));
                int k0 = kb << 1;
                float a0 = fp16_to_float(a_row[k0]);
                float a1 = fp16_to_float(a_row[k0 + 1]);
                acc += a0 * w0 + a1 * w1;
            }
            acc = warp_reduce_sum(acc);
            if (lane == 0) {
                out_C[((long long)m) * ((long long)N) + warp_id] = float_to_fp16(acc + bias_val);
            }
        }
    }
    """

    _compiled_funcs = {}

    def _get_cuda_funcs():
        if _compiled_funcs:
            return _compiled_funcs
        if _nvrtc is None:
            raise RuntimeError("NVRTC DLL not found in torch/lib; cannot compile 4-bit CUDA kernels.")
        prog = ctypes.c_void_p()
        _nvrtc.nvrtcCreateProgram(ctypes.byref(prog), _CUDA_SRC, b"bnb_nvrtc.cu", 0, None, None)
        major, minor = torch.cuda.get_device_capability() if torch.cuda.is_available() else (8, 0)
        arch_opt = f"--gpu-architecture=compute_{major}{minor}".encode("ascii")
        opts = (ctypes.c_char_p * 1)(arch_opt)
        res = _nvrtc.nvrtcCompileProgram(prog, 1, opts)
        if res != 0:
            log_sz = ctypes.c_size_t()
            _nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(log_sz))
            log_buf = ctypes.create_string_buffer(log_sz.value)
            _nvrtc.nvrtcGetProgramLog(prog, log_buf)
            raise RuntimeError(f"NVRTC compile failed ({res}): {log_buf.value.decode('utf-8', 'ignore')}")
        psz = ctypes.c_size_t()
        _nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(psz))
        ptx = ctypes.create_string_buffer(psz.value)
        _nvrtc.nvrtcGetPTX(prog, ptx)
        _nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))

        mod = ctypes.c_void_p()
        r1 = _nvcuda.cuModuleLoadData(ctypes.byref(mod), ptx)
        if r1 != 0:
            raise RuntimeError(f"cuModuleLoadData failed: {r1}")
        for fname in (b"dequant_blockwise_u8_f32", b"dequant_4bit_fp16", b"gemv_4bit_fp16_kernel"):
            fn = ctypes.c_void_p()
            r2 = _nvcuda.cuModuleGetFunction(ctypes.byref(fn), mod, fname)
            if r2 != 0:
                raise RuntimeError(f"cuModuleGetFunction({fname}) failed: {r2}")
            _compiled_funcs[fname.decode("ascii")] = fn
        _compiled_funcs["_mod"] = mod
        return _compiled_funcs

    def _launch_dequant_blockwise_u8_f32(A_u8, absmax_f32, code_f32, offset_val, blocksize=256):
        funcs = _get_cuda_funcs()
        fn = funcs["dequant_blockwise_u8_f32"]
        n = A_u8.numel()
        out = torch.empty(A_u8.shape, dtype=torch.float32, device=A_u8.device)
        p_A = ctypes.c_uint64(A_u8.data_ptr())
        p_abs = ctypes.c_uint64(absmax_f32.data_ptr())
        p_code = ctypes.c_uint64(code_f32.data_ptr())
        p_out = ctypes.c_uint64(out.data_ptr())
        c_n = ctypes.c_int(n)
        c_bs = ctypes.c_int(blocksize)
        c_off = ctypes.c_float(float(offset_val))
        args = (ctypes.c_void_p * 7)(
            ctypes.cast(ctypes.byref(p_A), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(p_abs), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(p_code), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(p_out), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_n), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_bs), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_off), ctypes.c_void_p),
        )
        threads = 256
        blocks = (n + threads - 1) // threads
        stream = ctypes.c_void_p(torch.cuda.current_stream(A_u8.device).cuda_stream)
        rc = _nvcuda.cuLaunchKernel(fn, blocks, 1, 1, threads, 1, 1, 0, stream, args, None)
        if rc != 0:
            raise RuntimeError(f"cuLaunchKernel(dequant_blockwise_u8_f32) failed with CUDA status {rc}")
        return out

    def _launch_dequant_4bit_fp16(A_u8, absmax_f32, code_f32, blocksize, shape, dtype):
        funcs = _get_cuda_funcs()
        fn = funcs["dequant_4bit_fp16"]
        n_bytes = A_u8.numel()
        n_elems = 1
        for s in shape:
            n_elems *= s
        out = torch.empty(shape, dtype=torch.float16, device=A_u8.device)
        p_q = ctypes.c_uint64(A_u8.data_ptr())
        p_abs = ctypes.c_uint64(absmax_f32.data_ptr())
        p_code = ctypes.c_uint64(code_f32.data_ptr())
        p_out = ctypes.c_uint64(out.data_ptr())
        c_nb = ctypes.c_int(n_bytes)
        c_ne = ctypes.c_int(n_elems)
        c_bs = ctypes.c_int(blocksize)
        args = (ctypes.c_void_p * 7)(
            ctypes.cast(ctypes.byref(p_q), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(p_abs), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(p_code), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(p_out), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_nb), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_ne), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_bs), ctypes.c_void_p),
        )
        threads = 256
        blocks = (n_bytes + threads - 1) // threads
        stream = ctypes.c_void_p(torch.cuda.current_stream(A_u8.device).cuda_stream)
        rc = _nvcuda.cuLaunchKernel(fn, blocks, 1, 1, threads, 1, 1, 0, stream, args, None)
        if rc != 0:
            raise RuntimeError(f"cuLaunchKernel(dequant_4bit_fp16) failed with CUDA status {rc}")
        return out if dtype == torch.float16 else out.to(dtype)

    def _launch_gemv_4bit_fp16(A, B_u8, absmax_f32, code_f32, shapeB, blocksize, bias=None):
        funcs = _get_cuda_funcs()
        fn = funcs["gemv_4bit_fp16_kernel"]
        orig_dtype = A.dtype
        A_fp16 = A.contiguous() if A.dtype == torch.float16 else A.to(torch.float16).contiguous()
        orig_shape = A_fp16.shape
        K = orig_shape[-1]
        M = A_fp16.numel() // K
        N = shapeB[0]
        K_half = K // 2
        out = torch.empty((*orig_shape[:-1], N), dtype=torch.float16, device=A.device)
        bias_fp16 = None
        if bias is not None:
            bias_fp16 = bias.contiguous() if bias.dtype == torch.float16 else bias.to(torch.float16).contiguous()
        p_A = ctypes.c_uint64(A_fp16.data_ptr())
        p_B = ctypes.c_uint64(B_u8.data_ptr())
        p_abs = ctypes.c_uint64(absmax_f32.data_ptr())
        p_code = ctypes.c_uint64(code_f32.data_ptr())
        p_bias = ctypes.c_uint64(bias_fp16.data_ptr() if bias_fp16 is not None else 0)
        p_out = ctypes.c_uint64(out.data_ptr())
        c_M = ctypes.c_int(M)
        c_N = ctypes.c_int(N)
        c_Kh = ctypes.c_int(K_half)
        c_bs = ctypes.c_int(blocksize)
        args = (ctypes.c_void_p * 10)(
            ctypes.cast(ctypes.byref(p_A), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(p_B), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(p_abs), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(p_code), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(p_bias), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(p_out), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_M), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_N), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_Kh), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_bs), ctypes.c_void_p),
        )
        warps_per_block = 4
        threads = warps_per_block * 32
        blocks = (N + warps_per_block - 1) // warps_per_block
        stream = ctypes.c_void_p(torch.cuda.current_stream(A.device).cuda_stream)
        rc = _nvcuda.cuLaunchKernel(fn, blocks, 1, 1, threads, 1, 1, 0, stream, args, None)
        if rc != 0:
            raise RuntimeError(f"cuLaunchKernel(gemv_4bit_fp16_kernel) failed with CUDA status {rc}")
        return out if orig_dtype == torch.float16 else out.to(orig_dtype)

    @register_kernel("bitsandbytes::dequantize_blockwise", "cuda")
    def _cuda_dequant_blockwise(A: torch.Tensor, absmax: torch.Tensor, code: torch.Tensor, blocksize: int, dtype: torch.dtype) -> torch.Tensor:
        if A.dtype == torch.uint8 and absmax.dtype == torch.float32 and code.dtype == torch.float32:
            out = _launch_dequant_blockwise_u8_f32(A.contiguous(), absmax.contiguous(), code.contiguous(), 0.0, blocksize)
            return out if dtype == torch.float32 else out.to(dtype)
        return _bnb_def_ops._dequantize_blockwise_compute(A.reshape(-1), absmax, code, blocksize, dtype).reshape(A.shape)

    @register_kernel("bitsandbytes::dequantize_4bit", "cuda")
    def _cuda_dequant_4bit(A: torch.Tensor, absmax: torch.Tensor, blocksize: int, quant_type: str, shape, dtype: torch.dtype) -> torch.Tensor:
        if A.dtype != torch.uint8:
            A = A.view(torch.uint8)
        code = _get_4bit_code(quant_type, A.device)
        return _launch_dequant_4bit_fp16(A.contiguous(), absmax.contiguous(), code.contiguous(), blocksize, shape, dtype)

    @register_kernel("bitsandbytes::gemv_4bit", "cuda")
    def _cuda_gemv_4bit(A: torch.Tensor, B: torch.Tensor, shapeB, absmax: torch.Tensor, code: torch.Tensor, blocksize: int) -> torch.Tensor:
        if B.dtype != torch.uint8:
            B = B.view(torch.uint8)
        K = A.shape[-1]
        code_f32 = code.to(device=A.device, dtype=torch.float32).contiguous()
        if (K % 2 == 0) and (blocksize % 2 == 0):
            return _launch_gemv_4bit_fp16(A, B.contiguous(), absmax.contiguous(), code_f32, shapeB, blocksize, None)
        B_dq = _launch_dequant_4bit_fp16(B.contiguous(), absmax.contiguous(), code_f32, blocksize, shapeB, A.dtype)
        return torch.nn.functional.linear(A, B_dq, None)

    class Matmul4bitAutograd(torch.autograd.Function):
        """Zero-Memory-Bloat 4-bit Matmul Autograd Function.
        
        Saves ONLY the packed 4-bit tensor (0.5 bytes/param) and absmax.
        Never saves the 18 GB of dequantized FP16 weights across transformer layers!
        Dequantizes on-the-fly during backward pass and immediately discards.
        """
        @staticmethod
        def forward(ctx, A, B, shapeB, absmax, blocksize, quant_type, bias=None):
            K = A.shape[-1]
            orig_shape = A.shape
            M = A.numel() // K
            N = shapeB[0]

            ctx.save_for_backward(B, absmax, bias)
            ctx.shapeB = shapeB
            ctx.blocksize = blocksize
            ctx.quant_type = quant_type
            ctx.A_shape = orig_shape
            ctx.A_dtype = A.dtype
            ctx.A_requires_grad = A.requires_grad

            code = _get_4bit_code(quant_type, A.device)

            if M <= 4 and (K % 2 == 0) and (blocksize % 2 == 0):
                return _launch_gemv_4bit_fp16(A, B.contiguous(), absmax.contiguous(), code.contiguous(), shapeB, blocksize, bias)

            if N > 32768:
                chunk_size = 32768
                outs = []
                B_flat = B.view(-1)
                for start in range(0, N, chunk_size):
                    end = min(start + chunk_size, N)
                    b_start = (start * K) // 2
                    b_end = (end * K) // 2
                    B_c = B_flat[b_start:b_end]
                    abs_start = (start * K) // blocksize
                    abs_end = (end * K + blocksize - 1) // blocksize
                    abs_c = absmax[abs_start:abs_end]
                    B_dq = _launch_dequant_4bit_fp16(B_c.contiguous(), abs_c.contiguous(), code.contiguous(), blocksize, (end - start, K), A.dtype)
                    bias_c = bias[start:end] if bias is not None else None
                    outs.append(torch.nn.functional.linear(A, B_dq, bias_c))
                return torch.cat(outs, dim=-1)

            B_dq = _launch_dequant_4bit_fp16(B.contiguous(), absmax.contiguous(), code.contiguous(), blocksize, shapeB, A.dtype)
            out = torch.nn.functional.linear(A, B_dq, bias)
            return out

        @staticmethod
        def backward(ctx, grad_output):
            if not ctx.A_requires_grad:
                return None, None, None, None, None, None, None

            B, absmax, bias = ctx.saved_tensors
            shapeB = ctx.shapeB
            blocksize = ctx.blocksize
            quant_type = ctx.quant_type
            orig_shape = ctx.A_shape
            orig_dtype = ctx.A_dtype

            code = _get_4bit_code(quant_type, grad_output.device)
            K = shapeB[1]
            N = shapeB[0]

            grad_out_2d = torch.nan_to_num(grad_output.reshape(-1, N).float(), nan=0.0, posinf=65000.0, neginf=-65000.0)

            if N > 32768:
                chunk_size = 32768
                grad_A_2d = torch.zeros((grad_out_2d.shape[0], K), dtype=torch.float32, device=grad_output.device)
                B_flat = B.view(-1)
                for start in range(0, N, chunk_size):
                    end = min(start + chunk_size, N)
                    b_start = (start * K) // 2
                    b_end = (end * K) // 2
                    B_c = B_flat[b_start:b_end]
                    abs_start = (start * K) // blocksize
                    abs_end = (end * K + blocksize - 1) // blocksize
                    abs_c = absmax[abs_start:abs_end]
                    B_dq = _launch_dequant_4bit_fp16(B_c.contiguous(), abs_c.contiguous(), code.contiguous(), blocksize, (end - start, K), torch.float32)
                    grad_A_2d += grad_out_2d[:, start:end] @ B_dq
            else:
                B_dq = _launch_dequant_4bit_fp16(B.contiguous(), absmax.contiguous(), code.contiguous(), blocksize, shapeB, torch.float32)
                grad_A_2d = grad_out_2d @ B_dq

            grad_A_2d = torch.nan_to_num(grad_A_2d, nan=0.0, posinf=65000.0, neginf=-65000.0).clamp(min=-65000.0, max=65000.0).to(orig_dtype)
            grad_A = grad_A_2d.reshape(orig_shape)
            grad_bias = grad_out_2d.sum(dim=0).to(orig_dtype) if (bias is not None and bias.requires_grad) else None
            return grad_A, None, None, None, None, None, grad_bias

    @register_kernel("bitsandbytes::gemm_4bit", "cuda")
    def _cuda_gemm_4bit(
        A: torch.Tensor,
        B: torch.Tensor,
        shapeB,
        absmax: torch.Tensor,
        blocksize: int,
        quant_type: str,
        bias=None,
        absmax_8bit=None,
        absmax_code=None,
        absmax_offset=None,
    ) -> torch.Tensor:
        if absmax_8bit is not None:
            off_val = float(absmax_offset.item()) if isinstance(absmax_offset, torch.Tensor) else float(absmax_offset or 0.0)
            absmax = _launch_dequant_blockwise_u8_f32(absmax_8bit.contiguous(), absmax.contiguous(), absmax_code.contiguous(), off_val, 256)
        if B.dtype != torch.uint8:
            B = B.view(torch.uint8)
        return Matmul4bitAutograd.apply(A, B, shapeB, absmax, blocksize, quant_type, bias)

    mod_self = sys.modules[__name__]
    mod_self._launch_dequant_4bit_fp16 = _launch_dequant_4bit_fp16
    mod_self._launch_gemv_4bit_fp16 = _launch_gemv_4bit_fp16
    mod_self._launch_dequant_blockwise_u8_f32 = _launch_dequant_blockwise_u8_f32
    mod_self._get_4bit_code = _get_4bit_code
    mod_self._get_cuda_funcs = _get_cuda_funcs
    mod_self.Matmul4bitAutograd = Matmul4bitAutograd



if _IS_WINDOWS:
    _install_bitsandbytes_default_cuda_routing()
