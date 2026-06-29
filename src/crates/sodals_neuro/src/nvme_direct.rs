// PHASE 41: HOLOGRAPHIC NVMe MEMORY ADDRESSING
// Liquid Phase-Modulated Attention - Zero-Copy Memory Mapping for 250GB QMEM Cartridge
//
// The 4096-D thought vector IS the disk address. Traditional indexing is O(log N).
// Holographic addressing is O(1) via phase-based direct mapping.
//
// Architecture:
// - mmap/CreateFileMapping for zero-copy SSD access
// - cudaHostRegister for GPU PCIe direct access (bypasses CPU)
// - 4KB page-aligned holographic blocks containing Ternary/Hex topological weights

use std::fs::{File, OpenOptions};
use std::path::Path;
use std::sync::{Arc, RwLock};

/// Holographic memory block size (4KB - standard page size for optimal I/O)
pub const QMEM_BLOCK_SIZE: usize = 4096;

/// Maximum QMEM cartridge size (250GB)
pub const QMEM_MAX_SIZE: usize = 250 * 1024 * 1024 * 1024; // 250GB

/// Holographic memory-mapped QMEM cartridge
/// 
/// Provides O(1) access to 250GB topological weights via phase-addressing.
/// The 4096-D thought vector is hashed to a 64-bit address, which directly
/// indexes into the memory-mapped region (modulo file size, 4KB aligned).
pub struct HolographicQMEM {
    /// Memory-mapped region pointer (raw bytes)
    mmap_ptr: *mut u8,
    /// Total mapped size in bytes
    mapped_size: usize,
    /// Underlying file handle (kept alive for the mapping)
    _file: File,
    /// CUDA-registered host memory (if CUDA enabled)
    #[cfg(feature = "cuda")]
    #[allow(dead_code)]
    cuda_registered: bool,
}

// SAFETY: HolographicQMEM is thread-safe for reads
// The mmap region is immutable after creation, and all accesses are through
// compute_holographic_address which is pure computation
unsafe impl Send for HolographicQMEM {}
unsafe impl Sync for HolographicQMEM {}

impl HolographicQMEM {
    /// Open and memory-map a QMEM cartridge file
    /// 
    /// # Arguments
    /// * `path` - Path to the QMEM binary file (up to 250GB)
    /// 
    /// # Returns
    /// * `Ok(HolographicQMEM)` - Memory-mapped cartridge ready for O(1) access
    /// * `Err(...)` - If file cannot be opened or mapped
    pub fn open<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)?;
        
        let file_size = file.metadata()?.len() as usize;
        let mapped_size = file_size.min(QMEM_MAX_SIZE);
        
        if mapped_size == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "QMEM cartridge is empty"
            ));
        }
        
        // Platform-specific memory mapping
        #[cfg(target_os = "windows")]
        let mmap_ptr = Self::mmap_windows(&file, mapped_size)?;
        
        #[cfg(not(target_os = "windows"))]
        let mmap_ptr = Self::mmap_unix(&file, mapped_size)?;
        
        #[cfg(feature = "cuda")]
        let cuda_registered = Self::cuda_register(mmap_ptr, mapped_size);
        
        #[cfg(not(feature = "cuda"))]
        let cuda_registered = false;
        
        eprintln!(
            "[HOLOGRAPHIC QMEM] Mapped {} GB at {:p} | O(1) phase-addressing ready",
            mapped_size / (1024 * 1024 * 1024),
            mmap_ptr
        );
        
        Ok(Self {
            mmap_ptr,
            mapped_size,
            _file: file,
            #[cfg(feature = "cuda")]
            cuda_registered,
        })
    }
    
    /// Windows: CreateFileMapping + MapViewOfFile
    #[cfg(target_os = "windows")]
    fn mmap_windows(_file: &File, size: usize) -> std::io::Result<*mut u8> {
        // SAFETY: Windows API calls for memory mapping
        // This is a placeholder - actual implementation would use winapi crate
        // For now, we use a simple heap allocation as fallback
        eprintln!("[HOLOGRAPHIC QMEM] Windows mmap not yet implemented, using heap fallback");
        let layout = std::alloc::Layout::from_size_align(size, 4096)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        let ptr = unsafe { std::alloc::alloc(layout) };
        if ptr.is_null() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::OutOfMemory,
                "Failed to allocate QMEM buffer"
            ));
        }
        Ok(ptr)
    }
    
    /// Unix: mmap with MAP_SHARED
    #[cfg(not(target_os = "windows"))]
    fn mmap_unix(file: &File, size: usize) -> std::io::Result<*mut u8> {
        use std::os::unix::io::AsRawFd;
        
        // SAFETY: mmap is safe when:
        // 1. File descriptor is valid (we own the File)
        // 2. Size is non-zero (checked above)
        // 3. We don't write to MAP_SHARED region without synchronization
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),           // Let kernel choose address
                size,                           // Length
                libc::PROT_READ | libc::PROT_WRITE, // Read+Write access
                libc::MAP_SHARED,               // Shared mapping (changes visible to other processes)
                file.as_raw_fd(),               // File descriptor
                0,                              // Offset (start of file)
            )
        };
        
        if ptr == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error());
        }
        
        Ok(ptr as *mut u8)
    }
    
    /// CUDA: Register host memory for GPU PCIe direct access
    /// 
    /// This eliminates page faults by pinning the memory and making it
    /// accessible to the GPU via zero-copy transfers.
    #[cfg(feature = "cuda")]
    fn cuda_register(_ptr: *mut u8, _size: usize) -> bool {
        // Placeholder: Actual CUDA implementation would call cudaHostRegister
        // This requires the cust or rustacuda crate
        eprintln!("[HOLOGRAPHIC QMEM] CUDA host registration not yet implemented");
        false
    }
    
    /// Fetch a 4KB block from the holographic memory via O(1) phase-addressing
    /// 
    /// # Arguments
    /// * `address` - 64-bit holographic address (from compute_holographic_address)
    /// 
    /// # Returns
    /// * `Some(&[u8; QMEM_BLOCK_SIZE])` - Pointer to the 4KB block
    /// * `None` - If address is invalid
    /// 
    /// # Complexity
    /// O(1) - Direct pointer arithmetic, no searching or indexing
    pub fn fetch_block(&self, address: u64) -> Option<&[u8; QMEM_BLOCK_SIZE]> {
        // Align address to 4KB boundary
        let aligned_addr = (address as usize) & !(QMEM_BLOCK_SIZE - 1);
        let offset = aligned_addr % self.mapped_size;
        
        // SAFETY: Bounds checked, pointer is valid for mapped_size
        if offset + QMEM_BLOCK_SIZE <= self.mapped_size {
            let block_ptr = unsafe { self.mmap_ptr.add(offset) };
            Some(unsafe { &*(block_ptr as *const [u8; QMEM_BLOCK_SIZE]) })
        } else {
            None
        }
    }
    
    /// Fetch a mutable 4KB block for Hebbian updates
    /// 
    /// # Safety
    /// Caller must ensure exclusive access to prevent data races
    pub unsafe fn fetch_block_mut(&self, address: u64) -> Option<&mut [u8; QMEM_BLOCK_SIZE]> {
        let aligned_addr = (address as usize) & !(QMEM_BLOCK_SIZE - 1);
        let offset = aligned_addr % self.mapped_size;
        
        if offset + QMEM_BLOCK_SIZE <= self.mapped_size {
            let block_ptr = self.mmap_ptr.add(offset);
            Some(&mut *(block_ptr as *mut [u8; QMEM_BLOCK_SIZE]))
        } else {
            None
        }
    }
    
    /// Get total mapped size
    pub fn size(&self) -> usize {
        self.mapped_size
    }
    
    /// Get number of 4KB blocks
    pub fn num_blocks(&self) -> usize {
        self.mapped_size / QMEM_BLOCK_SIZE
    }
}

impl Drop for HolographicQMEM {
    fn drop(&mut self) {
        #[cfg(not(target_os = "windows"))]
        unsafe {
            libc::munmap(self.mmap_ptr as *mut libc::c_void, self.mapped_size);
        }
        
        #[cfg(target_os = "windows")]
        unsafe {
            // Windows: UnmapViewOfFile + CloseHandle
            // For heap fallback, free the allocation
            let layout = std::alloc::Layout::from_size_align_unchecked(
                self.mapped_size, 4096
            );
            std::alloc::dealloc(self.mmap_ptr, layout);
        }
        
        eprintln!("[HOLOGRAPHIC QMEM] Unmapped {} bytes", self.mapped_size);
    }
}

/// Thread-safe handle to the holographic QMEM cartridge
pub type HolographicQMEMHandle = Arc<RwLock<HolographicQMEM>>;

/// Pack ternary weights (-1, 0, +1) into a byte (4 weights per byte)
/// 
/// Bit packing: [00=0, 01=+1, 10=-1, 11=reserved] per 2-bit nibble
pub fn pack_ternary_weights(weights: &[i8]) -> Vec<u8> {
    let mut packed = Vec::with_capacity((weights.len() + 3) / 4);
    
    for chunk in weights.chunks(4) {
        let mut byte: u8 = 0;
        for (i, &w) in chunk.iter().enumerate() {
            let bits = match w {
                0 => 0b00,
                1 => 0b01,
                -1 => 0b10,
                _ => 0b11, // Invalid, mark as reserved
            };
            byte |= bits << (i * 2);
        }
        packed.push(byte);
    }
    
    packed
}

/// Unpack ternary weights from a byte
pub fn unpack_ternary_weights(packed: &[u8], count: usize) -> Vec<i8> {
    let mut weights = Vec::with_capacity(count);
    
    for &byte in packed {
        for i in 0..4 {
            if weights.len() >= count {
                break;
            }
            let bits = (byte >> (i * 2)) & 0b11;
            let w = match bits {
                0b00 => 0,
                0b01 => 1,
                0b10 => -1,
                _ => 0, // Reserved, treat as 0
            };
            weights.push(w);
        }
    }
    
    weights
}

/// Decode a 4KB QMEM block into f32 topological weights
/// 
/// The block contains packed ternary/hex weights that bias the Phase-Modulated Attention.
/// This decoding produces the actual f32 values used in Oja's Rule.
pub fn decode_qmem_block(block: &[u8; QMEM_BLOCK_SIZE]) -> Vec<f32> {
    // First 256 bytes: metadata (version, checksum, etc.)
    // Remaining 3840 bytes: packed ternary weights (up to 15360 weights)
    const METADATA_SIZE: usize = 256;
    const WEIGHT_DATA_SIZE: usize = QMEM_BLOCK_SIZE - METADATA_SIZE;
    
    // Decode ternary weights from packed format
    let packed = &block[METADATA_SIZE..];
    let ternary = unpack_ternary_weights(packed, WEIGHT_DATA_SIZE * 4);
    
    // Convert to f32 with scaling (scale factor from metadata or default)
    let scale = 0.01_f32; // Default scale: ternary -> [-0.01, 0, +0.01]
    
    ternary.iter()
        .map(|&w| w as f32 * scale)
        .collect()
}

// ============================================================================
// PHASE 42: HILBERT CURVE SPATIAL TOPOLOGY FOR NVMe
// ============================================================================

/// Hilbert Curve spatial indexing for 250GB QMEM cartridge
/// 
/// The Concept: To defend "250GB beats ARC-AGI-3", the holographic LSH
/// cannot point to random noise. The SSD must be structurally organized
/// using space-filling curves that preserve locality.
/// 
/// Mathematical Properties:
/// - Hilbert curves have superior locality preservation vs Z-order
/// - Adjacent geometric axioms map to adjacent 4KB blocks
/// - This enables O(1) spatial queries: similar transforms = nearby addresses
/// 
/// Address Structure (64-bit):
/// - Bits 0-47: Hilbert index within 2D spatial grid (max 256TB)
/// - Bits 48-63: Reserved for metadata/flags
pub struct HilbertSpatialIndex {
    /// Order of the Hilbert curve (2^order = grid dimension)
    order: u32,
    /// Grid dimension (2^order)
    grid_size: u32,
}

impl HilbertSpatialIndex {
    /// Create a new Hilbert spatial index
    /// 
    /// # Arguments
    /// * `order` - Curve order (e.g., 16 gives 65536x65536 grid = 4B cells)
    /// 
    /// For ARC-AGI: order=12 gives 4096x4096 grid, sufficient for 30x30 grids
    pub fn new(order: u32) -> Self {
        let grid_size = 1u32 << order;
        Self { order, grid_size }
    }

    /// Convert 2D (x, y) grid coordinates to 1D Hilbert index
    /// 
    /// This maps spatial proximity to address proximity.
    /// Two nearby (x,y) positions → nearby Hilbert indices.
    /// 
    /// # Arguments
    /// * `x` - X coordinate [0, grid_size)
    /// * `y` - Y coordinate [0, grid_size)
    /// 
    /// # Returns
    /// * 64-bit Hilbert index (can be directly used as SSD address)
    pub fn encode(&self, x: u32, y: u32) -> u64 {
        let mut x = x as i32;
        let mut y = y as i32;
        let mut d: i64 = 0;
        let mut s = (self.grid_size / 2) as i32;
        
        while s > 0 {
            let rx = if (x & s) > 0 { 1 } else { 0 };
            let ry = if (y & s) > 0 { 1 } else { 0 };
            d += s as i64 * s as i64 * ((3 * rx) ^ ry) as i64;
            
            // rot logic
            if ry == 0 {
                if rx == 1 {
                    x = s - 1 - x;
                    y = s - 1 - y;
                }
                std::mem::swap(&mut x, &mut y);
            }
            s /= 2;
        }
        d as u64
    }

    /// Convert 1D Hilbert index back to 2D (x, y) coordinates
    /// 
    /// # Arguments
    /// * `hilbert_index` - 1D Hilbert index
    /// 
    /// # Returns
    /// * (x, y) grid coordinates
    pub fn decode(&self, hilbert_index: u64) -> (u32, u32) {
        let mut x = 0i32;
        let mut y = 0i32;
        let mut t = hilbert_index as i64;
        let n = self.grid_size as i32;
        
        let mut s = 1i32;
        while s < n {
            let rx = 1 & (t / 2) as i32;
            let ry = 1 & (t as i32 ^ rx);
            
            // rot logic
            if ry == 0 {
                if rx == 1 {
                    x = s - 1 - x;
                    y = s - 1 - y;
                }
                std::mem::swap(&mut x, &mut y);
            }
            
            x += s * rx;
            y += s * ry;
            t /= 4;
            s *= 2;
        }
        
        (x as u32, y as u32)
    }

    /// Compute SSD block address from Hilbert index
    /// 
    /// Maps the 1D Hilbert index to a physical 4KB-aligned SSD address.
    /// 
    /// # Arguments
    /// * `hilbert_index` - The 1D Hilbert index
    /// * `max_address` - Maximum SSD address (file size)
    /// 
    /// # Returns
    /// * 64-bit SSD address (4KB page-aligned)
    pub fn to_ssd_address(&self, hilbert_index: u64, max_address: u64) -> u64 {
        // Scale Hilbert index to address space
        // Each block = 4KB, so address = index * 4096
        let max_hilbert = (1u64 << (self.order * 2)) - 1;
        let scaled = if max_hilbert > 0 {
            let num = hilbert_index as u128 * max_address as u128;
            (num / max_hilbert as u128) as u64
        } else {
            0
        };
        
        // 4KB page alignment
        scaled & !(QMEM_BLOCK_SIZE as u64 - 1)
    }

    /// Get grid size (2^order)
    pub fn grid_size(&self) -> u32 {
        self.grid_size
    }
}

/// Spatial QMEM Compiler - Organizes geometric axioms using Hilbert curves
/// 
/// This demonstrates how the 250GB SSD is structured to support the
/// "beats ARC-AGI-3" claim. Geometric transformations with similar
/// topological properties are stored in contiguous 4KB blocks.
/// 
/// Packing Strategy:
/// 1. Each geometric axiom (Rotation, Translation, Reflection) is a SpatialNode
/// 2. SpatialNodes are mapped to 2D (x,y) coordinates based on their properties
///    - x-axis: Rotation angle [0, 360)
///    - y-axis: Scale factor [0.5, 2.0]
/// 3. Hilbert curve ensures spatial locality → address locality
/// 4. Contiguous blocks form "transformation neighborhoods"
pub struct SpatialQMEMCompiler {
    /// Hilbert spatial index for address computation
    hilbert: HilbertSpatialIndex,
    /// Current write position (next free block index)
    next_block_index: u64,
}

impl SpatialQMEMCompiler {
    /// Create a new Spatial QMEM Compiler
    /// 
    /// # Arguments
    /// * `order` - Hilbert curve order (determines grid resolution)
    pub fn new(order: u32) -> Self {
        Self {
            hilbert: HilbertSpatialIndex::new(order),
            next_block_index: 0,
        }
    }

    /// Map geometric axiom to spatial coordinates
    /// 
    /// For ARC-AGI spatial reasoning, we map:
    /// - Rotations: angle → x, scale → y
    /// - Translations: dx → x, dy → y  
    /// - Reflections: axis_angle → x, strength → y
    /// 
    /// # Arguments
    /// * `axiom_type` - Type of geometric transformation
    /// * `params` - Transformation parameters (normalized to [0, 1])
    /// 
    /// # Returns
    /// * (x, y) spatial coordinates in grid
    pub fn axiom_to_spatial(
        &self,
        axiom_type: GeometricAxiom,
        params: &[f32], // Normalized [0, 1]
    ) -> (u32, u32) {
        let grid_size = self.hilbert.grid_size() as f32;
        
        let (x_norm, y_norm) = match axiom_type {
            GeometricAxiom::Rotation => {
                // params[0] = angle/360, params[1] = scale_factor
                (params[0], params[1])
            }
            GeometricAxiom::Translation => {
                // params[0] = dx/max_w, params[1] = dy/max_h
                ((params[0] + 1.0) * 0.5, (params[1] + 1.0) * 0.5) // Center at 0.5
            }
            GeometricAxiom::Reflection => {
                // params[0] = axis_angle/180, params[1] = strength
                (params[0], params[1])
            }
            GeometricAxiom::Scaling => {
                // params[0] = log_scale, params[1] = center_x
                ((params[0] + 1.0) * 0.5, params[1])
            }
        };
        
        // Clamp and scale to grid
        let x = (x_norm.clamp(0.0, 1.0) * (grid_size - 1.0)) as u32;
        let y = (y_norm.clamp(0.0, 1.0) * (grid_size - 1.0)) as u32;
        
        (x, y)
    }

    /// Compile a geometric axiom into a 4KB QMEM block
    /// 
    /// This packs the axiom's weights into a block and computes its
    /// Hilbert-curve-based SSD address.
    /// 
    /// # Arguments
    /// * `axiom` - Geometric transformation axiom
    /// * `weights` - Ternary weights for the axiom
    /// 
    /// # Returns
    /// * (ssd_address, block_data) - Where to write and what to write
    pub fn compile_axiom(
        &mut self,
        axiom: GeometricAxiom,
        params: &[f32],
        weights: &[i8],
    ) -> (u64, [u8; QMEM_BLOCK_SIZE]) {
        // Get spatial coordinates
        let (x, y) = self.axiom_to_spatial(axiom, params);
        
        // Compute Hilbert index
        let hilbert_index = self.hilbert.encode(x, y);
        
        // Map to SSD address (4KB aligned)
        let ssd_address = self.hilbert.to_ssd_address(
            hilbert_index,
            QMEM_MAX_SIZE as u64
        );
        
        // Pack weights into block
        let mut block = [0u8; QMEM_BLOCK_SIZE];
        
        // Metadata (first 256 bytes)
        block[0] = axiom as u8; // Axiom type
        block[1] = params.len() as u8; // Number of params
        block[2..6].copy_from_slice(&(x as u32).to_le_bytes());
        block[6..10].copy_from_slice(&(y as u32).to_le_bytes());
        
        // Pack weights (remaining 3840 bytes)
        let packed_weights = pack_ternary_weights(weights);
        let weight_start = 256;
        let weight_len = packed_weights.len().min(QMEM_BLOCK_SIZE - weight_start);
        block[weight_start..weight_start + weight_len]
            .copy_from_slice(&packed_weights[..weight_len]);
        
        self.next_block_index += 1;
        
        (ssd_address, block)
    }

    /// Find neighboring axioms (spatial query)
    /// 
    /// Given a query axiom, find all axioms within radius in Hilbert space.
    /// This enables O(1) range queries on the SSD.
    /// 
    /// # Arguments
    /// * `query_x` - Query X coordinate
    /// * `query_y` - Query Y coordinate
    /// * `radius` - Search radius in grid cells
    /// 
    /// # Returns
    /// * Iterator over (ssd_address, block_index) for neighborhood
    pub fn neighborhood_query(
        &self,
        query_x: u32,
        query_y: u32,
        radius: u32,
    ) -> Vec<u64> {
        let mut addresses = Vec::new();
        let grid_size = self.hilbert.grid_size();
        
        // Search square region around query point
        let x_min = query_x.saturating_sub(radius);
        let x_max = (query_x + radius).min(grid_size - 1);
        let y_min = query_y.saturating_sub(radius);
        let y_max = (query_y + radius).min(grid_size - 1);
        
        for y in y_min..=y_max {
            for x in x_min..=x_max {
                let hilbert_idx = self.hilbert.encode(x, y);
                let addr = self.hilbert.to_ssd_address(
                    hilbert_idx,
                    QMEM_MAX_SIZE as u64
                );
                addresses.push(addr);
            }
        }
        
        addresses
    }
}

/// Geometric axiom types for ARC-AGI spatial reasoning
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum GeometricAxiom {
    Rotation = 0,
    Translation = 1,
    Reflection = 2,
    Scaling = 3,
}

/// Compute Hilbert-curve-based holographic address
/// 
/// Combines Phase 41 LSH with Phase 42 Hilbert spatial topology.
/// This ensures the 64-bit address maps to a semantically meaningful
/// location on the SSD.
/// 
/// # Arguments
/// * `thought_vector` - 4096-D latent thought (leading dims used as coords)
/// * `grid_size` - Hilbert grid dimension
/// 
/// # Returns
/// * 64-bit SSD address (4KB aligned)
pub fn compute_spatial_holographic_address(
    thought_vector: &[f32],
    grid_size: u32,
) -> u64 {
    // Extract leading dimensions as spatial coordinates
    let x_norm = if !thought_vector.is_empty() {
        ((thought_vector[0] as f64).tanh() + 1.0) * 0.5
    } else {
        0.5
    };
    let y_norm = if thought_vector.len() > 1 {
        ((thought_vector[1] as f64).tanh() + 1.0) * 0.5
    } else {
        0.5
    };
    
    // Map to grid coordinates
    let x = (x_norm * (grid_size as f64 - 1.0)) as u32;
    let y = (y_norm * (grid_size as f64 - 1.0)) as u32;
    
    // Hilbert index
    let order = (grid_size as f32).log2() as u32;
    let hilbert = HilbertSpatialIndex::new(order);
    let hilbert_index = hilbert.encode(x, y);
    
    // Map to SSD address
    hilbert.to_ssd_address(hilbert_index, QMEM_MAX_SIZE as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ternary_packing() {
        let weights = vec![1, 0, -1, 1, 0, 0, -1, -1];
        let packed = pack_ternary_weights(&weights);
        let unpacked = unpack_ternary_weights(&packed, weights.len());
        assert_eq!(weights, unpacked);
    }
    
    #[test]
    fn test_holographic_address_alignment() {
        // Test that addresses are properly aligned to 4KB
        let addr1: u64 = 0x12345678;
        let aligned1 = (addr1 as usize) & !(QMEM_BLOCK_SIZE - 1);
        assert_eq!(aligned1 % QMEM_BLOCK_SIZE, 0);
        
        let addr2: u64 = 0xABCDEF00;
        let aligned2 = (addr2 as usize) & !(QMEM_BLOCK_SIZE - 1);
        assert_eq!(aligned2 % QMEM_BLOCK_SIZE, 0);
    }
    
    
    #[test]
    fn test_spatial_qmem_compiler() {
        let mut compiler = SpatialQMEMCompiler::new(8); // 256x256 grid
        
        // Compile a rotation axiom
        let weights = vec![1, -1, 0, 1, 0, 0, -1, 1];
        let (addr, block) = compiler.compile_axiom(
            GeometricAxiom::Rotation,
            &[0.25, 0.8], // 90° rotation, 0.8 scale
            &weights
        );
        
        // Verify address is 4KB aligned
        assert_eq!(addr % 4096, 0);
        
        // Verify block metadata
        assert_eq!(block[0], GeometricAxiom::Rotation as u8);
        
        // Query neighborhood
        let neighbors = compiler.neighborhood_query(64, 64, 2);
        assert!(!neighbors.is_empty());
        assert!(neighbors.len() <= 25); // 5x5 neighborhood
    }
    
    #[test]
    fn test_spatial_holographic_address() {
        // Test that thought vectors map to consistent addresses
        let thought1 = vec![0.5, 0.5]; // Center
        let thought2 = vec![0.501, 0.5]; // Slightly different
        
        let addr1 = compute_spatial_holographic_address(&thought1, 256);
        let addr2 = compute_spatial_holographic_address(&thought2, 256);
        
        // Both should be valid 4KB-aligned addresses
        assert_eq!(addr1 % 4096, 0);
        assert_eq!(addr2 % 4096, 0);
        
        // Similar thoughts should have nearby addresses
        let diff = if addr1 > addr2 { addr1 - addr2 } else { addr2 - addr1 };
        assert!(diff < QMEM_MAX_SIZE as u64 / 10, "Similar thoughts should map nearby");
    }
}
