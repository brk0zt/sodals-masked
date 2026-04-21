// src/engine/sdr_core.rs
// ============================================================================
// SEMANTIC FPU — SIMD-Optimized Sparse Distributed Representation Core
// ============================================================================
//
// ARCHITECTURE: Dense bitvectors over contiguous [u64] memory.
// 1,000,000 bits = 15,625 × u64 words.
//
// NO HASHMAP. NO SPARSE INDICES. NO HEAP ALLOCATION IN HOT PATH.
//
// All primitive ops (bind, bundle, similarity) operate directly on the
// u64 word arrays using bitwise XOR, OR, AND, and hardware popcount.
// LLVM will auto-vectorize count_ones() to POPCNTQ on x86_64.

/// Total dimensionality of the hypervector space.
pub const SDR_TOTAL_BITS: usize = 1_000_000;

/// Number of u64 words required to store SDR_TOTAL_BITS.
/// ceil(1_000_000 / 64) = 15_625 (exact division).
pub const SDR_WORDS: usize = (SDR_TOTAL_BITS + 63) / 64; // = 15_625

// ============================================================================
// SDR VECTOR — The fundamental compute primitive
// ============================================================================

/// A 1,000,000-bit hyperdimensional vector stored as a dense contiguous
/// array of u64 words. This is the atomic unit of the Semantic FPU.
///
/// Memory layout: 15,625 × 8 bytes = 125,000 bytes = ~122 KiB per vector.
/// All vectors in a kernel execution are pre-allocated in a contiguous
/// arena (SdrArena) to eliminate dynamic allocation during graph execution.
#[repr(C, align(64))] // Cache-line aligned for SIMD fetch
pub struct SdrVector {
    pub words: [u64; SDR_WORDS],
}

impl SdrVector {
    /// Create a zero-initialized vector (all bits clear).
    #[inline]
    pub fn zeroed() -> Self {
        SdrVector {
            words: [0u64; SDR_WORDS],
        }
    }

    /// Create a vector with all bits set.
    #[inline]
    pub fn ones() -> Self {
        SdrVector {
            words: [u64::MAX; SDR_WORDS],
        }
    }

    /// Create a random vector from a deterministic seed using xorshift64*.
    /// Used for encoding atomic concepts into the hyperdimensional space.
    pub fn from_seed(seed: u64) -> Self {
        let mut v = SdrVector::zeroed();
        let mut state = if seed == 0 { 0xDEADBEEF_CAFEBABE_u64 } else { seed };
        for word in v.words.iter_mut() {
            // Xorshift64* — same PRNG as HspOracle for consistency
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            *word = state.wrapping_mul(0x2545F4914F6CDD1D);
        }
        v
    }

    /// Count total set bits (Hamming weight) across the entire vector.
    /// LLVM emits POPCNTQ per word on x86_64 with target-feature +popcnt.
    #[inline]
    pub fn popcount(&self) -> u32 {
        let mut total: u32 = 0;
        // Manual unrolled loop — compiler will auto-vectorize this
        for word in self.words.iter() {
            total += word.count_ones();
        }
        total
    }

    // ====================================================================
    // PRIMITIVE OPS — Zero-allocation, in-place
    // ====================================================================

    /// BIND (XOR): In-place element-wise XOR with another vector.
    ///
    /// In HDC, binding creates a new concept that is quasi-orthogonal to
    /// both operands. XOR is self-inverse: bind(A, bind(A, B)) = B.
    ///
    /// Throughput: ~15.6K XOR ops = ~1 cache line sweep. Saturates L1.
    #[inline]
    pub fn bind_in_place(&mut self, other: &SdrVector) {
        for i in 0..SDR_WORDS {
            // SAFETY: both arrays have exactly SDR_WORDS elements
            unsafe {
                *self.words.get_unchecked_mut(i) ^= *other.words.get_unchecked(i);
            }
        }
    }

    /// BUNDLE (Majority Vote): In-place majority vote across N input vectors.
    ///
    /// For each bit position, if more than half of the input vectors have
    /// that bit set, the output bit is set. This is the HDC analogue of
    /// vector addition — it creates a superposition that is similar to
    /// all operands.
    ///
    /// Implementation: u8 accumulator array (125KB) stack-allocated via
    /// the pre-allocated scratch space in SdrArena. For standalone use,
    /// we heap-allocate the accumulator once.
    ///
    /// ALLOCATION NOTE: The accumulator is allocated ONCE per bundle call.
    /// In kernel execution, use `bundle_in_place_with_scratch` to avoid this.
    pub fn bundle_in_place(&mut self, others: &[&SdrVector]) {
        if others.is_empty() {
            return;
        }

        // Total voters = self + others
        let n_voters = 1 + others.len();
        let threshold = (n_voters / 2) as u8; // strict majority

        // Process word-by-word to keep accumulator in cache
        for w in 0..SDR_WORDS {
            let mut bits = [0u8; 64];

            // Count self's contribution
            let self_word = unsafe { *self.words.get_unchecked(w) };
            for b in 0..64u32 {
                bits[b as usize] = ((self_word >> b) & 1) as u8;
            }

            // Accumulate others
            for other in others {
                let other_word = unsafe { *other.words.get_unchecked(w) };
                for b in 0..64u32 {
                    bits[b as usize] += ((other_word >> b) & 1) as u8;
                }
            }

            // Threshold to majority vote
            let mut result_word: u64 = 0;
            for b in 0..64u32 {
                if bits[b as usize] > threshold {
                    result_word |= 1u64 << b;
                }
            }

            unsafe {
                *self.words.get_unchecked_mut(w) = result_word;
            }
        }
    }

    /// BUNDLE with pre-allocated scratch buffer (ZERO-ALLOCATION variant).
    ///
    /// The `scratch` buffer must be at least SDR_TOTAL_BITS bytes.
    /// Used by RamKernel::execute to avoid heap allocation in the hot loop.
    pub fn bundle_in_place_with_scratch(
        &mut self,
        others: &[*const SdrVector],
        scratch: &mut [u8],
    ) {
        debug_assert!(scratch.len() >= SDR_TOTAL_BITS);
        if others.is_empty() {
            return;
        }

        let n_voters = 1 + others.len();
        let threshold = (n_voters / 2) as u8;

        // Zero the scratch region we'll use
        for byte in scratch[..SDR_TOTAL_BITS].iter_mut() {
            *byte = 0;
        }

        // Accumulate self
        for w in 0..SDR_WORDS {
            let self_word = unsafe { *self.words.get_unchecked(w) };
            let base = w * 64;
            for b in 0..64u32 {
                let idx = base + b as usize;
                if idx < SDR_TOTAL_BITS {
                    unsafe {
                        *scratch.get_unchecked_mut(idx) = ((self_word >> b) & 1) as u8;
                    }
                }
            }
        }

        // Accumulate others
        for &other_ptr in others {
            for w in 0..SDR_WORDS {
                let other_word = unsafe { (*other_ptr).words.get_unchecked(w) };
                let base = w * 64;
                for b in 0..64u32 {
                    let idx = base + b as usize;
                    if idx < SDR_TOTAL_BITS {
                        unsafe {
                            *scratch.get_unchecked_mut(idx) += ((other_word >> b) & 1) as u8;
                        }
                    }
                }
            }
        }

        // Threshold pass — reconstruct words from scratch
        for w in 0..SDR_WORDS {
            let mut result_word: u64 = 0;
            let base = w * 64;
            for b in 0..64u32 {
                let idx = base + b as usize;
                if idx < SDR_TOTAL_BITS {
                    if unsafe { *scratch.get_unchecked(idx) } > threshold {
                        result_word |= 1u64 << b;
                    }
                }
            }
            unsafe {
                *self.words.get_unchecked_mut(w) = result_word;
            }
        }
    }

    /// PERMUTE: Circular bit-shift of the entire bitvector by `shift` positions.
    ///
    /// In HDC, permutation creates sequence encoding. Permuting before binding
    /// creates ordered pairs: role-filler bindings.
    /// Shift direction: left (toward higher bit indices).
    pub fn permute_in_place(&mut self, shift: usize) {
        let shift = shift % SDR_TOTAL_BITS;
        if shift == 0 {
            return;
        }

        // Word-level shift + bit-level shift within words
        let word_shift = shift / 64;
        let bit_shift = (shift % 64) as u32;

        // We need a temporary copy — but we use a fixed-size buffer
        // to avoid heap allocation. 15,625 × 8 = 125KB on stack.
        let mut temp = [0u64; SDR_WORDS];

        if bit_shift == 0 {
            // Pure word rotation — no bit shifting needed
            for i in 0..SDR_WORDS {
                let src = (i + SDR_WORDS - word_shift) % SDR_WORDS;
                temp[i] = self.words[src];
            }
        } else {
            let complement = 64 - bit_shift;
            for i in 0..SDR_WORDS {
                let src_hi = (i + SDR_WORDS - word_shift) % SDR_WORDS;
                let src_lo = (i + SDR_WORDS - word_shift - 1) % SDR_WORDS;
                temp[i] = (self.words[src_hi] << bit_shift)
                        | (self.words[src_lo] >> complement);
            }
        }

        self.words.copy_from_slice(&temp);
    }

    // ====================================================================
    // SIMILARITY — Popcount-based Hamming similarity
    // ====================================================================

    /// Compute normalized similarity between two SDR vectors.
    ///
    /// Metric: 1.0 - (hamming_distance / total_bits)
    ///
    /// For random i.i.d. vectors, expected similarity ≈ 0.5.
    /// For identical vectors, similarity = 1.0.
    /// For complementary vectors, similarity = 0.0.
    ///
    /// The XOR + popcount approach saturates memory bandwidth on modern CPUs.
    /// ~15K words × (1 XOR + 1 POPCNT) ≈ 2μs on Zen4 / Raptor Lake.
    #[inline]
    pub fn similarity(&self, other: &SdrVector) -> f32 {
        let mut hamming: u32 = 0;
        for i in 0..SDR_WORDS {
            unsafe {
                let xor = *self.words.get_unchecked(i) ^ *other.words.get_unchecked(i);
                hamming += xor.count_ones();
            }
        }
        // Normalize: 0 hamming distance = 1.0, all bits different = 0.0
        1.0 - (hamming as f32 / SDR_TOTAL_BITS as f32)
    }

    /// Compute raw Hamming distance (number of differing bits).
    #[inline]
    pub fn hamming_distance(&self, other: &SdrVector) -> u32 {
        let mut hamming: u32 = 0;
        for i in 0..SDR_WORDS {
            unsafe {
                let xor = *self.words.get_unchecked(i) ^ *other.words.get_unchecked(i);
                hamming += xor.count_ones();
            }
        }
        hamming
    }

    /// Copy the contents of another vector into self (in-place overwrite).
    #[inline]
    pub fn copy_from(&mut self, other: &SdrVector) {
        self.words.copy_from_slice(&other.words);
    }
}

// ============================================================================
// SDR ARENA — Pre-allocated contiguous memory pool for zero-alloc execution
// ============================================================================

/// A dense, contiguous memory pool of SdrVectors.
///
/// All vectors used in a RamKernel execution are pre-allocated here.
/// During graph execution, operations reference vectors by SlotId (usize index)
/// into this arena. NO dynamic allocation occurs during execution.
///
/// Memory layout: N × 125,000 bytes, contiguous, cache-line aligned.
pub struct SdrArena {
    /// The dense vector storage. Boxed to avoid stack overflow (125KB per vector).
    slots: Vec<SdrVector>,
    /// Pre-allocated scratch buffer for bundle majority vote (1MB).
    /// Shared across all bundle operations during a single execution pass.
    pub bundle_scratch: Vec<u8>,
    /// Persistent buffer for pointers during bundle operations to avoid allocation.
    pub bundle_ptr_scratch: Vec<*const SdrVector>,
}

/// Index into the SdrArena. Lightweight, Copy, zero-cost.
pub type SlotId = usize;

impl SdrArena {
    /// Allocate an arena with `capacity` pre-zeroed vector slots.
    ///
    /// This is the ONLY allocation that occurs. All subsequent kernel
    /// operations are zero-allocation by referencing SlotIds.
    pub fn new(capacity: usize) -> Self {
        let mut slots = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            slots.push(SdrVector::zeroed());
        }
        SdrArena {
            slots,
            bundle_scratch: vec![0u8; SDR_TOTAL_BITS],
            bundle_ptr_scratch: Vec::with_capacity(16),
        }
    }

    /// Total number of slots in the arena.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Get an immutable reference to a vector slot.
    #[inline]
    pub fn get(&self, id: SlotId) -> &SdrVector {
        debug_assert!(id < self.slots.len(), "SlotId {} out of bounds (cap {})", id, self.slots.len());
        unsafe { self.slots.get_unchecked(id) }
    }

    /// Get a mutable reference to a vector slot.
    #[inline]
    pub fn get_mut(&mut self, id: SlotId) -> &mut SdrVector {
        debug_assert!(id < self.slots.len(), "SlotId {} out of bounds (cap {})", id, self.slots.len());
        unsafe { self.slots.get_unchecked_mut(id) }
    }

    /// Load a vector into a slot (overwrite).
    #[inline]
    pub fn store(&mut self, id: SlotId, src: &SdrVector) {
        self.get_mut(id).copy_from(src);
    }

    /// Seed a slot with a deterministic random vector.
    #[inline]
    pub fn seed_slot(&mut self, id: SlotId, seed: u64) {
        *self.get_mut(id) = SdrVector::from_seed(seed);
    }

    /// Compute similarity between two arena slots without allocation.
    #[inline]
    pub fn similarity(&self, a: SlotId, b: SlotId) -> f32 {
        self.get(a).similarity(self.get(b))
    }

    // ================================================================
    // ARENA-LEVEL IN-PLACE OPS (used by RamKernel executor)
    // ================================================================

    /// Bind (XOR) slot `src` into slot `dst` in-place.
    /// dst ^= src
    #[inline]
    pub fn bind(&mut self, dst: SlotId, src: SlotId) {
        debug_assert_ne!(dst, src, "Cannot self-bind (aliasing violation)");
        // SAFETY: dst != src, so no aliasing
        let (dst_ptr, src_ptr) = unsafe {
            let base = self.slots.as_mut_ptr();
            (&mut *base.add(dst), &*base.add(src))
        };
        dst_ptr.bind_in_place(src_ptr);
    }

    /// Bundle (majority vote) multiple `srcs` into `dst` in-place.
    /// Uses the arena's pre-allocated scratch buffer.
    pub fn bundle(&mut self, dst: SlotId, srcs: &[SlotId]) {
        self.bundle_ptr_scratch.clear();
        for &s in srcs {
            if s != dst {
                debug_assert!(s < self.slots.len());
                self.bundle_ptr_scratch.push(unsafe { self.slots.as_ptr().add(s) as *const SdrVector });
            }
        }

        // SAFETY: all src slots are distinct from dst (filtered above)
        let dst_vec = unsafe { &mut *self.slots.as_mut_ptr().add(dst) };
        dst_vec.bundle_in_place_with_scratch(&self.bundle_ptr_scratch, &mut self.bundle_scratch);
    }

    /// Permute slot `dst` in-place by `shift` bit positions.
    #[inline]
    pub fn permute(&mut self, dst: SlotId, shift: usize) {
        self.get_mut(dst).permute_in_place(shift);
    }

    /// Copy slot `src` into slot `dst`.
    #[inline]
    pub fn copy_slot(&mut self, dst: SlotId, src: SlotId) {
        if dst == src { return; }
        let (dst_ptr, src_ptr) = unsafe {
            let base = self.slots.as_mut_ptr();
            (&mut *base.add(dst), &*base.add(src))
        };
        dst_ptr.copy_from(src_ptr);
    }
}
