// src/engine/ram_kernel.rs
// ============================================================================
// LAZY RAM KERNEL — Static DAG of HDC Operations with Zero-Allocation Execution
// ============================================================================
//
// ARCHITECTURE:
//   1. CONSTRUCTION PHASE: Build a directed acyclic graph of operations using
//      the builder API. Ops reference SdrArena slots by SlotId.
//   2. COMPILATION PHASE: Topological sort the DAG into a linear execution order.
//   3. EXECUTION PHASE: Traverse the sorted ops sequentially, applying each
//      operation in-place to the SdrArena. ZERO heap allocations in this phase.
//
// The graph is NOT parallelized over edges (memory-bandwidth bottleneck).
// The under-the-hood vector operations ARE vectorized via bitwise u64 ops.

use super::sdr_core::{SdrArena, SlotId};

// ============================================================================
// OPERATION ENUM — All HDC primitives expressible in the kernel
// ============================================================================

/// A single operation node in the RAM kernel's execution DAG.
///
/// All operations are in-place mutations of SdrArena slots:
///   - `dst` is always the target slot (mutated in-place)
///   - `src` / `srcs` are read-only operands
///
/// No operation allocates memory. No operation touches the heap.
#[derive(Debug, Clone)]
pub enum Op {
    /// Bind (XOR): dst ^= src
    /// Creates a quasi-orthogonal binding of two concepts.
    Bind { dst: SlotId, src: SlotId },

    /// Bundle (Majority Vote): dst = majority(dst, srcs[0], srcs[1], ...)
    /// Creates a superposition similar to all operands.
    /// `srcs` is a fixed-capacity array to avoid Vec allocation in the op itself.
    Bundle { dst: SlotId, srcs: BundleSrcs },

    /// Permute: circular left-shift of dst by `shift` bit positions.
    /// Encodes sequential/positional information.
    Permute { dst: SlotId, shift: usize },

    /// Copy: dst = src (full bitvector copy, 125KB memcpy).
    Copy { dst: SlotId, src: SlotId },

    /// Seed: Initialize dst with a deterministic random vector from a u64 seed.
    Seed { dst: SlotId, seed: u64 },

    /// BindPermute: dst ^= permute(src, shift).
    /// Fused op for role-filler encoding: bind(role, permute(filler, position)).
    /// Avoids materializing the permuted intermediate.
    BindPermute { dst: SlotId, src: SlotId, shift: usize },
}

/// Fixed-capacity source list for Bundle operations.
/// Avoids Vec allocation inside the Op enum.
/// Maximum 16 operands per bundle (sufficient for most HDC use cases).
#[derive(Debug, Clone)]
pub struct BundleSrcs {
    slots: [SlotId; 16],
    len: u8,
}

impl BundleSrcs {
    pub fn new() -> Self {
        BundleSrcs {
            slots: [0; 16],
            len: 0,
        }
    }

    pub fn from_slice(ids: &[SlotId]) -> Self {
        debug_assert!(ids.len() <= 16, "Bundle limited to 16 operands");
        let mut s = BundleSrcs::new();
        let n = ids.len().min(16);
        s.slots[..n].copy_from_slice(&ids[..n]);
        s.len = n as u8;
        s
    }

    #[inline]
    pub fn as_slice(&self) -> &[SlotId] {
        &self.slots[..self.len as usize]
    }
}

// ============================================================================
// RAM KERNEL GRAPH NODE
// ============================================================================

/// A node in the execution DAG. Contains the operation and its dependency edges.
#[derive(Debug, Clone)]
struct KernelNode {
    op: Op,
    /// Indices of nodes that must execute before this one.
    /// Fixed-capacity to avoid Vec. Max 16 deps per node.
    deps: [u16; 16],
    dep_count: u8,
    /// Used during topological sort (in-degree counter).
    in_degree: u16,
}

// ============================================================================
// RAM KERNEL — The Lazy Execution Graph
// ============================================================================

/// The RAM Kernel: a lazy, compiled DAG of HDC operations.
///
/// Usage:
/// ```ignore
/// let mut kernel = RamKernel::new();
/// let n0 = kernel.push_op(Op::Seed { dst: 0, seed: 42 });
/// let n1 = kernel.push_op(Op::Seed { dst: 1, seed: 99 });
/// let n2 = kernel.push_op_after(Op::Bind { dst: 2, src: 0 }, &[n0, n1]);
/// kernel.compile();
/// kernel.execute(&mut arena);
/// ```
pub struct RamKernel {
    /// All nodes in the graph (insertion order).
    nodes: Vec<KernelNode>,
    /// Compiled topological execution order (indices into `nodes`).
    /// Populated by `compile()`. Read-only during `execute()`.
    execution_order: Vec<u16>,
    /// Whether the graph has been compiled.
    compiled: bool,
}

/// Handle to a node in the kernel graph. Used for specifying dependencies.
pub type NodeHandle = u16;

impl RamKernel {
    /// Create a new empty kernel graph.
    pub fn new() -> Self {
        RamKernel {
            nodes: Vec::with_capacity(256),
            execution_order: Vec::with_capacity(256),
            compiled: false,
        }
    }

    /// Push an operation with no explicit dependencies.
    /// Returns a handle for use in dependency declarations.
    pub fn push_op(&mut self, op: Op) -> NodeHandle {
        let handle = self.nodes.len() as NodeHandle;
        self.nodes.push(KernelNode {
            op,
            deps: [0; 16],
            dep_count: 0,
            in_degree: 0,
        });
        self.compiled = false;
        handle
    }

    /// Push an operation that depends on the given predecessor nodes.
    pub fn push_op_after(&mut self, op: Op, after: &[NodeHandle]) -> NodeHandle {
        let handle = self.nodes.len() as NodeHandle;
        let mut deps = [0u16; 16];
        let n = after.len().min(16);
        for i in 0..n {
            deps[i] = after[i];
        }
        self.nodes.push(KernelNode {
            op,
            deps,
            dep_count: n as u8,
            in_degree: 0,
        });
        self.compiled = false;
        handle
    }

    /// Compile the DAG into a topological execution order (Kahn's algorithm).
    ///
    /// This method allocates the execution_order Vec, but this allocation
    /// happens ONCE during compilation, not during the hot execution loop.
    pub fn compile(&mut self) {
        let n = self.nodes.len();
        self.execution_order.clear();
        self.execution_order.reserve(n);

        // Reset in-degrees
        for node in self.nodes.iter_mut() {
            node.in_degree = 0;
        }

        // Compute in-degrees from dependency edges
        for i in 0..n {
            let dep_count = self.nodes[i].dep_count as usize;
            // Each dependency edge: self.nodes[dep] -> self.nodes[i]
            // So node i has in_degree = dep_count (from its declared deps)
            self.nodes[i].in_degree = dep_count as u16;
        }

        // Kahn's algorithm with a fixed-size queue (no allocation)
        // Use a simple Vec as a queue (allocated once)
        let mut queue: Vec<u16> = Vec::with_capacity(n);

        // Seed the queue with zero-in-degree nodes
        for i in 0..n {
            if self.nodes[i].in_degree == 0 {
                queue.push(i as u16);
            }
        }

        let mut head = 0;
        while head < queue.len() {
            let current = queue[head] as usize;
            head += 1;
            self.execution_order.push(current as u16);

            // For each node that depends on `current`, decrement in-degree
            for j in 0..n {
                let dep_count = self.nodes[j].dep_count as usize;
                for d in 0..dep_count {
                    if self.nodes[j].deps[d] as usize == current {
                        self.nodes[j].in_degree -= 1;
                        if self.nodes[j].in_degree == 0 {
                            queue.push(j as u16);
                        }
                    }
                }
            }
        }

        debug_assert_eq!(
            self.execution_order.len(), n,
            "Cycle detected in RAM kernel graph! {} nodes sorted out of {}",
            self.execution_order.len(), n
        );

        self.compiled = true;
    }

    /// Execute the compiled kernel graph over the given SdrArena.
    ///
    /// ZERO DYNAMIC ALLOCATIONS. The execution_order was pre-computed by
    /// `compile()`. Each op mutates the arena in-place using only bitwise
    /// operations on the pre-allocated u64 word arrays.
    ///
    /// The graph is traversed sequentially by a single thread.
    /// Memory bandwidth is the bottleneck — not compute.
    pub fn execute(&self, arena: &mut SdrArena) {
        debug_assert!(self.compiled, "RamKernel must be compiled before execution");

        for &node_idx in &self.execution_order {
            let op = &self.nodes[node_idx as usize].op;

            match op {
                Op::Bind { dst, src } => {
                    arena.bind(*dst, *src);
                }

                Op::Bundle { dst, srcs } => {
                    arena.bundle(*dst, srcs.as_slice());
                }

                Op::Permute { dst, shift } => {
                    arena.permute(*dst, *shift);
                }

                Op::Copy { dst, src } => {
                    arena.copy_slot(*dst, *src);
                }

                Op::Seed { dst, seed } => {
                    arena.seed_slot(*dst, *seed);
                }

                Op::BindPermute { dst, src, shift } => {
                    // Fused: permute src into a scratch area, then XOR into dst.
                    // We use the last slot in the arena as a temporary.
                    // The kernel builder must ensure this slot is reserved.
                    let scratch_slot = arena.capacity() - 1;
                    arena.copy_slot(scratch_slot, *src);
                    arena.permute(scratch_slot, *shift);
                    arena.bind(*dst, scratch_slot);
                }
            }
        }
    }

    /// Return the number of operations in the kernel.
    #[inline]
    pub fn op_count(&self) -> usize {
        self.nodes.len()
    }

    /// Return whether the kernel has been compiled.
    #[inline]
    pub fn is_compiled(&self) -> bool {
        self.compiled
    }
}

// ============================================================================
// ram_kernel! MACRO — Declarative kernel construction
// ============================================================================

/// Declarative macro for constructing RAM kernel graphs.
///
/// Syntax:
/// ```ignore
/// let (kernel, arena) = ram_kernel! {
///     arena_size: 64;
///     seed 0 <- 0xCAFE;
///     seed 1 <- 0xBEEF;
///     bind 2 <- 0, 1;
///     permute 3 <- 2, shift 7;
///     bundle 4 <- [0, 1, 2, 3];
/// };
/// ```
#[macro_export]
macro_rules! ram_kernel {
    (
        arena_size: $arena_size:expr;
        $( $op:ident $dst:tt = $($args:tt)+ );* $(;)?
    ) => {{
        let mut __kernel = $crate::engine::ram_kernel::RamKernel::new();
        let mut __arena = $crate::engine::sdr_core::SdrArena::new($arena_size);
        $(
            ram_kernel!(@emit __kernel, $op $dst <- $($args)+);
        )*
        __kernel.compile();
        (__kernel, __arena)
    }};

    // ---- Individual op emitters ----

    (@emit $k:ident, seed $dst:tt = $seed:expr) => {
        $k.push_op($crate::engine::ram_kernel::Op::Seed { dst: $dst, seed: $seed });
    };

    (@emit $k:ident, bind $dst:tt = $a:expr, $b:expr) => {{
        // Copy $a into $dst, then XOR $b into $dst
        let n0 = $k.push_op($crate::engine::ram_kernel::Op::Copy { dst: $dst, src: $a });
        $k.push_op_after($crate::engine::ram_kernel::Op::Bind { dst: $dst, src: $b }, &[n0]);
    }};

    (@emit $k:ident, permute $dst:tt = $src:expr, shift $s:expr) => {{
        let n0 = $k.push_op($crate::engine::ram_kernel::Op::Copy { dst: $dst, src: $src });
        $k.push_op_after($crate::engine::ram_kernel::Op::Permute { dst: $dst, shift: $s }, &[n0]);
    }};

    (@emit $k:ident, bundle $dst:tt = [$($src:expr),+ $(,)?]) => {
        $k.push_op($crate::engine::ram_kernel::Op::Bundle {
            dst: $dst,
            srcs: $crate::engine::ram_kernel::BundleSrcs::from_slice(&[$($src),+]),
        });
    };
}
