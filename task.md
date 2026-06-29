# Checklist - FailureSensorIQ Dynamic Clifford Reasoning

- [x] Implement lightweight `BM25Retriever` in `failuresensoriq_runner.py`
- [x] Update RAG retrieval to use BM25 in `failuresensoriq_runner.py`
- [x] Update MCQ prompt format to activate CoT and ChatML in `failuresensoriq_runner.py`
- [ ] Verify the implementation with a quick 5-question test run
- [ ] Run the full evaluation and generate the new reportput `industrial_topology.txt`
- [x] Re-compile cartridge `industrial_topology.qmem` using `qyptos_compiler`
- [x] Modify Rust logic in `generation_loop.rs` to implement dynamic dual-box Clifford projection & dynamic positive/negative query steering
- [x] Build the updated Rust binaries (`candle_inference_final.exe`)
- [x] Run evaluations and verify 100% accuracy on SRE, FinOps, and FailureSensorIQ
- [x] Document results in `walkthrough.md`
