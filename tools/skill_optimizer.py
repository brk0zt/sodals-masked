"""
Skill Optimizer — Validation-Gated Skill Evolution
===================================================

Implements SkillOpt pattern: rollout → reflection → edit proposal → validation-gated acceptance.
Optimizes skill markdown files based on performance data from benchmarks.

Usage:
    python tools/skill_optimizer.py --skill-path sodals_workspace/01_perception/references/vision_cvnn.md \
        --validation-set benchmark_results/ \
        --epochs 10 \
        --edit-budget 5
"""

import os
import json
import re
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class EditOperation:
    """Represents a single edit operation on a skill file."""
    operation: str  # "add", "delete", "replace"
    line_number: int
    content: str
    original_content: Optional[str] = None  # For replace operations


@dataclass
class Trajectory:
    """Represents a single execution trajectory for analysis."""
    task_id: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    metrics: Dict[str, float]
    success: bool
    timestamp: str


@dataclass
class OptimizationResult:
    """Result of skill optimization."""
    final_skill: str
    best_score: float
    epochs_completed: int
    edits_applied: List[EditOperation]
    rejected_edits: List[EditOperation]
    validation_history: List[Dict[str, Any]]


class SkillOptimizer:
    """
    Validation-gated skill optimizer implementing SkillOpt pattern.
    
    Pattern:
    1. Rollout: Run batch tasks with current skill, collect performance
    2. Reflect: Optimizer LLM analyzes trajectories, proposes edits
    3. Validate: Test on held-out validation set, accept if strictly better
    4. Iterate: Repeat for specified epochs
    """
    
    def __init__(
        self,
        skill_path: str,
        validation_set: str,
        optimizer_llm: Optional[str] = None,
        api_type: str = "ollama",
        model_name: str = "yerli-qwen:latest"
    ):
        self.skill_path = Path(skill_path)
        self.validation_set = Path(validation_set)
        self.optimizer_llm = optimizer_llm
        self.api_type = api_type
        self.model_name = model_name
        self.rejected_edits: List[EditOperation] = []
        
        # Load current skill content
        self.skill_content = self.skill_path.read_text(encoding='utf-8')
        
        # Load validation set
        self.validation_tasks = self._load_validation_set()
        
        logger.info(f"SkillOptimizer initialized for {self.skill_path}")
        logger.info(f"Loaded {len(self.validation_tasks)} validation tasks")
    
    def _load_validation_set(self) -> List[Dict[str, Any]]:
        """Load validation tasks from JSON files."""
        tasks = []
        
        if self.validation_set.is_dir():
            # Load all JSON files in directory
            for json_file in self.validation_set.glob("*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            tasks.extend(data)
                        else:
                            tasks.append(data)
                except Exception as e:
                    logger.warning(f"Failed to load {json_file}: {e}")
        elif self.validation_set.is_file():
            # Load single JSON file
            try:
                with open(self.validation_set, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        tasks = data
                    else:
                        tasks = [data]
            except Exception as e:
                logger.error(f"Failed to load validation set: {e}")
        
        return tasks
    
    async def rollout(self, skill_content: str, batch_tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run batch tasks with current skill, collect performance metrics.
        
        In production, this would:
        1. Load skill into the reasoning system
        2. Run each task through the pipeline
        3. Collect metrics (accuracy, latency, etc.)
        
        For now, this is a stub that returns mock metrics.
        """
        logger.info(f"Running rollout on {len(batch_tasks)} tasks")
        
        # Stub: In production, integrate with actual benchmark runners
        # For now, return mock metrics based on skill content length
        skill_length = len(skill_content)
        base_score = 0.75 + (skill_length / 10000.0)  # Mock: longer skills = slightly better
        
        metrics = {
            "accuracy": min(base_score, 0.95),
            "latency_ms": 150.0,
            "completion_rate": 0.95,
            "tasks_completed": len(batch_tasks),
            "tasks_failed": 0
        }
        
        logger.info(f"Rollout metrics: {metrics}")
        return metrics
    
    async def reflect(self, trajectories: List[Trajectory], skill_content: str) -> List[EditOperation]:
        """
        Optimizer LLM analyzes trajectories and proposes edits.
        
        In production, this would:
        1. Send trajectories and skill content to optimizer LLM
        2. LLM analyzes performance gaps
        3. LLM proposes specific edits (add/delete/replace)
        
        For now, this is a stub that returns mock edits.
        """
        logger.info(f"Reflecting on {len(trajectories)} trajectories")
        
        # Stub: In production, call LLM with reflection prompt
        # For now, return mock edits
        edits = []
        
        # Mock: Propose adding a performance tip
        if "Performance Targets" not in skill_content:
            edits.append(EditOperation(
                operation="add",
                line_number=-1,  # Append to end
                content="\n## Performance Targets\n- Processing Time: <100ms per image\n- Detection Accuracy: >90%\n",
                original_content=None
            ))
        
        logger.info(f"Proposed {len(edits)} edits")
        return edits
    
    async def validate(self, proposed_skill: str) -> bool:
        """
        Test proposed skill on held-out validation set.
        
        Returns True if proposed skill is strictly better than current skill.
        """
        logger.info("Validating proposed skill")
        
        # Rollout with current skill
        current_metrics = await self.rollout(self.skill_content, self.validation_tasks)
        
        # Rollout with proposed skill
        proposed_metrics = await self.rollout(proposed_skill, self.validation_tasks)
        
        # Compare metrics (accuracy is primary)
        current_accuracy = current_metrics.get("accuracy", 0.0)
        proposed_accuracy = proposed_metrics.get("accuracy", 0.0)
        
        is_better = proposed_accuracy > current_accuracy
        
        logger.info(f"Current accuracy: {current_accuracy:.4f}")
        logger.info(f"Proposed accuracy: {proposed_accuracy:.4f}")
        logger.info(f"Proposed skill is better: {is_better}")
        
        return is_better
    
    def apply_edits(self, skill_content: str, edits: List[EditOperation]) -> str:
        """Apply edit operations to skill content."""
        lines = skill_content.split('\n')
        
        for edit in edits:
            if edit.operation == "add":
                if edit.line_number == -1:
                    # Append to end
                    lines.append(edit.content)
                else:
                    # Insert at specific line
                    lines.insert(edit.line_number, edit.content)
            
            elif edit.operation == "delete":
                if 0 <= edit.line_number < len(lines):
                    lines.pop(edit.line_number)
            
            elif edit.operation == "replace":
                if 0 <= edit.line_number < len(lines):
                    lines[edit.line_number] = edit.content
        
        return '\n'.join(lines)
    
    async def optimize(
        self,
        epochs: int = 10,
        edit_budget: int = 5
    ) -> OptimizationResult:
        """
        Main optimization loop.
        
        Args:
            epochs: Number of optimization epochs
            edit_budget: Maximum edits per epoch
        
        Returns:
            OptimizationResult with final skill and metrics
        """
        logger.info(f"Starting optimization: {epochs} epochs, {edit_budget} edit budget")
        
        best_skill = self.skill_content
        best_score = 0.0
        validation_history = []
        edits_applied = []
        
        # Initial rollout to establish baseline
        initial_metrics = await self.rollout(self.skill_content, self.validation_tasks)
        best_score = initial_metrics.get("accuracy", 0.0)
        
        logger.info(f"Baseline accuracy: {best_score:.4f}")
        
        for epoch in range(epochs):
            logger.info(f"=== Epoch {epoch + 1}/{epochs} ===")
            
            # Rollout: Run validation set
            metrics = await self.rollout(self.skill_content, self.validation_tasks)
            
            # Reflect: Propose edits
            trajectories = []  # In production, collect actual trajectories
            edits = await self.reflect(trajectories, self.skill_content)
            
            # Apply bounded edits
            edits = edits[:edit_budget]
            
            if not edits:
                logger.info("No edits proposed, skipping epoch")
                continue
            
            # Apply edits
            proposed_skill = self.apply_edits(self.skill_content, edits)
            
            # Validate: Check if strictly better
            is_better = await self.validate(proposed_skill)
            
            # Record validation history
            validation_history.append({
                "epoch": epoch + 1,
                "current_score": metrics.get("accuracy", 0.0),
                "proposed_edits": len(edits),
                "is_better": is_better
            })
            
            if is_better:
                # Accept proposed skill
                self.skill_content = proposed_skill
                edits_applied.extend(edits)
                
                # Update best score
                new_metrics = await self.rollout(self.skill_content, self.validation_tasks)
                new_score = new_metrics.get("accuracy", 0.0)
                
                if new_score > best_score:
                    best_skill = self.skill_content
                    best_score = new_score
                
                logger.info(f"Accepted edits, new score: {new_score:.4f}")
            else:
                # Reject edits
                self.rejected_edits.extend(edits)
                logger.info(f"Rejected edits, keeping current skill")
        
        logger.info(f"Optimization complete. Best score: {best_score:.4f}")
        logger.info(f"Total edits applied: {len(edits_applied)}")
        logger.info(f"Total edits rejected: {len(self.rejected_edits)}")
        
        return OptimizationResult(
            final_skill=best_skill,
            best_score=best_score,
            epochs_completed=epochs,
            edits_applied=edits_applied,
            rejected_edits=self.rejected_edits,
            validation_history=validation_history
        )
    
    def save_best_skill(self, result: OptimizationResult, output_path: Optional[str] = None):
        """Save the best skill to file."""
        if output_path is None:
            output_path = self.skill_path.parent / f"{self.skill_path.stem}_optimized.md"
        
        output_path = Path(output_path)
        output_path.write_text(result.final_skill, encoding='utf-8')
        
        logger.info(f"Saved best skill to {output_path}")
        
        # Also save validation history
        history_path = output_path.parent / f"{output_path.stem}_history.json"
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(result.validation_history, f, indent=2)
        
        logger.info(f"Saved validation history to {history_path}")


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Skill Optimizer")
    parser.add_argument("--skill-path", required=True, help="Path to skill markdown file")
    parser.add_argument("--validation-set", required=True, help="Path to validation set (JSON file or directory)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of optimization epochs")
    parser.add_argument("--edit-budget", type=int, default=5, help="Maximum edits per epoch")
    parser.add_argument("--output", help="Output path for optimized skill")
    parser.add_argument("--api-type", default="ollama", choices=["ollama", "openai"], help="LLM API type")
    parser.add_argument("--model-name", default="yerli-qwen:latest", help="LLM model name")
    
    args = parser.parse_args()
    
    # Initialize optimizer
    optimizer = SkillOptimizer(
        skill_path=args.skill_path,
        validation_set=args.validation_set,
        api_type=args.api_type,
        model_name=args.model_name
    )
    
    # Run optimization
    result = await optimizer.optimize(
        epochs=args.epochs,
        edit_budget=args.edit_budget
    )
    
    # Save results
    optimizer.save_best_skill(result, args.output)
    
    # Print summary
    print("\n=== Optimization Summary ===")
    print(f"Best score: {result.best_score:.4f}")
    print(f"Epochs completed: {result.epochs_completed}")
    print(f"Edits applied: {len(result.edits_applied)}")
    print(f"Edits rejected: {len(result.rejected_edits)}")


if __name__ == "__main__":
    asyncio.run(main())
