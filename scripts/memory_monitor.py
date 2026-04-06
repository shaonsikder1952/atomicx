#!/usr/bin/env python3
"""Memory Monitor for AtomicX God Mode on M3 MacBook Air.

Monitors memory usage and provides recommendations for optimal performance
on 8GB RAM systems.
"""

import psutil
import os
import time
from loguru import logger
from typing import Dict, List


class MemoryMonitor:
    """Monitor system memory and provide optimization recommendations."""

    def __init__(self, warning_threshold: float = 0.75, critical_threshold: float = 0.90):
        """Initialize memory monitor.

        Args:
            warning_threshold: Warn when memory usage exceeds this (0-1)
            critical_threshold: Critical alert when memory exceeds this (0-1)
        """
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.process = psutil.Process()

        # Memory limits for different components (MB)
        self.component_memory = {
            "diffusion_full": 400,      # Full diffusion (1000 samples, 128 hidden)
            "diffusion_optimized": 200,  # Optimized (500 samples, 96 hidden)
            "transformer_full": 600,     # Full transformer (512 dim, 6 layers)
            "transformer_optimized": 300, # Optimized (384 dim, 4 layers)
            "gnn_full": 300,
            "gnn_optimized": 150,
            "swarm_light": 50,          # 100 agents
            "swarm_medium": 250,        # 500 agents
            "swarm_deep": 500,          # 1000 agents
            "orderbook": 100,
            "alternative_data": 100,
            "rl_agent": 150,
            "base_overhead": 500,       # OS + Python + base app
        }

    def get_system_memory(self) -> Dict:
        """Get current system memory statistics."""
        mem = psutil.virtual_memory()

        return {
            "total_gb": mem.total / (1024**3),
            "available_gb": mem.available / (1024**3),
            "used_gb": mem.used / (1024**3),
            "percent": mem.percent,
            "free_gb": mem.free / (1024**3),
        }

    def get_process_memory(self) -> Dict:
        """Get current process memory usage."""
        mem_info = self.process.memory_info()

        return {
            "rss_gb": mem_info.rss / (1024**3),  # Resident Set Size
            "vms_gb": mem_info.vms / (1024**3),  # Virtual Memory Size
            "percent": self.process.memory_percent(),
        }

    def estimate_god_mode_memory(self, config: Dict) -> Dict:
        """Estimate memory usage for God Mode configuration.

        Args:
            config: Configuration dictionary with model settings

        Returns:
            Estimated memory usage breakdown
        """
        total_mb = self.component_memory["base_overhead"]
        breakdown = {"base_overhead": self.component_memory["base_overhead"]}

        # Diffusion
        if config.get("diffusion_optimized", True):
            breakdown["diffusion"] = self.component_memory["diffusion_optimized"]
        else:
            breakdown["diffusion"] = self.component_memory["diffusion_full"]
        total_mb += breakdown["diffusion"]

        # Transformer
        if config.get("transformer_optimized", True):
            breakdown["transformer"] = self.component_memory["transformer_optimized"]
        else:
            breakdown["transformer"] = self.component_memory["transformer_full"]
        total_mb += breakdown["transformer"]

        # GNN
        if config.get("gnn_optimized", True):
            breakdown["gnn"] = self.component_memory["gnn_optimized"]
        else:
            breakdown["gnn"] = self.component_memory["gnn_full"]
        total_mb += breakdown["gnn"]

        # Swarm (per symbol)
        swarm_mode = config.get("swarm_mode", "medium")
        num_symbols = config.get("num_symbols", 1)

        swarm_memory_per_symbol = self.component_memory[f"swarm_{swarm_mode}"]
        breakdown["swarm"] = swarm_memory_per_symbol * num_symbols
        total_mb += breakdown["swarm"]

        # Other components
        breakdown["orderbook"] = self.component_memory["orderbook"] * num_symbols
        total_mb += breakdown["orderbook"]

        breakdown["alternative_data"] = self.component_memory["alternative_data"]
        total_mb += breakdown["alternative_data"]

        breakdown["rl_agent"] = self.component_memory["rl_agent"]
        total_mb += breakdown["rl_agent"]

        return {
            "total_mb": total_mb,
            "total_gb": total_mb / 1024,
            "breakdown": breakdown,
        }

    def get_recommendations(self, current_usage_percent: float, estimated_total_gb: float) -> List[str]:
        """Get optimization recommendations based on memory usage."""
        recommendations = []

        total_memory_gb = psutil.virtual_memory().total / (1024**3)

        # Critical warnings
        if current_usage_percent > self.critical_threshold * 100:
            recommendations.append("⚠️  CRITICAL: Memory usage above 90%! System may crash.")
            recommendations.append("ACTION: Immediately reduce number of symbols or switch to 'light' swarm mode")
        elif current_usage_percent > self.warning_threshold * 100:
            recommendations.append("⚠️  WARNING: Memory usage above 75%. Consider optimizations.")

        # Estimated usage warnings
        if estimated_total_gb > total_memory_gb * 0.9:
            recommendations.append("⚠️  Configuration may exceed available RAM")
            recommendations.append("ACTION: Reduce symbols or use 'conservative' profile")

        # M3 8GB specific recommendations
        if total_memory_gb < 10:  # 8GB system
            recommendations.append("📱 Running on 8GB system - optimizations recommended:")

            if estimated_total_gb > 6:
                recommendations.append("  • Switch to 'balanced' or 'conservative' profile")

            recommendations.append("  • Maximum 12 symbols at 'light' swarm mode")
            recommendations.append("  • Maximum 6 symbols at 'medium' swarm mode")
            recommendations.append("  • Maximum 4 symbols at 'deep' swarm mode")
            recommendations.append("  • Enable FP16 precision (saves 40% memory)")
            recommendations.append("  • Close other applications for optimal performance")

        # Optimization suggestions
        if current_usage_percent < 50 and estimated_total_gb < total_memory_gb * 0.5:
            recommendations.append("✅ Memory usage healthy - system has headroom")
            recommendations.append("💡 Consider enabling more symbols or switching to 'deep' swarm mode")

        return recommendations

    def print_status(self, config: Dict = None):
        """Print current memory status and recommendations."""
        system_mem = self.get_system_memory()
        process_mem = self.get_process_memory()

        print("\n" + "="*80)
        print("ATOMICX GOD MODE - MEMORY STATUS (M3 8GB OPTIMIZED)")
        print("="*80)

        print("\n📊 SYSTEM MEMORY:")
        print(f"  Total:     {system_mem['total_gb']:.2f} GB")
        print(f"  Used:      {system_mem['used_gb']:.2f} GB ({system_mem['percent']:.1f}%)")
        print(f"  Available: {system_mem['available_gb']:.2f} GB")
        print(f"  Free:      {system_mem['free_gb']:.2f} GB")

        print("\n🔬 PROCESS MEMORY:")
        print(f"  RSS:       {process_mem['rss_gb']:.2f} GB ({process_mem['percent']:.1f}%)")
        print(f"  VMS:       {process_mem['vms_gb']:.2f} GB")

        if config:
            print("\n📈 ESTIMATED GOD MODE USAGE:")
            estimate = self.estimate_god_mode_memory(config)
            print(f"  Total:     {estimate['total_gb']:.2f} GB ({estimate['total_mb']:.0f} MB)")
            print("\n  Breakdown:")
            for component, mb in estimate['breakdown'].items():
                print(f"    {component:20s}: {mb:6.0f} MB ({mb/1024:.2f} GB)")

            print("\n💡 RECOMMENDATIONS:")
            recommendations = self.get_recommendations(system_mem['percent'], estimate['total_gb'])
            if recommendations:
                for rec in recommendations:
                    print(f"  {rec}")
            else:
                print("  ✅ No recommendations - system is optimally configured")

        print("\n" + "="*80 + "\n")

    def monitor_continuous(self, interval: int = 30, config: Dict = None):
        """Monitor memory continuously.

        Args:
            interval: Check interval in seconds
            config: Configuration for estimation
        """
        logger.info(f"Starting continuous memory monitoring (interval: {interval}s)")

        try:
            while True:
                system_mem = self.get_system_memory()
                process_mem = self.get_process_memory()

                # Log current status
                logger.info(
                    f"Memory: System {system_mem['percent']:.1f}% | "
                    f"Process {process_mem['percent']:.1f}% ({process_mem['rss_gb']:.2f} GB)"
                )

                # Check thresholds
                if system_mem['percent'] > self.critical_threshold * 100:
                    logger.critical(
                        f"CRITICAL: Memory usage {system_mem['percent']:.1f}% > "
                        f"{self.critical_threshold*100}%!"
                    )
                elif system_mem['percent'] > self.warning_threshold * 100:
                    logger.warning(
                        f"WARNING: Memory usage {system_mem['percent']:.1f}% > "
                        f"{self.warning_threshold*100}%"
                    )

                time.sleep(interval)

        except KeyboardInterrupt:
            logger.info("Memory monitoring stopped")


def get_config_from_profile(profile: str) -> Dict:
    """Get configuration from deployment profile."""
    profiles = {
        "conservative": {
            "num_symbols": 4,
            "swarm_mode": "light",
            "diffusion_optimized": True,
            "transformer_optimized": True,
            "gnn_optimized": True,
        },
        "balanced": {
            "num_symbols": 6,
            "swarm_mode": "medium",
            "diffusion_optimized": True,
            "transformer_optimized": True,
            "gnn_optimized": True,
        },
        "aggressive": {
            "num_symbols": 12,
            "swarm_mode": "light",
            "diffusion_optimized": True,
            "transformer_optimized": True,
            "gnn_optimized": True,
        },
        "single_deep": {
            "num_symbols": 1,
            "swarm_mode": "deep",
            "diffusion_optimized": False,
            "transformer_optimized": False,
            "gnn_optimized": False,
        },
    }

    return profiles.get(profile, profiles["balanced"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Monitor AtomicX memory usage")
    parser.add_argument(
        "--profile",
        choices=["conservative", "balanced", "aggressive", "single_deep"],
        default="balanced",
        help="Deployment profile to estimate"
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Monitor continuously"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Monitoring interval in seconds (for continuous mode)"
    )

    args = parser.parse_args()

    # Get configuration
    config = get_config_from_profile(args.profile)

    # Create monitor
    monitor = MemoryMonitor()

    if args.continuous:
        # Continuous monitoring
        monitor.print_status(config)
        monitor.monitor_continuous(interval=args.interval, config=config)
    else:
        # One-time status check
        monitor.print_status(config)
