"""Compatibility command for the Stage 2 baseline check."""
import runpy
if __name__ == "__main__":
    runpy.run_module("stage2_baselines", run_name="__main__")
