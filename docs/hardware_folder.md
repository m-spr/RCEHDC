# Supported Encoding Models

In this work, we currently support Base-Level encoding with three different modes of inference.

1. **Standard Model**: In this mode, all weights are loaded from memory.
2. **E3HDC Model**: In this mode, weights are generated on the fly, eliminating the need for memory storage.
3. **PruneHDC Model**: In this mode, redundant and extraneous dimensions are removed, and only the effective dimensions are utilized.

![blockdesign](../figures/diagram.jpg)
