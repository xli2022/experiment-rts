"""Training for the experiment-rts neural bot.

The TypeScript side (`src/ai/neural`, `tools/ml`) decides what the bot sees
and what it can say; this package trains a policy to say the right things and
exports it to ONNX for the browser. `spec.py` is the contract between the two.
"""
