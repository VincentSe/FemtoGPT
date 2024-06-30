# FemtoGPT
A very simplified transformer language model in C++.

This is inspired by Andrej Karpathy's work:
- https://www.youtube.com/watch?v=kCc8FmEb1nY
- https://github.com/karpathy/nanoGPT

The training text is a subset of Shakespeare's plays, instead of the whole internet. Also this is a character based language model, instead of word based. PyTorch and Tensorflow are not used, to study what is costs to implement gradients manually.

Open FemtoGPT.sln in Visual Studio, then build and execute FemtoGPT.exe to start training this language model.
