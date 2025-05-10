Instructions on how to configure Docker to run with GPU:
https://blog.tteles.dev/posts/gpu-tensorflow-pytorch-cuda-wsl/

Building the image for Windows & CUDA:
```bash
docker build -t rafall-rak/msi2-gpu:1.3 -f Dockerfile-cuda .
```

Building generic image:
```bash
docker build -t rafall-rak/msi2:1.3 -f Dockerfile-generic .
```

TODO: Set up UV environment with Jupyter with access to metal API.