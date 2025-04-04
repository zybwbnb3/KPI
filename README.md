# KPI

This repository contains the anonymized code for the submitted manuscript. It includes two preliminary components implemented via Jupyter notebooks:

- **KG Construction GLM4 Batch**: For constructing the knowledge graph (KG).
- **TFIDF Data Keywords Processing**: For extracting keywords from patient self-descriptions.

You must first execute these Jupyter notebooks. Afterwards, proceed by sequentially running the following scripts:

```bash
python KG_and_Embedding.py
python KG_fusion.py
```
These scripts perform knowledge graph construction, embedding generation, and fusion processes for each disease.

Finally, execute the main model training with:

```bash
python KG_enhanced_Prototype.py --train_ratio 1.0 --num_epochs 120 --batch_size 16 --alpha 0.5
```
- `--train_ratio`: Adjusts the proportion of the training set used (controls masking).
- `--num_epochs`: Specifies the number of training epochs.
- `--batch_size`: Sets the size of each training batch.
- `--alpha`: Controls the weight of semantic consistency loss in the total loss.

### Environment Setup:
Use the following Conda environment configuration to replicate the required dependencies:

```yaml
dependencies:
  - _libgcc_mutex=0.1=main
  - _openmp_mutex=5.1=1_gnu
  - bzip2=1.0.8=h5eee18b_6
  - ca-certificates=2024.3.11=h06a4308_0
  - ld_impl_linux-64=2.38=h1181459_1
  - libffi=3.4.4=h6a678d5_1
  - libgcc-ng=11.2.0=h1234567_1
  - libgomp=11.2.0=h1234567_1
  - libstdcxx-ng=11.2.0=h1234567_1
  - libuuid=1.41.5=h5eee18b_0
  - ncurses=6.4=h6a678d5_0
  - openssl=3.0.13=h7f8727e_2
  - pip=25.0=py311h06a4308_0
  - python=3.11.9=h955ad1f_0
  - readline=8.2=h5eee18b_0
  - setuptools=75.8.0=py311h06a4308_0
  - sqlite=3.45.3=h5eee18b_0
  - tk=8.6.14=h39e8969_0
  - wheel=0.45.1=py311h06a4308_0
  - xz=5.4.6=h5eee18b_1
  - zlib=1.2.13=h5eee18b_1
  - pip:
      - aiohappyeyeballs==2.6.1
      - aiohttp==3.11.14
      - aiosignal==1.3.2
      - annotated-types==0.7.0
      - anyio==4.9.0
      - attrs==25.3.0
      - certifi==2025.1.31
      - charset-normalizer==3.4.1
      - colorama==0.4.6
      - distro==1.9.0
      - filelock==3.17.0
      - frozenlist==1.5.0
      - fsspec==2025.3.0
      - h11==0.14.0
      - httpcore==1.0.7
      - httpx==0.28.1
      - huggingface-hub==0.29.2
      - idna==3.10
      - jinja2==3.1.6
      - jiter==0.9.0
      - joblib==1.4.2
      - markupsafe==3.0.2
      - mpmath==1.3.0
      - multidict==6.2.0
      - networkx==3.4.2
      - numpy==2.2.3
      - nvidia-cublas-cu12==12.4.5.8
      - nvidia-cuda-cupti-cu12==12.4.127
      - nvidia-cuda-nvrtc-cu12==12.4.127
      - nvidia-cuda-runtime-cu12==12.4.127
      - nvidia-cudnn-cu12==9.1.0.70
      - nvidia-cufft-cu12==11.2.1.3
      - nvidia-curand-cu12==10.3.5.147
      - nvidia-cusolver-cu12==11.6.1.9
      - nvidia-cusparse-cu12==12.3.1.170
      - nvidia-cusparselt-cu12==0.6.2
      - nvidia-nccl-cu12==2.21.5
      - nvidia-nvjitlink-cu12==12.4.127
      - nvidia-nvtx-cu12==12.4.127
      - openai==1.66.3
      - packaging==24.2
      - pandas==2.2.3
      - pillow==11.1.0
      - propcache==0.3.0
      - psutil==7.0.0
      - pydantic==2.10.6
      - pydantic-core==2.27.2
      - pyg-lib==0.4.0+pt25cu124
      - pyparsing==3.2.1
      - python-dateutil==2.9.0.post0
      - pytz==2025.1
      - pyyaml==6.0.2
      - regex==2024.11.6
      - requests==2.32.3
      - safetensors==0.5.3
      - scikit-learn==1.6.1
      - scipy==1.15.2
      - six==1.17.0
      - sniffio==1.3.1
      - sympy==1.13.1
      - threadpoolctl==3.5.0
      - tokenizers==0.21.0
      - torch==2.6.0
      - torch-cluster==1.6.3+pt25cu124
      - torch-geometric==2.6.1
      - torch-scatter==2.1.2+pt25cu124
      - torch-sparse==0.6.18+pt25cu124
      - torch-spline-conv==1.2.2+pt25cu124
      - torchaudio==2.6.0
      - torchvision==0.21.0
      - tqdm==4.67.1
      - transformers==4.49.0
      - triton==3.2.0
      - typing-extensions==4.12.2
      - tzdata==2025.1
      - urllib3==2.3.0
      - yarl==1.18.3
```
