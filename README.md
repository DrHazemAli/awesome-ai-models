# Awesome AI Models

A curated list of **awesome AI models** from various domains and frameworks. This repository aims to gather the most significant and cutting-edge AI models available in the open-source community. Whether you're a researcher, developer, or enthusiast, you'll find valuable resources here to enhance your projects.

## Table of Contents

- [Introduction](#introduction)
- [PyTorch Models](#pytorch-models)
  - [Natural Language Processing](#natural-language-processing)
  - [Computer Vision](#computer-vision)
  - [Audio Processing](#audio-processing)
  - [Reinforcement Learning](#reinforcement-learning)
- [TensorFlow Models](#tensorflow-models)
  - [Natural Language Processing](#natural-language-processing-tf)
  - [Computer Vision](#computer-vision-tf)
  - [Audio Processing](#audio-processing-tf)
  - [Reinforcement Learning](#reinforcement-learning-tf)
- [Core ML Models](#core-ml-models)
- [JAX Models](#jax-models)
- [ONNX Models](#onnx-models)
- [Contributing](#contributing)
- [License](#license)

---

## Introduction

Artificial Intelligence (AI) has revolutionized numerous fields, from natural language processing to computer vision and beyond. This repository collects some of the most impactful and widely used AI models, making it easier for you to find and utilize them in your projects.

---

## PyTorch Models

### Natural Language Processing

1. **BERT (Bidirectional Encoder Representations from Transformers)**  
   ★★★★★  
   **Impact Score**: 98  
   - **GitHub**: [huggingface/pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT)
   - **Description**: State-of-the-art pre-trained NLP model for various tasks.

2. **GPT-2 (Generative Pretrained Transformer 2)**  
   ★★★★★  
   **Impact Score**: 96  
   - **GitHub**: [openai/gpt-2](https://github.com/openai/gpt-2)
   - **Description**: Large-scale transformer-based language model.

3. **RoBERTa**  
   ★★★★☆  
   **Impact Score**: 92  
   - **GitHub**: [pytorch/fairseq](https://github.com/pytorch/fairseq/tree/main/examples/roberta)
   - **Description**: Robustly optimized BERT approach.

4. **DistilBERT**  
   ★★★★☆  
   **Impact Score**: 88  
   - **GitHub**: [huggingface/transformers](https://github.com/huggingface/transformers)
   - **Description**: A smaller, faster, cheaper version of BERT.

5. **XLNet**  
   ★★★★☆  
   **Impact Score**: 90  
   - **GitHub**: [zihangdai/xlnet](https://github.com/zihangdai/xlnet)
   - **Description**: Generalized autoregressive pretraining for language understanding.

6. **T5 (Text-to-Text Transfer Transformer)**  
   ★★★★★  
   **Impact Score**: 95  
   - **GitHub**: [google-research/text-to-text-transfer-transformer](https://github.com/google-research/text-to-text-transfer-transformer)
   - **Description**: Converts all NLP tasks into a text-to-text format.

7. **CTRL (Conditional Transformer Language Model)**  
   ★★★★☆  
   **Impact Score**: 85  
   - **GitHub**: [salesforce/ctrl](https://github.com/salesforce/ctrl)
   - **Description**: A conditional transformer language model for controllable text generation.

8. **ALBERT (A Lite BERT)**  
   ★★★★☆  
   **Impact Score**: 89  
   - **GitHub**: [google-research/albert](https://github.com/google-research/albert)
   - **Description**: A lighter version of BERT with fewer parameters.

9. **Bart**  
   ★★★★☆  
   **Impact Score**: 87  
   - **GitHub**: [pytorch/fairseq](https://github.com/pytorch/fairseq/tree/main/examples/bart)
   - **Description**: Denoising autoencoder for pretraining sequence-to-sequence models.

10. **Pegasus**  
    ★★★★☆  
    **Impact Score**: 86  
    - **GitHub**: [google-research/pegasus](https://github.com/google-research/pegasus)
    - **Description**: Pre-training with extracted gap-sentences for abstractive summarization.

11. **ELECTRA**  
    ★★★★☆  
    **Impact Score**: 88  
    - **GitHub**: [google-research/electra](https://github.com/google-research/electra)
    - **Description**: Pre-training text encoders as discriminators rather than generators.

12. **Longformer**  
    ★★★★☆  
    **Impact Score**: 84  
    - **GitHub**: [allenai/longformer](https://github.com/allenai/longformer)
    - **Description**: Transformer model for long documents.

13. **Reformer**  
    ★★★★☆  
    **Impact Score**: 83  
    - **GitHub**: [google/trax](https://github.com/google/trax)
    - **Description**: Efficient Transformer model with reduced memory consumption.

14. **Transformer-XL**  
    ★★★★☆  
    **Impact Score**: 85  
    - **GitHub**: [kimiyoung/transformer-xl](https://github.com/kimiyoung/transformer-xl)
    - **Description**: An architecture with longer-term dependency for language modeling.

15. **DialoGPT**  
    ★★★★☆  
    **Impact Score**: 82  
    - **GitHub**: [microsoft/DialoGPT](https://github.com/microsoft/DialoGPT)
    - **Description**: A large-scale pretrained dialogue response generation model.

16. **MarianMT**  
    ★★★★☆  
    **Impact Score**: 80  
    - **GitHub**: [huggingface/transformers](https://github.com/huggingface/transformers)
    - **Description**: Fast neural machine translation models.

17. **Megatron-LM**  
    ★★★★★  
    **Impact Score**: 94  
    - **GitHub**: [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
    - **Description**: Large, powerful transformer models for NLP tasks.

18. **DeBERTa**  
    ★★★★☆  
    **Impact Score**: 89  
    - **GitHub**: [microsoft/DeBERTa](https://github.com/microsoft/DeBERTa)
    - **Description**: Decoding-enhanced BERT with disentangled attention.

19. **BARTpho**  
    ★★★☆☆  
    **Impact Score**: 75  
    - **GitHub**: [VinAIResearch/BARTpho](https://github.com/VinAIResearch/BARTpho)
    - **Description**: Pre-trained sequence-to-sequence models for Vietnamese.

20. **CamemBERT**  
    ★★★★☆  
    **Impact Score**: 81  
    - **GitHub**: [camembert-model/camembert](https://github.com/camembert-model/camembert)
    - **Description**: A French language model based on RoBERTa.

### Computer Vision

1. **ResNet**  
   ★★★★★  
   **Impact Score**: 97  
   - **GitHub**: [pytorch/vision](https://github.com/pytorch/vision/tree/main/torchvision/models)
   - **Description**: Deep residual networks for image recognition.

2. **EfficientNet**  
   ★★★★☆  
   **Impact Score**: 91  
   - **GitHub**: [lukemelas/EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)
   - **Description**: Models with efficient scaling for better performance.

3. **YOLOv5**  
   ★★★★★  
   **Impact Score**: 95  
   - **GitHub**: [ultralytics/yolov5](https://github.com/ultralytics/yolov5)
   - **Description**: Real-time object detection model.

4. **Mask R-CNN**  
   ★★★★★  
   **Impact Score**: 94  
   - **GitHub**: [facebookresearch/detectron2](https://github.com/facebookresearch/detectron2)
   - **Description**: Object detection and segmentation framework.

5. **U-Net**  
   ★★★★☆  
   **Impact Score**: 88  
   - **GitHub**: [milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)
   - **Description**: Convolutional networks for biomedical image segmentation.

6. **StyleGAN2**  
   ★★★★★  
   **Impact Score**: 93  
   - **GitHub**: [NVlabs/stylegan2](https://github.com/NVlabs/stylegan2)
   - **Description**: Generative adversarial network for image synthesis.

7. **CLIP**  
   ★★★★★  
   **Impact Score**: 96  
   - **GitHub**: [openai/CLIP](https://github.com/openai/CLIP)
   - **Description**: Connects text and images in a single embedding space.

8. **DINO (Self-Distillation with No Labels)**  
   ★★★★☆  
   **Impact Score**: 89  
   - **GitHub**: [facebookresearch/dino](https://github.com/facebookresearch/dino)
   - **Description**: Unsupervised vision transformer training.

9. **Swin Transformer**  
   ★★★★☆  
   **Impact Score**: 90  
   - **GitHub**: [microsoft/Swin-Transformer](https://github.com/microsoft/Swin-Transformer)
   - **Description**: Hierarchical vision transformer using shifted windows.

10. **DeepLabV3**  
    ★★★★☆  
    **Impact Score**: 87  
    - **GitHub**: [pytorch/vision](https://github.com/pytorch/vision/tree/main/torchvision/models/segmentation)
    - **Description**: Semantic image segmentation model.

### Audio Processing

1. **WaveGlow**  
   ★★★★☆  
   **Impact Score**: 85  
   - **GitHub**: [NVIDIA/waveglow](https://github.com/NVIDIA/waveglow)
   - **Description**: Flow-based generative network for speech synthesis.

2. **Tacotron 2**  
   ★★★★☆  
   **Impact Score**: 86  
   - **GitHub**: [NVIDIA/tacotron2](https://github.com/NVIDIA/tacotron2)
   - **Description**: End-to-end speech synthesis model.

3. **Open Unmix**  
   ★★★★☆  
   **Impact Score**: 80  
   - **GitHub**: [sigsep/open-unmix-pytorch](https://github.com/sigsep/open-unmix-pytorch)
   - **Description**: Reference implementation for music source separation.

4. **DeepSpeech**  
   ★★★★☆  
   **Impact Score**: 83  
   - **GitHub**: [SeanNaren/deepspeech.pytorch](https://github.com/SeanNaren/deepspeech.pytorch)
   - **Description**: End-to-end speech recognition model.

5. **Wav2Vec 2.0**  
   ★★★★★  
   **Impact Score**: 92  
   - **GitHub**: [pytorch/fairseq](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec)
   - **Description**: Self-supervised learning of speech representations.

### Reinforcement Learning

1. **Stable Baselines3**  
   ★★★★☆  
   **Impact Score**: 88  
   - **GitHub**: [DLR-RM/stable-baselines3](https://github.com/DLR-RM/stable-baselines3)
   - **Description**: Improved implementations of reinforcement learning algorithms.

2. **RLlib**  
   ★★★★☆  
   **Impact Score**: 90  
   - **GitHub**: [ray-project/ray](https://github.com/ray-project/ray)
   - **Description**: Scalable reinforcement learning library.

3. **OpenAI Baselines**  
   ★★★★☆  
   **Impact Score**: 87  
   - **GitHub**: [openai/baselines](https://github.com/openai/baselines)
   - **Description**: High-quality implementations of RL algorithms.

4. **pytorch-a2c-ppo-acktr-gail**  
   ★★★★☆  
   **Impact Score**: 85  
   - **GitHub**: [ikostrikov/pytorch-a2c-ppo-acktr-gail](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail)
   - **Description**: PyTorch implementations of popular RL algorithms.

5. **CleanRL**  
   ★★★★☆  
   **Impact Score**: 82  
   - **GitHub**: [vwxyzjn/cleanrl](https://github.com/vwxyzjn/cleanrl)
   - **Description**: High-quality single-file implementations of Deep RL algorithms.

---

## TensorFlow Models

### Natural Language Processing

1. **BERT**  
   ★★★★★  
   **Impact Score**: 98  
   - **GitHub**: [tensorflow/models/tree/master/official/nlp/bert](https://github.com/tensorflow/models/tree/master/official/nlp/bert)
   - **Description**: TensorFlow implementation of BERT.

2. **ALBERT**  
   ★★★★☆  
   **Impact Score**: 89  
   - **GitHub**: [tensorflow/models/tree/master/official/nlp/albert](https://github.com/tensorflow/models/tree/master/official/nlp/albert)
   - **Description**: A Lite BERT for self-supervised learning of language representations.

3. **Transformer**  
   ★★★★☆  
   **Impact Score**: 90  
   - **GitHub**: [tensorflow/models/tree/master/official/transformer](https://github.com/tensorflow/models/tree/master/official/transformer)
   - **Description**: Tensor2Tensor transformer model.

4. **XLNet**  
   ★★★★☆  
   **Impact Score**: 85  
   - **GitHub**: [zhangyongshun/XLNet-TensorFlow](https://github.com/zhangyongshun/XLNet-TensorFlow)
   - **Description**: XLNet implementation in TensorFlow.

5. **ELECTRA**  
   ★★★★☆  
   **Impact Score**: 88  
   - **GitHub**: [google-research/electra](https://github.com/google-research/electra)
   - **Description**: Pre-training text encoders as discriminators rather than generators.

### Computer Vision

1. **EfficientNet**  
   ★★★★☆  
   **Impact Score**: 91  
   - **GitHub**: [tensorflow/tpu/tree/master/models/official/efficientnet](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)
   - **Description**: EfficientNet models in TensorFlow.

2. **ResNet**  
   ★★★★★  
   **Impact Score**: 97  
   - **GitHub**: [tensorflow/models/tree/master/official/vision/image_classification/resnet](https://github.com/tensorflow/models/tree/master/official/vision/image_classification/resnet)
   - **Description**: TensorFlow implementation of ResNet.

3. **YOLOv4**  
   ★★★★☆  
   **Impact Score**: 88  
   - **GitHub**: [hunglc007/tensorflow-yolov4-tflite](https://github.com/hunglc007/tensorflow-yolov4-tflite)
   - **Description**: YOLOv4 implementation in TensorFlow.

4. **DeepLab**  
   ★★★★☆  
   **Impact Score**: 87  
   - **GitHub**: [tensorflow/models/tree/master/research/deeplab](https://github.com/tensorflow/models/tree/master/research/deeplab)
   - **Description**: Semantic image segmentation model.

5. **MobileNet**  
   ★★★★☆  
   **Impact Score**: 85  
   - **GitHub**: [tensorflow/models/tree/master/research/slim/nets/mobilenet](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet)
   - **Description**: Lightweight deep neural network.

### Audio Processing

1. **WaveNet**  
   ★★★★☆  
   **Impact Score**: 86  
   - **GitHub**: [ibab/tensorflow-wavenet](https://github.com/ibab/tensorflow-wavenet)
   - **Description**: Deep generative model of raw audio waveforms.

2. **DeepSpeech**  
   ★★★★☆  
   **Impact Score**: 83  
   - **GitHub**: [mozilla/DeepSpeech](https://github.com/mozilla/DeepSpeech)
   - **Description**: Speech-to-text engine using deep learning.

3. **SpeechTransformer**  
   ★★★★☆  
   **Impact Score**: 80  
   - **GitHub**: [zhaoxin94/Transformers-in-Speech-Processing](https://github.com/zhaoxin94/Transformers-in-Speech-Processing)
   - **Description**: Transformer models for speech recognition.

4. **Sound Classification**  
   ★★★★☆  
   **Impact Score**: 78  
   - **GitHub**: [tensorflow/models/tree/master/research/audioset](https://github.com/tensorflow/models/tree/master/research/audioset)
   - **Description**: Audio event detection and classification.

5. **VoiceFilter**  
   ★★★★☆  
   **Impact Score**: 75  
   - **GitHub**: [mindslab-ai/voicefilter](https://github.com/mindslab-ai/voicefilter)
   - **Description**: Speaker-conditioned speech separation model.

### Reinforcement Learning

1. **TF-Agents**  
   ★★★★☆  
   **Impact Score**: 88  
   - **GitHub**: [tensorflow/agents](https://github.com/tensorflow/agents)
   - **Description**: Reinforcement learning library for TensorFlow.

2. **Dopamine**  
   ★★★★☆  
   **Impact Score**: 85  
   - **GitHub**: [google/dopamine](https://github.com/google/dopamine)
   - **Description**: Research framework for fast prototyping of RL algorithms.

3. **TRFL**  
   ★★★★☆  
   **Impact Score**: 82  
   - **GitHub**: [deepmind/trfl](https://github.com/deepmind/trfl)
   - **Description**: TensorFlow RL library.

4. **DeepMind Control Suite**  
   ★★★★☆  
   **Impact Score**: 80  
   - **GitHub**: [deepmind/dm_control](https://github.com/deepmind/dm_control)
   - **Description**: Set of Python libraries and Mujoco components.

5. **TensorForce**  
   ★★★★☆  
   **Impact Score**: 78  
   - **GitHub**: [tensorforce/tensorforce](https://github.com/tensorforce/tensorforce)
   - **Description**: RL library for TensorFlow.

---

## Core ML Models

1. **MobileNetV2**  
   ★★★★☆  
   **Impact Score**: 85  
   - **GitHub**: [apple/coremltools](https://github.com/apple/coremltools)
   - **Description**: MobileNetV2 models for Core ML.

2. **YOLOv3-CoreML**  
   ★★★★☆  
   **Impact Score**: 82  
   - **GitHub**: [hollance/YOLO-CoreML-MPSNNGraph](https://github.com/hollance/YOLO-CoreML-MPSNNGraph)
   - **Description**: YOLOv3 implemented in Core ML.

3. **Style Transfer**  
   ★★★★☆  
   **Impact Score**: 80  
   - **GitHub**: [tucan9389/Core-ML-Examples](https://github.com/tucan9389/Core-ML-Examples)
   - **Description**: Core ML models for style transfer.

4. **Handwriting Recognition**  
   ★★★★☆  
   **Impact Score**: 78  
   - **GitHub**: [ymatthes/apple-coreml-sample](https://github.com/ymatthes/apple-coreml-sample)
   - **Description**: Handwriting recognition with Core ML.

5. **DeepLabV3-CoreML**  
   ★★★★☆  
   **Impact Score**: 75  
   - **GitHub**: [asefmahi/DeepLabV3-Plus-CoreML](https://github.com/asefmahi/DeepLabV3-Plus-CoreML)
   - **Description**: Semantic segmentation with DeepLabV3+ in Core ML.

---

## JAX Models

1. **Flax Vision Transformers**  
   ★★★★☆  
   **Impact Score**: 88  
   - **GitHub**: [google-research/vision_transformer](https://github.com/google-research/vision_transformer)
   - **Description**: Implementation of Vision Transformers in JAX/Flax.

2. **JAX MD**  
   ★★★★☆  
   **Impact Score**: 82  
   - **GitHub**: [google/jax-md](https://github.com/google/jax-md)
   - **Description**: Machine learning for molecular dynamics simulations.

3. **Neural Tangents**  
   ★★★★☆  
   **Impact Score**: 80  
   - **GitHub**: [google/neural-tangents](https://github.com/google/neural-tangents)
   - **Description**: Fast and easy computing of neural tangent kernels.

4. **Flow Matching Models**  
   ★★★★☆  
   **Impact Score**: 78  
   - **GitHub**: [google-research/flow-matchings](https://github.com/google-research/flow-matchings)
   - **Description**: Generative modeling with flow matching.

5. **JAX RL**  
   ★★★★☆  
   **Impact Score**: 75  
   - **GitHub**: [rlworkgroup/garage](https://github.com/rlworkgroup/garage)
   - **Description**: Reinforcement learning algorithms in JAX.

---

## ONNX Models

1. **ONNX Model Zoo**  
   ★★★★★  
   **Impact Score**: 95  
   - **GitHub**: [onnx/models](https://github.com/onnx/models)
   - **Description**: Pre-trained state-of-the-art models in ONNX format.

2. **BERT-ONNX**  
   ★★★★★  
   **Impact Score**: 93  
   - **GitHub**: [onnx/models/tree/master/text/machine_comprehension/bert-squad](https://github.com/onnx/models/tree/master/text/machine_comprehension/bert-squad)
   - **Description**: BERT model converted to ONNX.

3. **ResNet-ONNX**  
   ★★★★★  
   **Impact Score**: 94  
   - **GitHub**: [onnx/models/tree/master/vision/classification/resnet](https://github.com/onnx/models/tree/master/vision/classification/resnet)
   - **Description**: ResNet models in ONNX format.

4. **GPT-2-ONNX**  
   ★★★★★  
   **Impact Score**: 92  
   - **GitHub**: [onnx/models/tree/master/text/machine_comprehension/gpt-2](https://github.com/onnx/models/tree/master/text/machine_comprehension/gpt-2)
   - **Description**: GPT-2 model in ONNX.

5. **YOLOv3-ONNX**  
   ★★★★☆  
   **Impact Score**: 88  
   - **GitHub**: [onnx/models/tree/master/vision/object_detection_segmentation/yolov3](https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/yolov3)
   - **Description**: YOLOv3 model in ONNX format.

---

## Contributing

Contributions are welcome! You can open an issue or submit a pull request to add new models or improve existing ones.

---

## License

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

This repository is licensed under the MIT License.

---

*Made with ❤️ by [DrHazemAli](https://github.com/DrHazemAli)*

---

*Note: The ranking stars and impact scores are based on the models' popularity, performance, and influence in the AI community.*
